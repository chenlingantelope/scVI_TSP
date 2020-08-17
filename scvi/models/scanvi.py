from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, kl_divergence as kl
from torch.nn.functional import one_hot

from scvi.models.classifier import Classifier
from scvi.models.modules import Decoder, Encoder
from scvi.models.utils import broadcast_labels
from scvi.models.vae import VAE


class SCANVI(VAE):
    r"""A semi-supervised Variational auto-encoder model - inspired from M1 + M2 model,
    as described in (https://arxiv.org/pdf/1406.5298.pdf). SCANVI stands for single-cell annotation using
    variational inference.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param y_prior: If None, initialized to uniform probability over cell types
    :param ontology: cell ontology adjacency matrix
    :param use_ontology: Whether to use the ontology
    :param kl_symmetric: Whether to use symmetric KL distance

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> scanvi = SCANVI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> gene_dataset = SyntheticDataset(n_labels=3)
        >>> scanvi = SCANVI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=3, y_prior=torch.tensor([[0.1,0.5,0.4]]), labels_groups=[0,0,1])
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        y_prior=None,
        ontology: np.array = None,
        use_ontology: bool = False,
        full_hierarchy: bool = True,
        classifier_parameters: dict = dict(),
        symmetric_kl: bool = False,
        provide_onto_info: bool = False,
    ):
        super().__init__(
            n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            n_batch=n_batch,
            dispersion=dispersion,
            log_variational=log_variational,
            reconstruction_loss=reconstruction_loss,
        )

        self.provide_onto_info = provide_onto_info
        self.n_labels = n_labels
        self.symmetric_kl = symmetric_kl
        # Classifier takes n_latent as input
        cls_parameters = {
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "dropout_rate": dropout_rate,
        }
        cls_parameters.update(classifier_parameters)
        self.classifier = Classifier(n_latent, n_labels=n_labels, **cls_parameters)
        self.use_ontology = use_ontology
        if self.use_ontology:
            assert ontology is not None, "Specify ontology"
            self.ontology = ontology
            self.depth = len(ontology) + 1
            assert "number of layers has to be greater than 1", self.depth > 1
            self.classifiers = torch.nn.ModuleList()

            use_logits = full_hierarchy
            for x in ontology:
                self.classifiers.append(
                    Classifier(
                        n_latent,
                        n_labels=x.shape[0],
                        logits=use_logits,
                        **cls_parameters
                    )
                )
            self.classifiers.append(
                Classifier(
                    n_latent, n_labels=n_labels, logits=use_logits, **cls_parameters
                )
            )
            ancestral_dims = 0
            for a in self.ontology:
                ancestral_dims += len(a)
            self.dims_ancestral = ancestral_dims
            self.dim_leaves = self.ontology[-1].shape[-1]

        # parent_ids = []
        # for a in self.ontology:
        #     parent_id = torch.cuda.ByteTensor([np.where(col == 1)[0][0] for col in a])
        #     parent_ids.append(parent_id)
        # self.parent_ids = parent_ids
        if provide_onto_info:
            n_encoder_z2_z1 = n_latent + self.dims_ancestral
            n_decoder_z1_z2 = n_latent + self.dims_ancestral
        else:
            n_encoder_z2_z1 = n_latent
            n_decoder_z1_z2 = n_latent

        self.encoder_z2_z1 = Encoder(
            n_encoder_z2_z1,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.decoder_z1_z2 = Decoder(
            n_decoder_z1_z2,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

        self.y_prior = torch.nn.Parameter(
            y_prior
            if y_prior is not None
            else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )

        self.full_hierarchy = full_hierarchy

    def hiearchical_classifier(self, z, depth):
        if not self.full_hierarchy:
            # classifier = self.classifiers[depth - 1]
            if depth == 1:
                w_y = self.classifiers[depth - 1](z)
                w_y = (w_y + 1e-16) / (w_y + 1e-16).sum(-1, keepdim=True)
                w_y = w_y.log()
            else:
                unw_y = self.classifiers[depth - 1](z)
                unw_y = (unw_y + 1e-16) / (unw_y + 1e-16).sum(-1, keepdim=True)
                unw_y = unw_y.log()

                w_g = self.hiearchical_classifier(z, depth - 1)

                # Using matrix
                A = torch.cuda.FloatTensor(self.ontology[depth - 2])
                w_y = unw_y + torch.matmul(w_g, A)
                w_y = torch.where(
                    torch.isnan(w_y), float("-inf") * torch.ones_like(w_y), w_y
                )

                if torch.isinf(w_y).any():
                    print("inf value in classifiers")
                if torch.isnan(w_y).any():
                    print("NaN value in classifiers")
                w_y = w_y - torch.logsumexp(w_y, -1, keepdim=True)
            return w_y

        else:
            # classifier = self.classifiers[depth - 1]
            if depth == 1:
                w_y = self.classifiers[depth - 1](z)
                w_y = nn.LogSoftmax(-1)(w_y)
                # w_y = (w_y + 1e-16) / (w_y + 1e-16).sum(-1, keepdim=True)
                # w_y = w_y.log()
            else:
                n_batch = len(z)
                A = torch.cuda.FloatTensor(self.ontology[depth - 2])
                n_parent, n_children = A.shape
                reverse_mask = 1.0 - A
                offsetter = reverse_mask.masked_fill(reverse_mask.bool(), float("-inf"))

                # Should be logits, a regular softmax will mess things up
                _unw_y = self.classifiers[depth - 1](z)
                _w = A * _unw_y.unsqueeze(1)
                _logits = _w + offsetter
                _logprobs = nn.LogSoftmax(-1)(_logits)
                idx = (
                    A.argmax(0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand([n_batch, 1, n_children])
                )
                unw_y = torch.gather(_logprobs, 1, index=idx).squeeze(1)

                # Partial reweighting

                w_g = self.hiearchical_classifier(z, depth - 1)

                # Using matrix
                w_y = unw_y + torch.matmul(w_g, A)
                w_y = torch.where(
                    torch.isnan(w_y), float("-inf") * torch.ones_like(w_y), w_y
                )

                if torch.isinf(w_y).any():
                    print("inf value in classifiers")
                if torch.isnan(w_y).any():
                    print("NaN value in classifiers")

                # print("_unw_y", _unw_y.shape)
                # print("_w", _w.shape)
                # print("_logits", _logits.shape)
                # print("_logprobs", _logprobs.shape)
                # print("w_y", w_y.shape)
                # print("unw_y", unw_y.shape)
                # print("w_g", w_g.shape)
                # print("idx", idx.shape)

                # Safeguard logsumexp, but should not be needed in theory
                w_y = w_y - torch.logsumexp(w_y, -1, keepdim=True)
        return w_y

    def classify(self, x, full_predictions=False, depth=None):
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, _, z = self.z_encoder(x)
        z = qz_m  # We classify using the inferred mean parameter of z_1 in the latent space
        if self.use_ontology:
            # if full_predictions:
            #     w_ys = []
            #     for dep in range(1, self.depth + 1):
            #         w_yi = self.hiearchical_classifier(z, dep).exp()
            #         w_ys.append(w_yi)
            #     return w_ys
            if depth is None:
                depth = self.depth
            w_y = self.hiearchical_classifier(z, depth=depth).exp()
        else:
            w_y = self.classifier(z)
        return w_y

    def get_latents(self, x, y=None):
        zs = super().get_latents(x)
        qz2_m, qz2_v, z2 = self.encoder_z2_z1(zs[0], y)
        if not self.training:
            z2 = qz2_m
        return [zs[0], z2]

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        is_labelled = False if y is None else True

        outputs = self.inference(x, batch_index, y)
        px_r = outputs["px_r"]
        px_rate = outputs["px_rate"]
        px_dropout = outputs["px_dropout"]
        qz1_m = outputs["qz_m"]
        qz1_v = outputs["qz_v"]
        z1 = outputs["z"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]

        # Enumerate choices of label
        ys, z1s = broadcast_labels(y, z1, n_broadcast=self.n_labels)
        if self.provide_onto_info:
            onto_info_oh = self.obtain_onto_info(ys)
            _z1s = torch.cat([z1s, onto_info_oh], -1)
            qz2_m, qz2_v, z2 = self.encoder_z2_z1(_z1s, ys)
            _z2 = torch.cat([z2, onto_info_oh], -1)
            pz1_m, pz1_v = self.decoder_z1_z2(_z2, ys)

        else:
            qz2_m, qz2_v, z2 = self.encoder_z2_z1(z1s, ys)
            pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        # KL Divergence
        mean = torch.zeros_like(qz2_m)
        scale = torch.ones_like(qz2_v)

        kl_divergence_z2 = kl(
            Normal(qz2_m, torch.sqrt(qz2_v)), Normal(mean, scale)
        ).sum(dim=1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = Normal(qz1_m, torch.sqrt(qz1_v)).log_prob(z1).sum(dim=-1)
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)
        probs = self.classify(x)

        if is_labelled:
            return (
                reconst_loss + loss_z1_weight + loss_z1_unweight,
                kl_divergence_z2 + kl_divergence_l,
                0.0,
            )

        probs = self.classify(x)
        reconst_loss += loss_z1_weight + (
            (loss_z1_unweight).view(self.n_labels, -1).t() * probs
        ).sum(dim=1)

        kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(
            dim=1
        )
        kl_c = kl(
            Categorical(probs=probs),
            Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
        )

        if self.symmetric_kl:
            kl_c = 0.5 * kl_c + 0.5 * kl(
                Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
                Categorical(probs=probs),
            )

        kl_divergence += kl_c

        kl_divergence += kl_divergence_l

        return reconst_loss, kl_divergence, 0.0

    def obtain_onto_info(self, y):
        all_lbls = [y.byte().argmax(-1).view(-1)]
        for a in self.ontology[::-1]:
            A = torch.cuda.FloatTensor(a)
            new_labels = A[:, all_lbls[-1].squeeze()].argmax(0)
            all_lbls.append(new_labels)
        # leaves, higher ... top
        # top, ..., higher, leaves
        all_lbls = all_lbls[::-1]
        # higher ... top
        all_lbls = all_lbls[:-1]

        onto_info_oh = []
        for lbl, a in zip(all_lbls, self.ontology):
            # print(a.shape)
            # print(lbl.max())
            onto_info_oh.append(one_hot(lbl, len(a)))
        onto_info_oh = torch.cat(onto_info_oh, -1)
        return onto_info_oh.float()
