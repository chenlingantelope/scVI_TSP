import torch

from anndata import read_h5ad
import pandas as pd
import pickle as pkl
import os
import numpy as np

from scvi.dataset import SyntheticDataset, AnnDatasetFromAnnData
from scvi.inference import SemiSupervisedTrainer

from scvi.models import SCANVI
from scvi import set_seed

set_seed(0)
use_cuda = True

DATA_PATH = "/data/yosef2/users/pierreboyeau/scVI_TSP/hier_data"

ontology_dict = pd.read_pickle(os.path.join(DATA_PATH, "ontology_dict.pkl"))
with open(os.path.join(DATA_PATH, "adjm.pkl"), "rb") as f:
    adjm = pkl.load(f)
combined = read_h5ad(os.path.join(DATA_PATH, "anndata.h5"))


def get_leaf_sampling_probs(
    all_to_leaves, id_to_label, label_to_id, leaves_counts,
):
    nodes_to_leaves_probs = torch.zeros(
        len(id_to_label), len(leaves_counts), device="cuda"
    )
    for idx, label in id_to_label.items():
        leaves = all_to_leaves[label]
        for leaf in leaves:
            nodes_to_leaves_probs[idx, label_to_id[leaf]] = leaves_counts[leaf]
    nodes_to_leaves_probs = nodes_to_leaves_probs / nodes_to_leaves_probs.sum(
        1, keepdims=True
    )
    return nodes_to_leaves_probs


where_query = combined.obs["free_annotation"] != "nan"

leaf_labels = (
    ontology_dict[combined.obs["ann"]]
    .reset_index(drop=True)
    .str.split("_", expand=True)
    .loc[:, 0]
)

leaf_labels.loc[where_query.values] = "test"

leaf_labels.head()

final_labels = [lbl for lbl in leaf_labels.unique() if lbl != "test"]

where_labelled_leaf = np.where(~np.isin(leaf_labels, ["test", "unassigned"]))[0]
where_change_nodes = np.random.choice(where_labelled_leaf, 5000)

## Capturing the tree structure

### Mapping nodes to list of leaves

# Create dictionnary of adjm to final labels
all_cts = [list(ser.index.values) + list(ser.columns.values) for ser in adjm]
all_cts_f = []
for vals in all_cts:
    all_cts_f += vals
all_cts_f = np.unique(all_cts_f)
print(len(all_cts_f))
all_cts_f[:3]

to_split = pd.Series({val: val.split("_")[0] for val in all_cts_f})
print(len(to_split.unique()))

leaf_to_parents = adjm[-1].apply(lambda x: x[lambda x: x == 1].index.values).squeeze()
parents_to_leaves = (
    adjm[-1].apply(lambda x: x[lambda x: x == 1].index.values, axis=1).squeeze()
)
print(parents_to_leaves.shape, leaf_to_parents.shape)
print(parents_to_leaves.sample(5))

leaf_to_leaf = pd.Series([[lab] for lab in final_labels], index=final_labels)
# Step 1: mapping nodes to leaves list
all_to_leaves = pd.concat([leaf_to_leaf, parents_to_leaves])

### OPTIONAL: Constructing labels that include parent nodes


expanded_labels = leaf_labels.copy()
expanded_labels.iloc[where_change_nodes] = leaf_to_parents[
    leaf_labels.iloc[where_change_nodes]
].values

leaf_labels.unique()

### Mapping nodes to integer representation

all_labels = expanded_labels.unique()
non_leaves_labels = all_labels[~np.isin(all_labels, final_labels)]

leaves_labels = [lbl for lbl in final_labels if not lbl in ["test", "unassigned"]]
assert set(leaves_labels) == set(adjm[-1].columns)
non_leaves_labels = [
    lbl for lbl in non_leaves_labels if not lbl in ["test", "unassigned"]
]

# full_labels = ["test"] + list(final_labels) + ["unassigned"] + list(non_leaves_labels)
full_labels = list(adjm[-1].columns) + ["unassigned"] + list(non_leaves_labels)

id_to_label = pd.Series(
    full_labels,
    #     index=-1 + np.arange(len(full_labels))
    index=np.arange(len(full_labels)),
)
label_to_id = pd.Series(id_to_label.index, index=id_to_label)

## Constructing node to leaves sampling matrix

leaves_counts = leaf_labels.groupby(leaf_labels).size()
# Step 3: Get sampling prababilities to use during training
nodes_to_leaves_probs = get_leaf_sampling_probs(
    all_to_leaves=all_to_leaves,
    id_to_label=id_to_label,
    label_to_id=label_to_id,
    leaves_counts=leaves_counts,
)

## Dataset construction

a_labels = label_to_id[expanded_labels].values
# a_labels[np.isnan(a_labels)] = -1
a_labels[np.isnan(a_labels)] = len(final_labels) - 1

dataset = AnnDatasetFromAnnData(combined, batch_label="batch")
dataset.cell_types = label_to_id.index.values
dataset.labels = a_labels
dataset.n_labels = len(dataset.cell_types)
n_leaves_labels = len(final_labels)
dataset.filter_genes_by_count(per_batch=True)
dataset.subsample_genes(new_n_genes=500)
dataset.labels = a_labels
dataset.n_labels = len(dataset.cell_types)

# need final label estimates
# leaves_estimates = db.Categorical(nodes_to_leaves_probs[a_labels]).sample().cpu().numpy()
leaf_labels_ids = label_to_id[leaf_labels].values
leaf_labels_ids[np.isnan(leaf_labels_ids)] = -1

adjmc = []
for x in adjm:
    a = x.shape[0]
    b = x.shape[1]
    temp = np.zeros((a + 1, b + 1))
    temp[:a, :b] = np.asarray(x)
    temp[a, b] = 1
    adjmc.append(temp)

dataset.labels

indices_labelled = np.where(~where_query.values)[0]
indices_unlabelled = np.where(where_query.values)[0]

print("indices_labelled: ", np.unique(dataset.labels[indices_labelled]))
print("indices_unlabelled: ", np.unique(dataset.labels[indices_unlabelled]))


def test_hierarchy():
    ### Model training

    scanvi = SCANVI(
        dataset.nb_genes,
        dataset.n_batches,
        n_leaves_labels,
        n_layers=2,
        n_latent=30,
        symmetric_kl=True,
        use_ontology=True,
        ontology=adjmc,
    )
    scanvi.cuda()

    trainer_scanvi = SemiSupervisedTrainer(
        scanvi,
        dataset,
        n_labels_final=n_leaves_labels,
        labels_of_use=leaf_labels_ids,
        n_epochs_classifier=100,
        lr_classification=5e-3,
        seed=1,
        n_epochs_kl_warmup=1,
        nodes_to_leaves_probs=nodes_to_leaves_probs,
        indices_labelled=indices_labelled,
        indices_unlabelled=indices_unlabelled,
    )

    trainer_scanvi.train()
