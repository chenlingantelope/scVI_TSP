import numpy as np

from scvi.dataset import SyntheticDataset
from scvi.inference import JointSemiSupervisedTrainer

from scvi.models import SCANVI
from scvi import set_seed

set_seed(0)
use_cuda = True



def test_hierarchy():
    synthetic_dataset = SyntheticDataset()
    svaec = SCANVI(
        synthetic_dataset.nb_genes,
        synthetic_dataset.n_batches,
        synthetic_dataset.n_labels,
        ontology=[np.array([[1,1,0],[0,0,1]]), np.array([[1,0,1,0],[0,0,1,0], [0,0,1,1]])],
        use_ontology=True,
        reconstruction_loss="zinb",
        n_layers=3,
    )
    trainer_synthetic_svaec = JointSemiSupervisedTrainer(
        svaec, synthetic_dataset, use_cuda=use_cuda
    )
    trainer_synthetic_svaec.train(n_epochs=1)


test_hierarchy()
