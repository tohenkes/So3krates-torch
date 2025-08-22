from so3krates_torch.tools.eval import test_model, test_ensemble
from ase.io import read
import torch
from so3krates_torch.tools.utils import create_dataloader
from mace.data.utils import KeySpecification

device = 'cuda'

data = read("example/aqm_small.xyz", index=":")
model = torch.load(
    f="src/so3krates_torch/pretrained/so3lr/so3lr.model",
    map_location=device,
    weights_only=False,
)
model2 = torch.load(
    f="src/so3krates_torch/pretrained/so3lr/so3lr.model",
    map_location=device,
    weights_only=False,
)

model.r_max_lr = 12.0
model.dispersion_energy_cutoff_lr_damping = 2.0
model2.r_max_lr = 12.0
model2.dispersion_energy_cutoff_lr_damping = 2.0

ensemble = {
    "model1": model,
    "model2": model2
}


keyspec = KeySpecification(
    info_keys={
        "energy": "REF_energy",
        "dipole": "REF_dipole"
    },
    arrays_keys={
        "hirshfeld_ratios": "REF_hirsh_ratios",
        "forces": "REF_forces",
    }
) 

results = test_ensemble(
    ensemble=ensemble,
    batch_size=5,
    output_args={
        "energy": True,
        "forces": True,
        "stress": False,
        "virials": True,
        "dipole_vec": True,
        "hirshfeld_ratios": True,
    },
    device=device,
    atoms_list=data,
    r_max_lr=12.0,
)
exit()