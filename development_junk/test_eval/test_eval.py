from so3krates_torch.tools.eval import evaluate_model, ensemble_prediction
from ase.io import read
import torch
import numpy as np
import h5py

def save_results_hdf5(
    results,
    filename,
    is_ensemble: bool = False
):
    with h5py.File(filename, 'w') as f:
        for k, v in results.items():
            if v is not None:
                if is_ensemble:
                    for i, ensemble_result in enumerate(v):
                        model_id = i
                        grp = f.create_group(f"{k}/model_{model_id}")
                        for j, result in enumerate(ensemble_result):
                            grp.create_dataset(f"item_{j:06d}", data=result)

                else:
                    grp = f.create_group(k)
                    for i, result in enumerate(v):
                        grp.create_dataset(f"item_{i:06d}", data=result)

            else:
                # Store None as an empty dataset with attribute
                dset = f.create_dataset(k, data=np.array([]))
                dset.attrs['is_none'] = True


device = 'cuda'

data = read("example/md17_test_small.xyz", index=":")
multispecies = True
#data = read("example/md17_ethanol_small.xyz", index=":")


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
models = [model, model2]


if len(models) == 1:
    results = evaluate_model(
        atoms_list=data,
        model=model,
        batch_size=4,
        device=device,
        model_type="so3lr",
        r_max_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
        dtype="float32",
        multi_species=multispecies,
        compute_stress=True,
        compute_dipole=True,
        compute_hirshfeld=True,
        compute_partial_charges=True,
    )
else:
    results = ensemble_prediction(
        models=models,
        atoms_list=data,
        device=device,
        model_type="so3lr",
        dtype="float32",
        batch_size=4,
        multi_species=multispecies,
        r_max_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
        compute_stress=True,
        compute_hirshfeld=True,
        compute_dipole=True,
        compute_partial_charges=True,
    )

save_results_hdf5(results, "results.h5", is_ensemble=len(models) > 1)

exit()
loaded_data = {}
with h5py.File("results.h5", 'r') as f:
    if len(models) > 1:
        for key in f.keys():
            loaded_data[key] = {}
            if isinstance(f[key], h5py.Group):
                for i in range(len(models)):
                    model_name = f'model_{i}'
                    loaded_data[key][model_name] = [
                        f[key][model_name][item][()] for item in f[key][model_name].keys()
                    ]

    else:
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                loaded_data[key] = [f[key][item][()] for item in f[key].keys()]
            else:
                loaded_data[key] = f[key][()]

