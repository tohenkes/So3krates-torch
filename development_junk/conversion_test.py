from so3krates_torch.tools.jax_torch_conversion import (
    convert_flax_to_torch,
    convert_torch_to_flax
)
from mlff.calculators.ase_calculator import AseCalculatorSparse
from ase.io import read
import jax
from mace.tools import torch_geometric, utils
import torch
from mace import data
from ase.io import read
from mlff.calculators.ase_calculator import AseCalculatorSparse
import numpy as np
from ase.stress import full_3x3_to_voigt_6_stress
from so3krates_torch.data.atomic_data import AtomicData as So3Data
import pickle
import yaml
import json

#### TEST GEO
mol = read('So3krates-torch/example/water_64.xyz')
mol = read('So3krates-torch/example/ala15.xyz')

charge = 3
mol.info["total_charge"] = charge
mol.info["total_spin"] = 0.5
mol.info["charge"] = charge
mol.info["multiplicity"] = 2 * mol.info["total_spin"] + 1
num_unpaired_electrons = mol.info["total_spin"] * 2

#### SETTINGS
compute_stress = False
r_max=4.5
r_max_lr = 12.0
dispersion_energy_cutoff_lr_damping = 2.0
device="cpu"
compute = True

dtype_str = "float64"
if dtype_str == "float32":
    dtype = torch.float32
else:
    dtype = torch.float64

torch.set_default_dtype(dtype)
if dtype == torch.float64:
    jax.config.update("jax_enable_x64", True)
else:
    jax.config.update("jax_enable_x64", False)


z_table = utils.AtomicNumberTable(
            [int(z) for z in range(1, 119)]
        )
keyspec = data.KeySpecification(
    info_keys={"total_charge":"total_charge","total_spin":"total_spin"}, arrays_keys={"charges": "Qs"}
)

config = data.config_from_atoms(
    mol, key_specification=keyspec, head_name="default"
)
data_loader = torch_geometric.dataloader.DataLoader(
    dataset=20*[
        So3Data.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
            cutoff_lr=r_max_lr
        )
    ],
    batch_size=1,
    shuffle=False,
    drop_last=False,
)
batch = next(iter(data_loader))



#### MODELS
model = convert_flax_to_torch(
    path_to_flax_params="So3krates-torch/example/so3lr/params.pkl",
    path_to_flax_hyperparams="So3krates-torch/example/so3lr/hyperparameters.json",
    so3lr=True,
    device="cpu",
    use_defined_shifts=False,
    trainable_rbf=False,
    dtype=dtype,
    save_torch_settings="So3krates-torch/example/so3lr/torch_model_settings.yaml"
)
model.r_max_lr = r_max_lr
model.dispersion_energy_cutoff_lr_damping = dispersion_energy_cutoff_lr_damping

jax_calc = AseCalculatorSparse.create_from_workdir(
    workdir="So3krates-torch/example/so3lr",
    from_file=True,
    calculate_stress=compute_stress,
    lr_cutoff=r_max_lr,
    dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping
)

model.to(device).eval()
batch_torch_model = batch.to(device)
batch_torch_model = batch_torch_model.to_dict()
batch_torch_model["positions"].requires_grad_(True)
result = model(batch_torch_model,
               compute_virials=True,
               compute_stress=compute_stress)


with open("So3krates-torch/example/so3lr/torch_model_settings.yaml", "r") as f:
    torch_settings = yaml.safe_load(f)

cfg, params = convert_torch_to_flax(
    torch_state_dict=model.state_dict(),
    torch_settings=torch_settings,
    dtype=dtype_str,
)

cfg_dict = cfg.to_dict()
# save json
with open("So3krates-torch/example/so3lr_torch/hyperparameters.json", "w") as f:
    json.dump(cfg_dict, f)
with open("So3krates-torch/example/so3lr_torch/params.pkl", "wb") as f:
    pickle.dump(params, f)

jax_calc2 = AseCalculatorSparse.create_from_workdir(
    workdir="/home/tobias/Uni/Promotion/Research/torchkrates/So3krates-torch/example/so3lr/test2",
    from_file=True,
    calculate_stress=compute_stress,
    lr_cutoff=r_max_lr,
    dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping
)

#### TEST
with jax.disable_jit():
    if compute:
        if compute_stress:
            jax_calc.calculate(mol, properties=['energy', 'forces', 'stress'],)
            jax_calc2.calculate(mol, properties=['energy', 'forces', 'stress'],)
        else:
            jax_calc.calculate(mol, properties=['energy', 'forces'],)
            jax_calc2.calculate(mol, properties=['energy', 'forces'],)

#### RESULTS
print("\n\n")
print("##############  RESULTS  ##############")
print("Energy")
print(f"TORCH version: {result['energy'].item():.6f}")
if compute:
    print(f"JAX version  : {jax_calc.results['energy']:.6f}")
    print(f"JAX version2  : {jax_calc2.results['energy']:.6f}")
    print(f"Difference   : {jax_calc.results['energy'] - result['energy'].item():.6f}")

print("\n\n")
print("Forces")
print("TORCH version: ", result['forces'][0,:])    
print("JAX version  : ", jax_calc.results['forces'][0,:])
print("JAX version2  : ", jax_calc2.results['forces'][0,:])
force_diff = np.abs(jax_calc.results['forces'][0,:] - result['forces'][0,:].cpu().numpy())
print(f"Difference   :  {force_diff.mean().item():.6f}")

if compute_stress:
    print("\n\n")
    print("Stress")
    torch_stress = full_3x3_to_voigt_6_stress(
        result["stress"].detach().cpu().numpy()
    )
    print("TORCH version: ", torch_stress)
    print("JAX version  : ", jax_calc.results['stress'])
    print("JAX version2  : ", jax_calc2.results['stress'])
    stress_diff = np.abs(jax_calc.results['stress'] - torch_stress)
    print(f"Difference   :  {stress_diff.mean().item():.6f}")

    print("\n\n")