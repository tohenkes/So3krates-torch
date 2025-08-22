from so3krates_torch.calculator.so3 import TorchkratesCalculator, SO3LRCalculator

from mlff.calculators.ase_calculator import AseCalculatorSparse
from ase.io import read
import jax
import torch

mol = read('So3krates-torch/example/water_64.xyz')
mol = read('So3krates-torch/example/ala15.xyz')

charge = 3
mol.info["total_charge"] = charge
mol.info["total_spin"] = 0.5
mol.info["charge"] = charge
mol.info["multiplicity"] = 2 * mol.info["total_spin"] + 1

compute_stress = False
r_max_lr = 12.0
dispersion_energy_cutoff_lr_damping = 2.0
device="cpu"

dtype_str = "float64"

if dtype_str == "float64":
    jax.config.update("jax_enable_x64", True)
else:
    jax.config.update("jax_enable_x64", False)

jax_calc = AseCalculatorSparse.create_from_workdir(
    workdir="So3krates-torch/example/so3lr",
    from_file=True,
    calculate_stress=compute_stress,
    lr_cutoff=r_max_lr,
    dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping
)

torch_calc = TorchkratesCalculator(
    model_paths="/home/tobias/Uni/Promotion/Research/torchkrates/So3krates-torch/src/so3krates_torch/pretrained/so3lr/so3lr.model",
    r_max_lr=r_max_lr,
    compute_stress=compute_stress,
    dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
    device=device,
    default_dtype=dtype_str,
)

so3lr_calc = SO3LRCalculator(
    r_max_lr=r_max_lr,
    compute_stress=compute_stress,
    dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
    device=device,
    default_dtype=dtype_str,
)




torch_calc.calculate(mol, properties=['energy', 'forces', 'stress'])
so3lr_calc.calculate(mol, properties=['energy', 'forces', 'stress'])
#### TEST
with jax.disable_jit():
        jax_calc.calculate(mol, properties=['energy', 'forces', 'stress'])

#### RESULTS
print("\n\n")
print("##############  RESULTS  ##############")
print("Energy")
print(f"Torch calc version: {torch_calc.results['energy']:.6f}")
print(f"SO3LR calc version: {so3lr_calc.results['energy']:.6f}")
print(f"JAX version  : {jax_calc.results['energy']:.6f}")

print("\n\n")
print("Forces")
print("TORCH version: ", torch_calc.results['forces'][0,:])    
print("SO3LR version: ", so3lr_calc.results['forces'][0,:])    
print("JAX version  : ", jax_calc.results['forces'][0,:])

if compute_stress:
    print("\n\n")
    print("Stress")
    print("TORCH version: ", torch_calc.results['stress'])
    print("SO3LR version: ", so3lr_calc.results['stress'])
    print("JAX version  : ", jax_calc.results['stress'])

    print("\n\n")