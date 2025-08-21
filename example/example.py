from so3krates_torch.data.atomic_data import AtomicData as So3Data
from so3krates_torch.modules.models import So3krates, SO3LR
from mace.tools import torch_geometric, utils
from ase.build import molecule
import torch
from mace import data
import time
from e3nn.util import jit
from ase.io import read
from mace.modules.utils import compute_avg_num_neighbors
from mace.calculators import mace_mp
from so3krates_torch.tools.compile import prepare
import numpy as np
from math import inf
import sys

def nan_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN in module: {module.__class__.__name__}")

mol = molecule('H2O')
mol = read('So3krates-torch/example/water_64.xyz')
mol = read('So3krates-torch/example/ala15.xyz')
r_max = 4.5
#r_max_lr = float(10e6)
r_max_lr = 10.0
charge = 3
mol.info["total_charge"] = charge
mol.info["total_spin"] = 0.5
mol.info["charge"] = charge
mol.info["multiplicity"] = 2 * mol.info["total_spin"] + 1
num_unpaired_electrons = mol.info["total_spin"] * 2

z_table = utils.AtomicNumberTable(
            [int(z) for z in range(1, 119)]
        )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Using device: {device}")

dtype = 'float32'  # or torch.float64, depending on your model's requirements
torch.set_default_dtype(getattr(torch, dtype))

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
            cutoff_lr=r_max_lr,
        )
    ],
    batch_size=1,
    shuffle=False,
    drop_last=False,
)
avg_num_neighbors = compute_avg_num_neighbors(
    data_loader
)

batch = next(iter(data_loader)).to(device)

print(batch['edge_index'][0].shape)
print(batch['edge_index_lr'][0].shape)

#mace_mp_model_medium= mace_mp()
#mace_mp_model_small = mace_mp(model='small')

degrees = [1,2,3,4]
use_so3krates = False  # Set to False to use SO3LR
if use_so3krates:
    model = So3krates(
        r_max=r_max,
        num_radial_basis=32,
        degrees=degrees,
        features_dim=132,
        num_att_heads=4,
        final_mlp_layers=2,  # TODO: check, does the last layer count?
        num_interactions=1,
        num_elements=len(z_table),
        avg_num_neighbors=avg_num_neighbors,
        message_normalization='avg_num_neighbors',
        seed=42,
        device=device,
        trainable_rbf=True,
        learn_atomic_type_shifts=True,
        learn_atomic_type_scales=True,
        energy_regression_dim=66,  # This is the energy regression dimension
        layer_normalization_1=True,
        layer_normalization_2=True,
        residual_mlp_1=True,
        residual_mlp_2=True,
        use_charge_embed=True,
        use_spin_embed=True,
    )
else:
    model = SO3LR(
        r_max=r_max,
        r_max_lr=r_max_lr,
        num_radial_basis=32,
        degrees=degrees,
        features_dim=132,
        num_att_heads=4,
        final_mlp_layers=2,  # TODO: check, does the last layer count?
        num_interactions=1,
        num_elements=len(z_table),
        avg_num_neighbors=avg_num_neighbors,
        message_normalization='avg_num_neighbors',
        seed=42,
        device=device,
        trainable_rbf=True,
        learn_atomic_type_shifts=True,
        learn_atomic_type_scales=True,
        energy_regression_dim=66,  # This is the energy regression dimension
        layer_normalization_1=True,
        layer_normalization_2=True,
        residual_mlp_1=True,
        residual_mlp_2=True,
        use_charge_embed=True,
        use_spin_embed=True,
        zbl_repulsion_bool=True,
        electrostatic_energy_bool=True,
        dispersion_energy_bool=True,
        dispersion_energy_cutoff_lr_damping=2.0,
        radial_basis_fn='bernstein'

    )

model.to(device).eval()

#model.register_forward_hook(print_hook("some_block"))
# print model parameters
if True:
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape} ({param.numel()} parameters)")

# save params:
#torch.save(
##    model.state_dict(),
 #   'So3krates-torch/example/torchkrates.pth',
 #   
#)

batch["positions"].requires_grad_(True)
batch["atomic_numbers"] = torch.argmax(batch["node_attrs"], dim=-1) + 1

torch._dynamo.config.suppress_errors = False
#scripted_model = jit.compile(model)
result = model(batch.to_dict(),compute_stress=True)
print(f"Not compiled: Energy: {result['energy'].item():.12f}")
print(f'Dipole vector: {result["dipole_vec"]}')
print(f'Forces mean: {result["forces"].mean(dim=0)}')
print(f'stress: {result["stress"]}')
print(f'hirshfeld_ratios: {result["hirshfeld_ratios"]}')

exit()



print("COMPILING...")
model = torch.compile(model)

batch= batch.to_dict()
result = model(batch)

print(f"Compiled: {result['energy'].item():.4f}")

exit()
time_start = time.time()
for i in range(100):
    outputs = model(batch)
time_end = time.time()
print('######### Original Model #########')
print(f"Time taken for 100 iterations: {time_end - time_start:.4f} seconds")
print(f'Average time per iteration: {(time_end - time_start) / 100:.4f} seconds')
print(f'Iterations per second: {100 / (time_end - time_start):.2f}')

scripted_model(batch)
time_start = time.time()
for i in range(100):
    outputs = scripted_model(batch)
time_end = time.time()
print('######### Scripted Model #########')
print(f"Time taken for 100 iterations: {time_end - time_start:.4f} seconds")
print(f'Average time per iteration: {(time_end - time_start) / 100:.4f} seconds')
print(f'Iterations per second: {100 / (time_end - time_start):.2f}')

compiled_model(batch)
time_start = time.time()
for i in range(100):
    outputs = compiled_model(batch)
time_end = time.time()
print('######### Compiled Model #########')
print(f"Time taken for 100 iterations: {time_end - time_start:.4f} seconds")
print(f'Average time per iteration: {(time_end - time_start) / 100:.4f} seconds')
print(f'Iterations per second: {100 / (time_end - time_start):.2f}')

time_start = time.time()
for i in range(100):
    mace_mp_model_medium.calculate(mol, properties=['energy', 'forces'])
time_end = time.time()
print('######### medium MACE MP Model #########')
print(f"Time taken for 100 iterations: {time_end - time_start:.4f} seconds")
print(f'Average time per iteration: {(time_end - time_start) / 100:.4f} seconds')
print(f'Iterations per second: {100 / (time_end - time_start):.2f}')

time_start = time.time()
for i in range(100):
    mace_mp_model_small.calculate(mol, properties=['energy', 'forces'])
time_end = time.time()
print('######### small MACE MP Model #########')
print(f"Time taken for 100 iterations: {time_end - time_start:.4f} seconds")
print(f'Average time per iteration: {(time_end - time_start) / 100:.4f} seconds')
print(f'Iterations per second: {100 / (time_end - time_start):.2f}')