from so3krates_torch.modules.models import So3krates
from mace.tools import torch_geometric, utils
from ase.build import molecule
import torch
from mace import data
import time
from e3nn.util import jit
from ase.io import read
from mace.modules.utils import compute_avg_num_neighbors
from mace.calculators import mace_mp

mol = molecule('H2O')
mol = read('So3krates-torch/example/aspirin.xyz')
mol = read('So3krates-torch/example/ala4.xyz')
r_max = 5.0

z_table = utils.AtomicNumberTable(
            [int(z) for z in range(1, 120)]
        )
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dtype = 'float32'  # or torch.float64, depending on your model's requirements
torch.set_default_dtype(getattr(torch, dtype))

keyspec = data.KeySpecification(
    info_keys={}, arrays_keys={"charges": "Qs"}
)
config = data.config_from_atoms(
    mol, key_specification=keyspec, head_name="default"
)
data_loader = torch_geometric.dataloader.DataLoader(
    dataset=20*[
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
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
mace_mp_model_medium= mace_mp()
mace_mp_model_small = mace_mp(model='small')

max_l = 3
model = So3krates(
    r_max=5.0,
    num_radial_basis=32,
    max_l=max_l,
    features_dim=132,
    num_att_heads=4,
    atomic_numbers=mol.get_atomic_numbers(),  # H and O
    final_mlp_layers=2,  # TODO: check, does the last layer count?
    num_interactions=1,
    num_elements=len(z_table),
    use_so3=False,
    avg_num_neighbors=avg_num_neighbors,
    seed=42,
    device=device,
    trainable_rbf=True,
)
model.to(device).eval()

# print model parameters
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape} ({param.numel()} parameters)")

# save params:
torch.save(
    model.state_dict(),
    'So3krates-torch/example/torchkrates.pth',
    
)


#scripted_model = jit.compile(model)
compiled_model = torch.compile(model)

batch= batch.to_dict()
result = model(batch)

print(f"{result['energy'].item():.4f}")

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