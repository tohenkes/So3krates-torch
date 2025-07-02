from so3krates_torch.modules.models import So3krates
from mace.tools import torch_geometric, utils
from ase.build import molecule
import torch
from mace import data
import time
from e3nn.util import jit


mol = molecule('H2O')
r_max = 5.0

z_table = utils.AtomicNumberTable(
            [int(z) for z in sorted(set(mol.get_atomic_numbers()))]
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

batch = next(iter(data_loader)).to(device)

max_l = 3
model = So3krates(
    r_max=5.0,
    num_radial_basis=32,
    max_l=max_l,
    features_dim=132,
    num_att_heads=4,
    atomic_numbers=[1, 8],  # H and O
    final_mlp_layers=2,
    num_interactions=3,
    num_elements=2,
    use_so3=False,
    avg_num_neighbors=2,
    seed=42,
    device=device,
)
model.to(device).eval()
scripted_model = jit.compile(model)
compiled_model = torch.compile(model)

batch= batch.to_dict()


model(batch)
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