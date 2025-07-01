from so3krates_torch.modules.models import So3krates
from so3krates_torch.blocks import so3_conv_invariants
from mace.tools import torch_geometric, utils
from ase.build import molecule
import torch
from mace import data
from e3nn import o3
import time

mol = molecule('H2O')
r_max = 5.0

z_table = utils.AtomicNumberTable(
            [int(z) for z in sorted(set(mol.get_atomic_numbers()))]
        )

complete_path = '/home/thenkes/Documents/Uni/Promotion/Research/torchkrates/test_mace/test-520.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'  # Use CPU for this example, change to 'cuda' if you have a GPU

dtype = 'float32'  # or torch.float64, depending on your model's requirements
torch.set_default_dtype(getattr(torch, dtype))

keyspec = data.KeySpecification(
    info_keys={}, arrays_keys={"charges": "Qs"}
)
config = data.config_from_atoms(
    mol, key_specification=keyspec, head_name="default"
)
data_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
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

max_l = 2

model = So3krates(
    r_max=5.0,
    num_radial_basis=8,
    max_l=max_l,
    features_dim=16,
    num_att_heads=4,
    atomic_numbers=[1, 8],  # H and O
    final_mlp_layers=2,
    num_interactions=2,
    num_elements=2,
    use_so3=False,
    avg_num_neighbors=2,
    seed=42,
    device=device,
)
model.to(device)
print(model(batch))

