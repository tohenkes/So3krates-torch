from so3krates_torch.modules.models import So3krates
from mace.tools import torch_geometric, utils
from ase.build import molecule
import torch
from mace import data

mol = molecule('H2O')
r_max = 5.0

z_table = utils.AtomicNumberTable(
            [int(z) for z in sorted(set(mol.get_atomic_numbers()))]
        )

complete_path = '/home/thenkes/Documents/Uni/Promotion/Research/torchkrates/test_mace/test-520.model'
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

model = So3krates(
    r_max=5.0,
    num_radial_basis=8,
    max_l=3,
    features_dim=16,
    ev_l=2,
    num_att_heads=4,
    atomic_numbers=[1, 8],  # H and O
    final_mlp_layers=2,
    num_interactions=2,
    num_elements=2,
    use_so3=False,
    avg_num_neighbors=2,
    seed=42,
)
model.to(device)
print(model(batch))