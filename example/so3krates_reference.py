from mlff.config.from_config import make_so3krates_sparse_from_config
import jax
from ml_collections import config_dict
from so3krates_torch.modules.models import So3krates
from mace.tools import torch_geometric, utils
from ase.build import molecule
import torch
from mace import data
from ase.io import read
from mace.modules.utils import compute_avg_num_neighbors
import jax.numpy as jnp
from flax.core import unfreeze
import flax
from mlff.calculators.ase_calculator import AseCalculatorSparse

def print_param_shapes(params, prefix=''):
    if isinstance(params, dict) or isinstance(params, flax.core.frozen_dict.FrozenDict):
        for k, v in params.items():
            print_param_shapes(v, prefix=f"{prefix}/{k}" if prefix else k)
    else:
        print(f"{prefix}: {params.shape}")
def print_param_shapes_and_count(params, prefix=''):
    total = 0

    def _recurse(p, prefix=''):
        nonlocal total
        if isinstance(p, dict) or isinstance(p, flax.core.frozen_dict.FrozenDict):
            for k, v in p.items():
                _recurse(v, prefix=f"{prefix}/{k}" if prefix else k)
        else:
            count = p.size  # number of elements in the array
            total += count
            print(f"{prefix}: shape={p.shape}, count={count}")

    _recurse(params, prefix)
    print(f"\nTotal parameters: {total}")
    
def collect_param_dtypes(params, prefix=''):
    dtypes = {}

    def _recurse(p, prefix=''):
        if isinstance(p, dict) or isinstance(p, flax.core.frozen_dict.FrozenDict):
            for k, v in p.items():
                _recurse(v, prefix=f"{prefix}/{k}" if prefix else k)
        else:
            dtypes[prefix] = p.dtype

    _recurse(params, prefix)
    return dtypes




#mol = molecule('H2O')
#mol = read('So3krates-torch/example/aspirin.xyz')
mol = read('So3krates-torch/example/ala4.xyz')
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
avg_num_neighbors = compute_avg_num_neighbors(
    data_loader
)

batch = next(iter(data_loader)).to_dict()




cfg = config_dict.ConfigDict()
cfg.model = config_dict.ConfigDict()

cfg.model.num_layers =1
cfg.model.num_features = 132
cfg.model.num_heads = 4
cfg.model.num_features_head = 32
cfg.model.radial_basis_fn = 'gaussian'
cfg.model.num_radial_basis_fn = 32
cfg.model.cutoff_fn = 'cosine'
cfg.model.cutoff = 5.0
cfg.model.cutoff_lr = None
cfg.model.degrees = [0,1,2,3]
cfg.model.residual_mlp_1 = False
cfg.model.residual_mlp_2 = False
cfg.model.layer_normalization_1 = False
cfg.model.layer_normalization_2 = False
cfg.model.message_normalization = 'avg_num_neighbors'
cfg.model.avg_num_neighbors = avg_num_neighbors
cfg.model.qk_non_linearity = 'identity'
cfg.model.activation_fn = 'silu'
cfg.model.layers_behave_like_identity_fn_at_init = False
cfg.model.output_is_zero_at_init = False
cfg.model.input_convention = 'positions'
cfg.model.use_charge_embed = False
cfg.model.use_spin_embed = False
cfg.model.energy_regression_dim = 132
cfg.model.energy_activation_fn = 'silu'
cfg.model.energy_learn_atomic_type_scales = False
cfg.model.energy_learn_atomic_type_shifts = False
cfg.model.electrostatic_energy_bool = False
cfg.model.electrostatic_energy_scale = 4.0
cfg.model.dispersion_energy_bool = False
cfg.model.dispersion_energy_cutoff_lr_damping = None
cfg.model.dispersion_energy_scale = 1.2
cfg.model.return_representations_bool = False
cfg.model.zbl_repulsion_bool = False
cfg.neighborlist_format_lr = 'sparse'
cfg.model.output_intermediate_quantities = False

cfg.data = config_dict.ConfigDict()
cfg.data.avg_num_neighbors = avg_num_neighbors

# save cfg as hyperparameters.json
import json
with open('So3krates-torch/example/hyperparameters.json', 'w', encoding='utf-8') as f:
    json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=4)

key = jax.random.PRNGKey(0)
model = make_so3krates_sparse_from_config(cfg)
#print(model.observables)

positions = jnp.array(batch['positions'])
num_nodes = batch['positions'].shape[0]
#create fake features by having an array of shape num_nodes, 132
features = jnp.zeros((num_nodes, 132), dtype=jnp.float32)
atomic_numbers = jnp.array(mol.get_atomic_numbers())
batch_segments = jnp.array(batch['batch'])
node_mask = jnp.ones((num_nodes,), dtype=jnp.bool_)
graph_mask = jnp.array([True])
idx_j, idx_i = jnp.array(batch['edge_index'][0]), jnp.array(batch['edge_index'][1])

inputs = {
    'x': features,
    'atomic_numbers': atomic_numbers,
    'batch_segments': batch_segments,
    'node_mask': node_mask,
    'graph_mask': graph_mask,
    'idx_i': idx_i,
    'idx_j': idx_j,
    'positions': positions,
    'total_charge': 0
}

params = model.init(key,inputs)
params = unfreeze(params)
print_param_shapes_and_count(params)

dtypes = collect_param_dtypes(params)
print("\nParameter dtypes:")
for k, v in dtypes.items():
    print(f"{k}: {v}")

#save params as params.pkl
import pickle
with open('So3krates-torch/example/params.pkl', 'wb') as f:
    pickle.dump(params, f)
    
calc = AseCalculatorSparse.create_from_workdir(
    workdir='So3krates-torch/example/',
    from_file=True,
)
calc.calculate(
    mol,
    properties=['energy']
)

print(calc.results['energy'])