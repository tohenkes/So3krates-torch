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
import numpy as np
from mlff.sph_ops import make_l0_contraction_fn
from so3krates_torch.blocks.so3_conv_invariants import SO3ConvolutionInvariants
from so3krates_torch.modules.spherical_harmonics import RealSphericalHarmonics

def atoms_to_batch(
        atoms,
        device,

        ):
    keyspec = data.KeySpecification(
    info_keys={"total_charge":"total_charge","total_spin":"total_spin"}, arrays_keys={"charges": "Qs"}
    )
    config = data.config_from_atoms(
        atoms, key_specification=keyspec, head_name="default"
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=r_max,
                heads=["Default"],
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to(device)
    return batch

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
    
def flatten_params(params, prefix=''):
    flat_params = {}

    def _recurse(p, prefix=''):
        if isinstance(p, dict) or isinstance(p, flax.core.frozen_dict.FrozenDict):
            for k, v in p.items():
                _recurse(v, prefix=f"{prefix}/{k}" if prefix else k)
        else:
            flat_params[prefix] = p

    _recurse(params, prefix)
    return flat_params


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



dtype = 'float64'

if dtype == 'float64':
    torch.set_default_dtype(torch.float64)
    jax.config.update("jax_enable_x64", True)
    dtype = torch.float64
else:
    torch.set_default_dtype(torch.float32)
    jax.config.update("jax_enable_x64", False)
    dtype = torch.float32


#mol = molecule('H2O')
#mol = read('So3krates-torch/example/aspirin.xyz')
#mol = read('So3krates-torch/example/ala4.xyz')
mol = read('So3krates-torch/example/water_64.xyz')
charge = 3
mol.info["total_charge"] = charge
mol.info["total_spin"] = 0.5
mol.info["charge"] = charge
mol.info["multiplicity"] = 2 * mol.info["total_spin"] + 1
num_unpaired_electrons = mol.info["total_spin"] * 2
r_max = 5.

z_table = utils.AtomicNumberTable(
            [int(z) for z in range(1, 119)]
        )
device = 'cuda' if torch.cuda.is_available() else 'cpu'


keyspec = data.KeySpecification(
    info_keys={"total_charge":"total_charge","total_spin":"total_spin"}, arrays_keys={"charges": "Qs"}
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

batch = next(iter(data_loader))
cfg = config_dict.ConfigDict()
cfg.model = config_dict.ConfigDict()

cfg.model.num_layers =3
cfg.model.num_features = 132
cfg.model.num_heads = 4
cfg.model.num_features_head = 32
cfg.model.radial_basis_fn = 'gaussian'
cfg.model.num_radial_basis_fn = 32
cfg.model.cutoff_fn = 'cosine'
cfg.model.cutoff = 5.0
cfg.model.cutoff_lr = None
cfg.model.degrees = [1,2,3,4]
cfg.model.residual_mlp_1 = True
cfg.model.residual_mlp_2 = True
cfg.model.layer_normalization_1 = True
cfg.model.layer_normalization_2 = True
cfg.model.message_normalization = 'avg_num_neighbors'
cfg.model.avg_num_neighbors = avg_num_neighbors
cfg.model.qk_non_linearity = 'identity'
cfg.model.activation_fn = 'silu'
cfg.model.layers_behave_like_identity_fn_at_init = False
cfg.model.output_is_zero_at_init = False
cfg.model.input_convention = 'positions'
cfg.model.use_charge_embed = True
cfg.model.use_spin_embed = True
cfg.model.energy_regression_dim = 66
cfg.model.energy_activation_fn = 'silu'
cfg.model.energy_learn_atomic_type_scales = True
cfg.model.energy_learn_atomic_type_shifts = True
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
features = jnp.zeros((num_nodes, cfg.model.num_features), dtype=jnp.float32)
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
    'total_charge': jnp.array([charge]),
    'num_unpaired_electrons': jnp.array([num_unpaired_electrons]),
}

params = model.init(key,inputs)
params = unfreeze(params)
flattened_params = flatten_params(params)

#for k, v in flattened_params.items():
#    print(f"{k}: {v.shape} ({v.size} elements)")
#exit()


def generate_full_flax_to_torch_mapping(
    cfg
    ):
    num_layers = cfg.model.num_layers
    layer_norm_1 = cfg.model.layer_normalization_1
    layer_norm_2 = cfg.model.layer_normalization_2
    residual_mlp_1 = cfg.model.residual_mlp_1
    residual_mlp_2 = cfg.model.residual_mlp_2
    learn_atomic_type_shifts = cfg.model.energy_learn_atomic_type_shifts
    learn_atomic_type_scales = cfg.model.energy_learn_atomic_type_scales
    use_charge_embed = cfg.model.use_charge_embed
    use_spin_embed = cfg.model.use_spin_embed
    mapping = {}

    # Embedding layers
    mapping["params/feature_embeddings_0/Embed_0/embedding"] = "inv_feature_embedding.embedding.weight"
    mapping["params/geometry_embeddings_0/rbf_fn/centers"] = "radial_embedding.radial_basis_fn.centers"
    mapping["params/geometry_embeddings_0/rbf_fn/widths"] = "radial_embedding.radial_basis_fn.widths"
    if use_charge_embed and not use_spin_embed:
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"] = "charge_embedding.Wq.weight"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"] = "charge_embedding.Wk"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"] = "charge_embedding.Wv"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"] = "charge_embedding.mlp.1.weight"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"] = "charge_embedding.mlp.3.weight"
    elif use_spin_embed and not use_charge_embed:
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"] = "spin_embedding.Wq.weight"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"] = "spin_embedding.Wk"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"] = "spin_embedding.Wv"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"] = "spin_embedding.mlp.1.weight"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"] = "spin_embedding.mlp.3.weight"
    elif use_charge_embed and use_spin_embed:
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"] = "charge_embedding.Wq.weight"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"] = "charge_embedding.Wk"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"] = "charge_embedding.Wv"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"] = "charge_embedding.mlp.1.weight"
        mapping["params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"] = "charge_embedding.mlp.3.weight"
        mapping["params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_0/embedding"] = "spin_embedding.Wq.weight"
        mapping["params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_1/embedding"] = "spin_embedding.Wk"
        mapping["params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_2/embedding"] = "spin_embedding.Wv"
        mapping["params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"] = "spin_embedding.mlp.1.weight"
        mapping["params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"] = "spin_embedding.mlp.3.weight"



    # Per-layer transformer mappings
    for i in range(num_layers):
        flax_prefix = f"params/layers_{i}/attention_block"
        torch_prefix = f"euclidean_transformers.{i}"

        # Radial filters (inv)
        mapping[f"{flax_prefix}/radial_filter1_layer_1/kernel"] = f"{torch_prefix}.filter_net_inv.mlp_rbf.0.weight"
        mapping[f"{flax_prefix}/radial_filter1_layer_1/bias"] = f"{torch_prefix}.filter_net_inv.mlp_rbf.0.bias"
        mapping[f"{flax_prefix}/radial_filter1_layer_2/kernel"] = f"{torch_prefix}.filter_net_inv.mlp_rbf.mlp_rbf_layer_1.0.weight"
        mapping[f"{flax_prefix}/radial_filter1_layer_2/bias"] = f"{torch_prefix}.filter_net_inv.mlp_rbf.mlp_rbf_layer_1.0.bias"

        # Radial filters (ev)
        mapping[f"{flax_prefix}/radial_filter2_layer_1/kernel"] = f"{torch_prefix}.filter_net_ev.mlp_rbf.0.weight"
        mapping[f"{flax_prefix}/radial_filter2_layer_1/bias"] = f"{torch_prefix}.filter_net_ev.mlp_rbf.0.bias"
        mapping[f"{flax_prefix}/radial_filter2_layer_2/kernel"] = f"{torch_prefix}.filter_net_ev.mlp_rbf.mlp_rbf_layer_1.0.weight"
        mapping[f"{flax_prefix}/radial_filter2_layer_2/bias"] = f"{torch_prefix}.filter_net_ev.mlp_rbf.mlp_rbf_layer_1.0.bias"

        # Spherical filters (inv)
        mapping[f"{flax_prefix}/spherical_filter1_layer_1/kernel"] = f"{torch_prefix}.filter_net_inv.mlp_ev.0.weight"
        mapping[f"{flax_prefix}/spherical_filter1_layer_1/bias"] = f"{torch_prefix}.filter_net_inv.mlp_ev.0.bias"
        mapping[f"{flax_prefix}/spherical_filter1_layer_2/kernel"] = f"{torch_prefix}.filter_net_inv.mlp_ev.mlp_ev_layer_1.0.weight"
        mapping[f"{flax_prefix}/spherical_filter1_layer_2/bias"] = f"{torch_prefix}.filter_net_inv.mlp_ev.mlp_ev_layer_1.0.bias"

        # Spherical filters (ev)
        mapping[f"{flax_prefix}/spherical_filter2_layer_1/kernel"] = f"{torch_prefix}.filter_net_ev.mlp_ev.0.weight"
        mapping[f"{flax_prefix}/spherical_filter2_layer_1/bias"] = f"{torch_prefix}.filter_net_ev.mlp_ev.0.bias"
        mapping[f"{flax_prefix}/spherical_filter2_layer_2/kernel"] = f"{torch_prefix}.filter_net_ev.mlp_ev.mlp_ev_layer_1.0.weight"
        mapping[f"{flax_prefix}/spherical_filter2_layer_2/bias"] = f"{torch_prefix}.filter_net_ev.mlp_ev.mlp_ev_layer_1.0.bias"

        # Attention weights
        mapping[f"{flax_prefix}/Wq1"] = f"{torch_prefix}.euclidean_attention_block.W_q_inv"
        mapping[f"{flax_prefix}/Wk1"] = f"{torch_prefix}.euclidean_attention_block.W_k_inv"
        mapping[f"{flax_prefix}/Wv1"] = f"{torch_prefix}.euclidean_attention_block.W_v_inv"
        mapping[f"{flax_prefix}/Wq2"] = f"{torch_prefix}.euclidean_attention_block.W_q_ev"
        mapping[f"{flax_prefix}/Wk2"] = f"{torch_prefix}.euclidean_attention_block.W_k_ev"

        # Exchange block
        mapping[f"params/layers_{i}/exchange_block/mlp_layer_2/kernel"] = f"{torch_prefix}.interaction_block.linear_layer.weight"
        mapping[f"params/layers_{i}/exchange_block/mlp_layer_2/bias"] = f"{torch_prefix}.interaction_block.linear_layer.bias"

        # Layer normalization
        if layer_norm_1:
            mapping[f"params/layers_{i}/layer_normalization_1/scale"] = f"{torch_prefix}.layer_norm_inv_1.weight"
            mapping[f"params/layers_{i}/layer_normalization_1/bias"] = f"{torch_prefix}.layer_norm_inv_1.bias"
        if layer_norm_2:
            mapping[f"params/layers_{i}/layer_normalization_2/scale"] = f"{torch_prefix}.layer_norm_inv_2.weight"
            mapping[f"params/layers_{i}/layer_normalization_2/bias"] = f"{torch_prefix}.layer_norm_inv_2.bias"
        
        # Residual MLPs
        if residual_mlp_1:
            mapping[f"params/layers_{i}/res_mlp_1_layer_1/kernel"] = f"{torch_prefix}.residual_mlp_1.1.weight"
            mapping[f"params/layers_{i}/res_mlp_1_layer_1/bias"] = f"{torch_prefix}.residual_mlp_1.1.bias"
            mapping[f"params/layers_{i}/res_mlp_1_layer_2/kernel"] = f"{torch_prefix}.residual_mlp_1.3.weight"
            mapping[f"params/layers_{i}/res_mlp_1_layer_2/bias"] = f"{torch_prefix}.residual_mlp_1.3.bias"
        if residual_mlp_2:
            mapping[f"params/layers_{i}/res_mlp_2_layer_1/kernel"] = f"{torch_prefix}.residual_mlp_2.1.weight"
            mapping[f"params/layers_{i}/res_mlp_2_layer_1/bias"] = f"{torch_prefix}.residual_mlp_2.1.bias"
            mapping[f"params/layers_{i}/res_mlp_2_layer_2/kernel"] = f"{torch_prefix}.residual_mlp_2.3.weight"
            mapping[f"params/layers_{i}/res_mlp_2_layer_2/bias"] = f"{torch_prefix}.residual_mlp_2.3.bias"
            
    # Output layers
    mapping["params/observables_0/energy_dense_regression/kernel"] = "energy_output_block.layers.0.weight"
    mapping["params/observables_0/energy_dense_regression/bias"] = "energy_output_block.layers.0.bias"
    mapping["params/observables_0/energy_dense_final/kernel"] = "energy_output_block.final_layer.weight"
    mapping["params/observables_0/energy_dense_final/bias"] = "energy_output_block.final_layer.bias"
    if learn_atomic_type_shifts:
        mapping["params/observables_0/energy_offset"] = "energy_output_block.energy_shifts.weight"
    if learn_atomic_type_scales:
        mapping["params/observables_0/atomic_scales"] = "energy_output_block.energy_scales.weight"

    return mapping



    
# Convert parameters from Flax to PyTorch

#save params as params.pkl
import pickle
with open('So3krates-torch/example/params.pkl', 'wb') as f:
    pickle.dump(params, f)
    
calc = AseCalculatorSparse.create_from_workdir(
    workdir='So3krates-torch/example/',
    from_file=True,
)





max_l = 3
model = So3krates(
    r_max=5.0,
    num_radial_basis=cfg.model.num_radial_basis_fn,
    degrees=cfg.model.degrees,
    features_dim=cfg.model.num_features,
    num_att_heads=cfg.model.num_heads,
    atomic_numbers=mol.get_atomic_numbers(),  # H and O
    final_mlp_layers=2,  # TODO: check, does the last layer count?
    num_interactions=cfg.model.num_layers,
    num_elements=len(z_table),
    avg_num_neighbors=avg_num_neighbors,
    message_normalization=cfg.model.message_normalization,
    seed=42,
    device=device,
    trainable_rbf=True,
    dtype=dtype,
    learn_atomic_type_shifts=cfg.model.energy_learn_atomic_type_shifts,
    learn_atomic_type_scales=cfg.model.energy_learn_atomic_type_scales,
    energy_regression_dim=cfg.model.energy_regression_dim,
    layer_normalization_1=cfg.model.layer_normalization_1, 
    layer_normalization_2=cfg.model.layer_normalization_2, 
    residual_mlp_1=cfg.model.residual_mlp_1,
    residual_mlp_2=cfg.model.residual_mlp_2,
    use_charge_embed=cfg.model.use_charge_embed,
    use_spin_embed=cfg.model.use_spin_embed,
)



state_dict = model.state_dict()
mapping = generate_full_flax_to_torch_mapping(
    cfg=cfg
)

#for k,v in mapping.items():
#    print(f"{k} -> {v}")

embeddings = [
    "inv_feature_embedding.embedding.weight",
    "charge_embedding.Wq.weight",
    "spin_embedding.Wq.weight",
]
special_embeddings = [
    "charge_embedding.Wk",
    "charge_embedding.Wv",
    "spin_embedding.Wk",
    "spin_embedding.Wv",
]
for flax_key, torch_key in mapping.items():
    flax_array = flattened_params[flax_key]
    flax_array_np = np.array(flax_array)
    torched = torch.from_numpy(flax_array_np)
    
    expected_shape = state_dict[torch_key].shape
    if flax_array.ndim == 2 and torch_key not in special_embeddings:
        torched = torched.T
    elif flax_array.ndim == 3:
        torched = torched.permute(0,2,1)
    
    
    if torch_key in embeddings:
        torched = torched[:,1:]
    if (
        flax_key == "params/observables_0/energy_offset" or flax_key == "params/observables_0/atomic_scales"):
        torched = torched[1:].unsqueeze(0) 

    if torched.shape != expected_shape:
        print(f"Shape mismatch for {torch_key}: expected {expected_shape}, got {torched.shape}")
    state_dict[torch_key] = torched
    


model.load_state_dict(state_dict, strict=True)


model.to(device).eval()
#model = torch.compile(model)
batch_torch_model = batch.to(device)
batch_torch_model = batch_torch_model.to_dict()
batch_torch_model["positions"].requires_grad_(True)
result = model(batch_torch_model)

compute = True
#compute = False


if compute:
    calc.calculate(mol, properties=['energy', 'forces'],)

print("\n\n")
print("##############  RESULTS  ##############")
print("Energy")
print(f"TORCH version: {result['energy'].item():.6f}")
if compute:
    print(f"JAX version  : {calc.results['energy']:.6f}")
    print(f"Difference   : {calc.results['energy'] - result['energy'].item():.6f}")

print("\n\n")
print("Forces")
print("TORCH version: ", result['forces'][0,:])    
print("JAX version  : ", calc.results['forces'][0,:])
print(f"Difference   :  {np.abs(calc.results['forces'] - np.array(result['forces'])).mean().item():.6f}")
print("\n\n")

inference = True
if inference:
    print("Inference timings")
    import time
    num_runs = 100
    # Warmup run
    model(batch_torch_model)
    calc.calculate(mol, properties=['energy', 'forces'],)

    time_start_torch = time.time()
    for i in range(num_runs):
        batch_temp = atoms_to_batch(
            mol,
            device=device,
        ).to_dict()
        batch_temp["positions"].requires_grad_(True)
        result = model(batch_temp)
    time_end_torch = time.time()
    timings_torch = time_end_torch - time_start_torch
    print('######### Torch Model #########')
    print(f"Time taken for {num_runs} iterations: {timings_torch:.4f} seconds")
    print(f'Average time per iteration: {(timings_torch) / num_runs:.4f} seconds')
    print(f'Iterations per second: {num_runs / (timings_torch):.2f}')
    time_start_jax = time.time()
    for i in range(num_runs):
        calc.calculate(mol, properties=['energy', 'forces'],)
    time_end_jax = time.time()
    timing_jax = time_end_jax - time_start_jax
    print('######### JAX Model #########')
    print(f"Time taken for {num_runs} iterations: {timing_jax:.4f} seconds")
    print(f'Average time per iteration: {(timing_jax) / num_runs:.4f} seconds')
    print(f'Iterations per second: {num_runs / (timing_jax):.2f}')
    jax_faster = timing_jax < timings_torch
    if jax_faster:
        print(f"The JAX model is {timings_torch / timing_jax:.2f} times faster than the Torch model.")
    else:
        print(f"The JAX model is {timing_jax / timings_torch:.2f} times slower than the Torch model.")

