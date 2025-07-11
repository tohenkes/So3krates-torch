from so3krates_torch.modules.models import So3krates
from mace.tools import torch_geometric, utils
import torch
from mace import data
from ase.io import read
from mace.modules.utils import compute_avg_num_neighbors
from torch.fx.experimental.proxy_tensor import make_fx
from typing import Dict, Sequence, List, Optional, Any, Final
import contextlib

def _list_to_dict(
    keys: Sequence[str], args: List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {key: arg for key, arg in zip(keys, args)}


def _list_from_dict(
    keys: Sequence[str], data: Dict[str, torch.Tensor]
) -> List[torch.Tensor]:
    return [data[key] for key in keys]


def _get_weights_buffers(model, weight_names, buffer_names):
    # get weights and buffers from trainable model
    weight_dict = dict(model.named_parameters())
    weights = [weight_dict[name] for name in weight_names]
    buffer_dict = dict(model.named_buffers())
    buffers = [buffer_dict[name] for name in buffer_names]
    return weights, buffers

class ListInputOutputWrapper(torch.nn.Module):
    """
    Wraps a ``torch.nn.Module`` that takes and returns ``Dict[str, torch.Tensor]`` to have it take and return ``Sequence[torch.Tensor]`` for specified input and output fields.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
    ):
        super().__init__()
        self.model = model
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        inputs = _list_to_dict(self.input_keys, args)
        outputs = self.model(inputs)
        return _list_from_dict(self.output_keys, outputs)


class DictInputOutputWrapper(torch.nn.Module):
    """
    Wraps a model that takes and returns ``Sequence[torch.Tensor]`` to have it take and return ``Dict[str, torch.Tensor]`` for specified input and output fields (i.e. the opposite of ``ListInputOutputWrapper``).
    """

    def __init__(self, model, input_keys: List[str], output_keys: List[str]):
        super().__init__()
        self.model = model
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, data):
        inputs = _list_from_dict(self.input_keys, data)
        with torch.inference_mode():
            outputs = self.model(inputs)
        return _list_to_dict(self.output_keys, outputs)
    
class ListInputOutputStateDictWrapper(ListInputOutputWrapper):
    """Like ``ListInputOutputWrapper``, but also updates the model with state dict entries before each ``forward``."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        state_dict_keys: Sequence[str],
    ):
        super().__init__(model, input_keys, output_keys)
        self.state_dict_keys = state_dict_keys

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        # won't check that `args` is of the correct length
        input_dict = _list_to_dict(self.input_keys, args[: len(self.input_keys)])
        state_dict = _list_to_dict(self.state_dict_keys, args[len(self.input_keys) :])
        # have to do it this way and not using state_dict directly for autograd reasons
        # the `.data` part is important
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(state_dict[name])
            for name, buffer in self.model.named_buffers():
                buffer.data.copy_(state_dict[name])
        output_dict = self.model(input_dict)
        return _list_from_dict(self.output_keys, output_dict)
    
@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    For our use of `make_fx` to unfold the autograd graph, we must set the following `use_duck_shape` parameter to `False` (it's `True` by default).
    It forces dynamic batch dims (num_frames, num_atoms, num_edges) to shape specialize if the batch dim is the same as that of a static dim.
    E.g. in training, shape specialization would occur if a weight tensor has a dimension with shape (16,) and we use a batch size of 16 (so the dynamic batch dim `num_frames` is 16) because of the duck shaping.
    """
    # save previous state
    init_duck_shape = torch.fx.experimental._config.use_duck_shape
    # set mode variables
    torch.fx.experimental._config.use_duck_shape = enabled
    try:
        yield
    finally:
        # restore state
        torch.fx.experimental._config.use_duck_shape = init_duck_shape
        
def _model_make_fx(model, inputs):
    with fx_duck_shape(False):
        return make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])
        



mol = read('./ala4.xyz')
r_max = 5.0
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
#print(batch.to_dict().keys())
degrees = [1,2,3,4]
model = So3krates(
    r_max=5.0,
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
    use_charge_embed=False,
    use_spin_embed=False,
).to(device)
batch = batch.to_dict()
result = model(batch)

input_fields = sorted(list(batch.keys()))
output_fields = sorted(list(result.keys()))
weight_names = [n for n, _ in model.named_parameters()]
buffer_names = [n for n, _ in model.named_buffers()]

model_to_trace = ListInputOutputStateDictWrapper(
    model=model,
    input_keys=input_fields,
    output_keys=output_fields,
    state_dict_keys=weight_names + buffer_names,
)
weights, buffers = _get_weights_buffers(model, weight_names, buffer_names)

test_data_list = [batch[key] for key in input_fields]

total_input_list = [i.clone() for i in test_data_list+ weights + buffers]
#model_to_trace(*total_input_list)


fx_model = _model_make_fx(model_to_trace, total_input_list)
print(fx_model)