from so3krates_torch.modules.models import So3krates
from mace.tools import torch_geometric, utils
import torch
from mace import data
from ase.io import read
from mace.modules.utils import compute_avg_num_neighbors
from torch.fx.experimental.proxy_tensor import make_fx
from typing import Dict, Sequence, List, Optional, Any, Final, Callable, Union
import contextlib
import math
import difflib
import os
import uuid

def _pt2_compile_error_message(key, tol, err, absval, model_dtype):
    return f"Compilation check MaxAbsError: {err:.6g} (tol: {tol}) for field `{key}`. This assert was triggered because the outputs of an eager model and a compiled model numerically differ above the specified tolerance. This may indicate a compilation error (bug) specific to certain machines and installation environments, or may be an artefact of poor initialization if the error is close to the tolerance. Note that the largest absolute (MaxAbs) entry of the model prediction is {absval:.6g} -- you can use this detail to discern if it is a numerical artefact (the errors could be large because the MaxAbs value is very large) or a more fundamental compilation error. Raise a GitHub issue if you believe it is a compilation error or are unsure. If you are confident that it is purely numerical, and want to bypass the tolerance check, you may set the following environment variables: `NEQUIP_FLOAT64_MODEL_TOL`, `NEQUIP_FLOAT32_MODEL_TOL`, or `NEQUIP_TF32_MODEL_TOL`, depending on the model dtype you are using (which is currently {model_dtype}) and whether TF32 is on."


def dtype_to_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, str):
        return name
    return {torch.float32: "float32", torch.float64: "float64"}[name]

# === floating point tolerances as env vars ===
_FLOAT64_MODEL_TOL: Final[float] = float(
    os.environ.get("NEQUIP_FLOAT64_MODEL_TOL", 1e-12)
)
_FLOAT32_MODEL_TOL: Final[float] = float(
    os.environ.get("NEQUIP_FLOAT32_MODEL_TOL", 5e-5)
)
_TF32_MODEL_TOL: Final[float] = float(os.environ.get("NEQUIP_TF32_MODEL_TOL", 2e-3))


def floating_point_tolerance(model_dtype: Union[str, torch.dtype]):
    """
    Consistent set of floating point tolerances for sanity checking based on ``model_dtype``, that also accounts for TF32 state.

    Assumes global dtype if ``float64``, and that TF32 will only ever be used if ``model_dtype`` is ``float32``.
    """
    using_tf32 = False
    if torch.cuda.is_available():
        # assume that both are set to be the same
        assert torch.backends.cuda.matmul.allow_tf32 == torch.backends.cudnn.allow_tf32
        using_tf32 = torch.torch.backends.cuda.matmul.allow_tf32
    return {
        torch.float32: _TF32_MODEL_TOL if using_tf32 else _FLOAT32_MODEL_TOL,
        "float32": _TF32_MODEL_TOL if using_tf32 else _FLOAT32_MODEL_TOL,
        torch.float64: _FLOAT64_MODEL_TOL,
        "float64": _FLOAT64_MODEL_TOL,
    }[model_dtype]
    
def _default_error_message(key, tol, err, absval, model_dtype):
    return f"MaxAbsError: {err:.6g} (tol: {tol} for {model_dtype} model) for field `{key}`. MaxAbs value: {absval:.6g}."

_NUM_EVAL_TRIALS = 5

def test_model_output_similarity_by_dtype(
    model1: Callable,
    model2: Callable,
    data: Dict[str, torch.Tensor],
    model_dtype: Union[str, torch.dtype],
    fields: Optional[List[str]] = None,
    error_message: Callable = _default_error_message,
):
    """
    Assumptions and behavior:
    - `model1` and `model2` have signature `AtomicDataDict -> AtomicDataDict`
    - if `fields` are not provided, the function will loop over `model1`'s output keys
    """
    tol = floating_point_tolerance(model_dtype)

    # do one evaluation to figure out the fields if not provided
    if fields is None:
        fields = model1(data.copy()).keys()

    # perform `_NUM_EVAL_TRIALS` evaluations with each model and average to account for numerical randomness
    out1_list, out2_list = {k: [] for k in fields}, {k: [] for k in fields}
    for _ in range(_NUM_EVAL_TRIALS):
        out1 = model1(data.copy())
        out2 = model2(data.copy())
        for k in fields:
            out1_list[k].append(out1[k].detach().double())
            out2_list[k].append(out2[k].detach().double())
        del out1, out2

    for k in fields:
        t1, t2 = (
            torch.mean(torch.stack(out1_list[k], -1), -1),
            torch.mean(torch.stack(out2_list[k], -1), -1),
        )
        err = torch.max(torch.abs(t1 - t2)).item()
        absval = t1.abs().max().item()

        assert torch.allclose(t1, t2, atol=tol, rtol=tol), error_message(
            k, tol, err, absval, model_dtype
        )

        del t1, t2, err, absval
        
def highlight_code_differences(code1, code2):
    differ = difflib.Differ()
    diff = list(differ.compare(code1.splitlines(), code2.splitlines()))

    highlighted = []
    for line in diff:
        if line.startswith("  "):  # unchanged line
            highlighted.append(line[2:])
        elif line.startswith("- "):  # removed line
            highlighted.append(f"\033[91m{line[2:]}\033[0m")  # red
        elif line.startswith("+ "):  # Added line
            highlighted.append(f"\033[92m{line[2:]}\033[0m")  # green
        elif line.startswith("? "):  # line with changes
            continue  # skip the change markers

    return "\n".join(highlighted)


def check_make_fx_diff(fx_model_1, fx_model_2, fields: List[str]):
    print("Checking for differences in FX models...")
    if fx_model_1.code != fx_model_2.code:
        # the following is commented to prevent obscuring the error message below
        # devs can uncomment for diagonosing shape specializations
        print(highlight_code_differences(fx_model_1.code, fx_model_2.code))
        dump_dir = str(os.getcwd()) + "/nequip_fx_dump_" + str(uuid.uuid4())
        os.mkdir(dump_dir)
        with open(dump_dir + "/fx_model_1.txt", "w") as f:
            f.write(f"# Argument order:\n{fields} + extra_inputs\n")
            f.write(fx_model_1.code)
        with open(dump_dir + "/fx_model_2.txt", "w") as f:
            f.write(f"# Argument order:\n{fields} + extra_inputs\n")
            f.write(fx_model_2.code)
        raise RuntimeError(
            f"An unexpected internal error has occurred (the fx'ed models for different input shapes do not agree) -- please report this issue on the NequIP GitHub, and upload the files in {dump_dir}."
        )

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
        
class CompiledModel(So3krates):
    """Wrapper that uses ``torch.compile`` to optimize the wrapped module while allowing it to be trained."""
    def __init__(
        self,
        model: torch.nn.Module,
        model_config: dict
    ) -> None:
        super().__init__(**model_config)
        # these will be updated when the model is compiled
        self._compiled_model = ()
        self.model = model
        self.input_fields = None
        self.output_fields = None
        # weights and buffers should be done lazily because model modification can happen after instantiation
        # such that parameters and buffers may change between class instantiation and the lazy compilation in the `forward`
        self.weight_names = None
        self.buffer_names = None
        self.output_keys = [
            "energy",
            "forces",
            "virials",
            "stress",
            "hessian",
            "edge_forces"
        ]
        self.model_dtype = torch.get_default_dtype()
    def forward(
        self, 
        data: Dict[str, torch.Tensor],        
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # short-circuit if one of the batch dims is 1 (0 would be an error)
        # this is related to the 0/1 specialization problem
        # see https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ&tab=t.0#heading=h.ez923tomjvyk
        # we just need something that doesn't have a batch dim of 1 to `make_fx` or else it'll shape specialize
        # the models compiled for more batch_size > 1 data cannot be used for batch_size=1 data
        # (under specific cases related to the `PerTypeScaleShift` module)
        # for now we just make sure to always use the eager model when the data has any batch dims of 1
        if (
            torch.unique(batch['batch']).shape[0] < 2
        ):
            # use parent class's forward
            return super().forward(data)

        # === compile ===
        # compilation happens on the first data pass when there are at least two atoms (hard to pre-emp pathological data)
        if not self._compiled_model:

            # get weight names and buffers
            self.weight_names = [n for n, _ in self.model.named_parameters()]
            self.buffer_names = [n for n, _ in self.model.named_buffers()]

            # == get input and output fields ==
            # use intersection of data keys and GraphModel input/outputs, which assumes
            # - correctness of irreps registration system
            # - all input `data` batches have the same keys, and contain all necessary inputs and reference labels (outputs)
            self.input_fields = sorted(list(data.keys()))
            self.output_fields = sorted(self.output_keys)

            # == preprocess model and make_fx ==
            model_to_trace = ListInputOutputStateDictWrapper(
                model=self.model,
                input_keys=self.input_fields,
                output_keys=self.output_fields,
                state_dict_keys=self.weight_names + self.buffer_names,
            )

            weights, buffers = self._get_weights_buffers()
            fx_model = model_make_fx(
                model=model_to_trace,
                data=data,
                fields=self.input_fields,
                extra_inputs=weights + buffers,
            )
            del weights, buffers

            # == compile exported program ==
            # see https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#running-the-exported-program
            # TODO: compile options
            self._compiled_model = (
                torch.compile(
                    fx_model,
                    dynamic=True,
                    fullgraph=True,
                ),
            )
            # NOTE: the compiled model is wrapped in a tuple so that it's not registered and saved in the state dict -- this is necessary to enable `GraphModel` to load `CompileGraphModel` state dicts
            # see https://discuss.pytorch.org/t/saving-nn-module-to-parent-nn-module-without-registering-paremeters/132082/6

            # run original model and compiled model with data to sanity check
            test_model_output_similarity_by_dtype(
                self._compiled_forward,
                self.model,
                {k: data[k] for k in self.input_fields},
                dtype_to_name(self.model_dtype),
                fields=self.output_fields,
                error_message=_pt2_compile_error_message,
            )

        # === run compiled model ===
        out_dict = self._compiled_forward(data)
        to_return = {k: None for k in self.output_keys}
        to_return.update(out_dict)
        return to_return

    def _compiled_forward(self, data):
        # run compiled model with data
        weights, buffers = self._get_weights_buffers()
        data_list = _list_from_dict(self.input_fields, data)
        out_list = self._compiled_model[0](*(data_list + weights + buffers))
        out_dict = _list_to_dict(self.output_fields, out_list)
        return out_dict

    def _get_weights_buffers(self):
        # get weights and buffers from trainable model
        weight_dict = dict(self.model.named_parameters())
        weights = [weight_dict[name] for name in self.weight_names]
        buffer_dict = dict(self.model.named_buffers())
        buffers = [buffer_dict[name] for name in self.buffer_names]
        return weights, buffers

def model_make_fx(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    fields: List[str],
    extra_inputs: List[torch.Tensor] = [],
    seed: int = 1,
):
    """
    Args:
        model (torch.nn.Module): model must only take in flat ``torch.Tensor`` inputs
        data (AtomicDataDict.Type): an ``AtomicDataDict``
        fields (List[str]): ``AtomicDataDict`` fields that are used as the flat inputs to model
        extra_inputs (List[torch.Tensor]): list of additional ``torch.Tensor`` input data that are not ``AtomicDataDict`` fields
        seed (int): optional seed for reproducibility
    """
    # we do it twice
    # 1. once with the original input data
    test_data_list = [data[key] for key in fields]
    fx_model = _model_make_fx(model, test_data_list + extra_inputs)

    # 2. Follow NequIP's approach: extract first graph, augment it, combine back
    device = data['positions'].device
    generator = torch.Generator(device).manual_seed(seed)

    # Extract the first graph from the batch
    def extract_first_graph(batch_data):
        """Extract the first graph from a batched PyG-style data structure"""
        first_graph_mask = batch_data['batch'] == 0
        num_first_graph_nodes = first_graph_mask.sum().item()
        
        # Find edges belonging to the first graph
        edge_mask = (first_graph_mask[batch_data['edge_index'][0]] &
                     first_graph_mask[batch_data['edge_index'][1]])
        
        single_graph = {}
        for key, value in batch_data.items():
            if key == 'edge_index':
                # Extract edges and remap indices to start from 0
                edges = value[:, edge_mask]
                node_mapping = torch.zeros(
                    len(first_graph_mask),
                    dtype=torch.long, device=value.device)
                node_mapping[first_graph_mask] = torch.arange(
                    num_first_graph_nodes, device=value.device)
                single_graph[key] = node_mapping[edges]
            elif key in ['shifts', 'unit_shifts']:  # Edge-level features
                single_graph[key] = value[edge_mask]
            elif key in ['charges', 'positions', 'forces', 'node_attrs']:
                # Node-level features
                single_graph[key] = value[first_graph_mask]
            elif key == 'batch':
                # Create new batch tensor for single graph
                single_graph[key] = torch.zeros(
                    num_first_graph_nodes, dtype=value.dtype, device=value.device)
            elif key == 'ptr':
                # Update ptr for single graph
                single_graph[key] = torch.tensor(
                    [0, num_first_graph_nodes], dtype=value.dtype,
                    device=value.device)
            else:  # Graph-level features - take first graph's values
                if (len(value.shape) > 0 and
                        value.shape[0] == batch_data['batch'].max().item() + 1):
                    single_graph[key] = value[0:1]  # Keep batch dimension
                else:
                    single_graph[key] = value
        
        return single_graph

    single_frame = extract_first_graph(data)

    # Apply node removal to the single graph
    num_nodes = single_frame['positions'].shape[0]
    node_idx = torch.randint(
        low=0,
        high=num_nodes,
        size=(max(2, math.ceil(num_nodes * 0.1)),),
        generator=generator,
        device=device,
    )

    def without_nodes(data_single, which_nodes):
        """Remove nodes from a single graph"""
        num_nodes_single = data_single['positions'].shape[0]
        node_mask = torch.ones(num_nodes_single, dtype=torch.bool, device=device)
        node_mask[which_nodes] = False
        
        # Edge mask: keep edges where both endpoints are kept
        edge_idx = data_single['edge_index']
        edge_mask = node_mask[edge_idx[0]] & node_mask[edge_idx[1]]
        
        # Create index mapping for remaining nodes
        new_index = torch.full(
            (num_nodes_single,), fill_value=-1,
            dtype=torch.long, device=device)
        new_index[node_mask] = torch.arange(
            node_mask.sum(), dtype=torch.long, device=device)
        
        new_dict = {}
        for key, value in data_single.items():
            if key == 'edge_index':
                filtered_edges = edge_idx[:, edge_mask]
                new_dict[key] = new_index[filtered_edges]
            elif key in ['shifts', 'unit_shifts']:  # Edge-level features
                new_dict[key] = value[edge_mask]
            elif key in ['charges', 'positions', 'forces', 'node_attrs']:
                # Node-level features
                new_dict[key] = value[node_mask]
            elif key == 'batch':
                new_dict[key] = torch.zeros(
                    node_mask.sum(), dtype=value.dtype, device=device)
            elif key == 'ptr':
                new_dict[key] = torch.tensor(
                    [0, node_mask.sum()], dtype=value.dtype, device=device)
            else:  # Graph-level features
                new_dict[key] = value
        
        return new_dict

    augmented_single = without_nodes(single_frame, node_idx)

    # Combine original batch with augmented single graph (NequIP approach)
    def combine_batch_with_single(original_batch, single_graph):
        """Combine original batch with an additional single graph"""
        combined = {}
        
        # Get dimensions
        orig_num_nodes = original_batch['positions'].shape[0]
        single_num_nodes = single_graph['positions'].shape[0]
        orig_num_graphs = original_batch['batch'].max().item() + 1
        
        for key, orig_value in original_batch.items():
            single_value = single_graph[key]
            
            if key == 'edge_index':
                # Shift single graph edge indices
                shifted_edges = single_value + orig_num_nodes
                combined[key] = torch.cat([orig_value, shifted_edges], dim=1)
            elif key in ['shifts', 'unit_shifts']:  # Edge-level features
                combined[key] = torch.cat([orig_value, single_value], dim=0)
            elif key in ['charges', 'positions', 'forces', 'node_attrs']:
                # Node-level features
                combined[key] = torch.cat([orig_value, single_value], dim=0)
            elif key == 'batch':
                # Assign new batch index to single graph
                new_batch_idx = torch.full(
                    (single_num_nodes,), orig_num_graphs,
                    dtype=orig_value.dtype, device=orig_value.device)
                combined[key] = torch.cat([orig_value, new_batch_idx], dim=0)
            elif key == 'ptr':
                # Update ptr to include new graph
                new_ptr = torch.tensor(
                    [orig_num_nodes + single_num_nodes],
                    dtype=orig_value.dtype, device=orig_value.device)
                combined[key] = torch.cat([orig_value, new_ptr])
            else:  # Graph-level features
                if (len(orig_value.shape) > 0 and
                        orig_value.shape[0] == orig_num_graphs):
                    combined[key] = torch.cat([orig_value, single_value], dim=0)
                else:
                    combined[key] = orig_value  # Keep as-is for scalar features
        
        return combined

    augmented_data = combine_batch_with_single(data, augmented_single)
    augmented_data_list = [augmented_data[key] for key in fields]
    augmented_fx_model = _model_make_fx(
        model, 
        augmented_data_list + extra_inputs
        )
    
    del augmented_data, augmented_single, single_frame, node_idx

    # because we use different batch dims for each fx model,
    # check that the fx graphs are identical to ensure that `make_fx` didn't shape-specialize
    check_make_fx_diff(fx_model, augmented_fx_model, fields)

    # clean up
    torch.cuda.empty_cache()

    return fx_model


seed = 42

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
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
        )
    ],
    batch_size=2,
    shuffle=False,
    drop_last=False,
)
avg_num_neighbors = compute_avg_num_neighbors(
    data_loader
)

batch = next(iter(data_loader)).to(device)

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

model_conig = dict(
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
)

#compiled_model = CompiledModel(
#    model=model,
#    model_config=model_conig
#)
#compiled_model(batch)




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



# torch._dynamo.config.capture_dynamic_output_shape_ops = True
fx = True
if fx:
    fx_model = model_make_fx(
        model=model_to_trace,
        data=batch,
        fields=input_fields,
        extra_inputs=weights + buffers,
        seed=seed,
    )
    compiled_model = torch.compile(
        fx_model,
        dynamic=True,
        fullgraph=True
    )
    # compiled_model(batch)

    out = compiled_model(
        *(test_data_list + weights + buffers)
    )
else:
    compiled_model = torch.compile(
        model,
        dynamic=True,
        fullgraph=False
    )
    out = compiled_model(batch)

print("Compiled model output:")
print(out, "\n")
out = model(batch)
print("Original model output:")
print(out)
exit()

# print(out)
# print(fx_model)