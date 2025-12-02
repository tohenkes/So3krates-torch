import os
from pathlib import Path
import torch
from so3krates_torch.data.atomic_data import AtomicData as So3Data
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools, utils
import h5py
import numpy as np
from typing import Optional, Tuple, List
from mace.modules.utils import (
    compute_forces,
    compute_forces_virials,
    compute_hessians_vmap,
)
from torch.func import jacrev

activation_fn_dict = {
    "silu": torch.nn.SiLU,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "identity": torch.nn.Identity,
}


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(
            torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
        )
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def compute_multihead_forces(energy, positions, batch, training=True):
    num_graphs, num_heads = energy.shape
    # change shape to [num_heads,num_graphs]
    energy_for_grad = energy.view(num_graphs, num_heads).permute(1, 0)
    grad_outputs = torch.zeros(
        num_heads,
        num_heads,
        num_graphs,
        device=energy.device,
        dtype=energy.dtype,
    )
    # picks the gradient for a specific head
    eye_for_heads = torch.eye(
        num_heads, device=energy.device, dtype=energy.dtype
    )
    # picks the gradient for a specific graph and head
    grad_outputs[:, :, :] = eye_for_heads.unsqueeze(-1).expand(
        -1, -1, num_graphs
    )

    grad_all = torch.autograd.grad(
        outputs=[energy_for_grad],
        inputs=[positions],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
        is_grads_batched=True,  # treat the first dim (heads) as batch
    )[0]
    forces = -grad_all
    return forces


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
    is_multihead: bool = False,
    batch: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian or compute_edge_forces),
        )
    elif compute_force:
        if is_multihead:
            forces, virials, stress = (
                compute_multihead_forces(
                    energy=energy,
                    positions=positions,
                    training=(
                        training or compute_hessian or compute_edge_forces
                    ),
                    batch=batch,
                ),
                None,
                None,
            )
        else:
            forces, virials, stress = (
                compute_forces(
                    energy=energy,
                    positions=positions,
                    training=(
                        training or compute_hessian or compute_edge_forces
                    ),
                ),
                None,
                None,
            )
    else:
        forces, virials, stress = (None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    if compute_edge_forces and vectors is not None:
        edge_forces = compute_forces(
            energy=energy,
            positions=vectors,
            training=(training or compute_hessian),
        )
        if edge_forces is not None:
            edge_forces = -1 * edge_forces  # Match LAMMPS sign convention
    else:
        edge_forces = None
    return forces, virials, stress, hessian, edge_forces


def load_results_hdf5(filename, is_ensemble: bool = False):
    """Load results from HDF5 format."""
    loaded_data = {}

    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if key == "att_scores":
                # Special handling for attention scores
                if is_ensemble:
                    # Ensemble attention scores
                    att_models = {}
                    att_grp = f[key]

                    for model_name in att_grp.keys():
                        model_grp = att_grp[model_name]
                        att_list = []

                        item_names = sorted(
                            [
                                name
                                for name in model_grp.keys()
                                if name.startswith("item_")
                            ]
                        )
                        for item_name in item_names:
                            item_grp = model_grp[item_name]
                            att_dict = {}

                            # Load 'ev' and 'inv' nested dictionaries
                            for key_type in ["ev", "inv"]:
                                if key_type in item_grp:
                                    key_grp = item_grp[key_type]
                                    att_dict[key_type] = {}
                                    for layer_idx in key_grp.keys():
                                        att_dict[key_type][int(layer_idx)] = (
                                            key_grp[layer_idx][()]
                                        )

                            # Load 'senders' and 'receivers' tensors
                            for key_type in ["senders", "receivers"]:
                                if key_type in item_grp:
                                    att_dict[key_type] = item_grp[key_type][()]

                            att_list.append(att_dict)

                        att_models[model_name] = att_list

                    loaded_data[key] = att_models
                else:
                    # Single model attention scores
                    att_list = []
                    att_grp = f[key]

                    item_names = sorted(
                        [
                            name
                            for name in att_grp.keys()
                            if name.startswith("item_")
                        ]
                    )
                    for item_name in item_names:
                        item_grp = att_grp[item_name]
                        att_dict = {}

                        # Load 'ev' and 'inv' nested dictionaries
                        for key_type in ["ev", "inv"]:
                            if key_type in item_grp:
                                key_grp = item_grp[key_type]
                                att_dict[key_type] = {}
                                for layer_idx in key_grp.keys():
                                    att_dict[key_type][int(layer_idx)] = (
                                        key_grp[layer_idx][()]
                                    )

                        # Load 'senders' and 'receivers' tensors
                        for key_type in ["senders", "receivers"]:
                            if key_type in item_grp:
                                att_dict[key_type] = item_grp[key_type][()]

                        att_list.append(att_dict)

                    loaded_data[key] = att_list

            elif is_ensemble:
                # Handle ensemble results for other keys
                loaded_data[key] = {}
                if isinstance(f[key], h5py.Group):
                    ensemble_grp = f[key]
                    for model_name in ensemble_grp.keys():
                        model_grp = ensemble_grp[model_name]
                        if isinstance(model_grp, h5py.Group):
                            # Multiple items per model
                            if "result" in model_grp:
                                # Single result per model
                                loaded_data[key][model_name] = model_grp[
                                    "result"
                                ][()]
                            else:
                                # Multiple items per model
                                items = []
                                item_names = sorted(model_grp.keys())
                                for item_name in item_names:
                                    items.append(model_grp[item_name][()])
                                loaded_data[key][model_name] = items
                        else:
                            # Single item per model
                            loaded_data[key][model_name] = model_grp[()]

            else:
                # Single model results for other keys
                if isinstance(f[key], h5py.Group):
                    # List of items
                    items = []
                    item_names = sorted(f[key].keys())
                    for item_name in item_names:
                        items.append(f[key][item_name][()])
                    loaded_data[key] = items
                else:
                    # Single item or None
                    if "is_none" in f[key].attrs:
                        loaded_data[key] = None
                    else:
                        loaded_data[key] = f[key][()]

    return loaded_data


def save_results_hdf5(results, filename, is_ensemble: bool = False):
    with h5py.File(filename, "w") as f:
        for k, v in results.items():
            if v is not None:
                if k == "att_scores":
                    continue
                    # Special handling for attention scores
                    if is_ensemble:
                        # Ensemble attention scores: list of lists of dicts
                        att_grp = f.create_group(k)
                        for model_idx, model_att_scores in enumerate(v):
                            model_grp = att_grp.create_group(
                                f"model_{model_idx}"
                            )
                            for i, att_dict in enumerate(model_att_scores):
                                item_grp = model_grp.create_group(
                                    f"item_{i:06d}"
                                )

                                # Handle 'ev' and 'inv' nested dictionaries
                                for key_type in ["ev", "inv"]:
                                    if key_type in att_dict:
                                        key_grp = item_grp.create_group(
                                            key_type
                                        )
                                        for layer_idx, tensor in att_dict[
                                            key_type
                                        ].items():
                                            if isinstance(
                                                tensor, torch.Tensor
                                            ):
                                                tensor_data = (
                                                    tensor.detach()
                                                    .cpu()
                                                    .numpy()
                                                )
                                            else:
                                                tensor_data = np.array(tensor)
                                            key_grp.create_dataset(
                                                str(layer_idx),
                                                data=tensor_data,
                                            )

                                # Handle 'senders' and 'receivers' tensors
                                for key_type in ["senders", "receivers"]:
                                    if key_type in att_dict:
                                        if isinstance(
                                            att_dict[key_type], torch.Tensor
                                        ):
                                            tensor_data = (
                                                att_dict[key_type]
                                                .detach()
                                                .cpu()
                                                .numpy()
                                            )
                                        else:
                                            tensor_data = np.array(
                                                att_dict[key_type]
                                            )
                                        item_grp.create_dataset(
                                            key_type, data=tensor_data
                                        )
                    else:
                        # Single model attention scores: list of dicts
                        att_grp = f.create_group(k)
                        for i, att_dict in enumerate(v):
                            item_grp = att_grp.create_group(f"item_{i:06d}")

                            # Handle 'ev' and 'inv' nested dictionaries
                            for key_type in ["ev", "inv"]:
                                if key_type in att_dict:
                                    key_grp = item_grp.create_group(key_type)
                                    for layer_idx, tensor in att_dict[
                                        key_type
                                    ].items():
                                        if isinstance(tensor, torch.Tensor):
                                            tensor_data = (
                                                tensor.detach().cpu().numpy()
                                            )
                                        else:
                                            tensor_data = np.array(tensor)
                                        key_grp.create_dataset(
                                            str(layer_idx), data=tensor_data
                                        )

                            # Handle 'senders' and 'receivers' tensors
                            for key_type in ["senders", "receivers"]:
                                if key_type in att_dict:
                                    if isinstance(
                                        att_dict[key_type], torch.Tensor
                                    ):
                                        tensor_data = (
                                            att_dict[key_type]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                    else:
                                        tensor_data = np.array(
                                            att_dict[key_type]
                                        )
                                    item_grp.create_dataset(
                                        key_type, data=tensor_data
                                    )

                elif is_ensemble:
                    # Handle ensemble results for other keys
                    ensemble_grp = f.create_group(k)
                    for model_idx, model_results in enumerate(v):
                        model_grp = ensemble_grp.create_group(
                            f"model_{model_idx}"
                        )
                        if isinstance(model_results, list):
                            for j, result in enumerate(model_results):
                                if isinstance(result, torch.Tensor):
                                    result = result.detach().cpu().numpy()
                                model_grp.create_dataset(
                                    f"item_{j:06d}", data=result
                                )
                        else:
                            # Single result per model
                            if isinstance(model_results, torch.Tensor):
                                model_results = (
                                    model_results.detach().cpu().numpy()
                                )
                            model_grp.create_dataset(
                                "result", data=model_results
                            )

                else:
                    # Single model results for other keys
                    if isinstance(v, list):
                        grp = f.create_group(k)
                        for i, result in enumerate(v):
                            if isinstance(result, torch.Tensor):
                                result = result.detach().cpu().numpy()
                            elif isinstance(result, list):
                                result = np.array(result)
                            grp.create_dataset(f"item_{i:06d}", data=result)
                    else:
                        # Single array/tensor
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu().numpy()
                        f.create_dataset(k, data=v)
            else:
                # Store None as an empty dataset with attribute
                dset = f.create_dataset(k, data=np.array([]))
                dset.attrs["is_none"] = True


def ensemble_from_folder(path_to_models: str, device: str, dtype: str) -> dict:
    """
    Load an ensemble of models from a folder.

    Args:
        path_to_models (str): Path to the folder containing the models.
        device (str): Device to load the models on.
        dtype (str): Data type to load the models as.

    Returns:
        dict: Dictionary of models.
    """

    assert os.path.exists(path_to_models)
    assert os.listdir(Path(path_to_models))

    ensemble = {}
    for filename in os.listdir(path_to_models):
        if os.path.isfile(os.path.join(path_to_models, filename)):
            complete_path = os.path.join(path_to_models, filename)
            model = torch.load(
                complete_path, map_location=device, weights_only=False
            ).to(dtype)
            filename_without_suffix = os.path.splitext(filename)[0]
            ensemble[filename_without_suffix] = model
    return ensemble

def create_configs_from_list(
    atoms_list: list,
    key_specification: mace_data.utils.KeySpecification = mace_data.utils.KeySpecification(),
    head_name: str = None,
):
    configs = [
        mace_data.config_from_atoms(
            atoms, 
            key_specification=key_specification,
            head_name=head_name
            )
        for atoms in atoms_list
    ]
    return configs

def create_data_from_configs(
    config_list: list,
    r_max: float,
    r_max_lr: float,
    all_heads: list = None,
    z_table: utils.AtomicNumberTable = None,
):

    if z_table is None:
        z_table = utils.AtomicNumberTable([int(z) for z in range(1, 119)])
    return [
        So3Data.from_config(
            config,
            z_table=z_table,
            cutoff=float(r_max),
            cutoff_lr=r_max_lr,
            heads=all_heads,
        )
        for config in config_list
    ]   

def create_data_from_list(
    atoms_list: list,
    r_max: float,
    r_max_lr: float,
    head_name: str = None,
    all_heads: list = None,
    key_specification: mace_data.utils.KeySpecification = mace_data.utils.KeySpecification(),
    z_table: utils.AtomicNumberTable = None,
):

    configs = create_configs_from_list(
        atoms_list,
        key_specification=key_specification,
        head_name=head_name,
    )
    return create_data_from_configs(
        configs,
        r_max,
        r_max_lr,
        all_heads=all_heads,
        z_table=z_table,
    )


def create_dataloader_from_list(
    atoms_list: list,
    batch_size: int,
    r_max: float,
    r_max_lr: float,
    key_specification: mace_data.utils.KeySpecification = mace_data.utils.KeySpecification(),
    shuffle: bool = False,
    drop_last: bool = False,
    z_table: utils.AtomicNumberTable = None,
    head_name: str = None,
):
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=create_data_from_list(
            atoms_list,
            r_max,
            r_max_lr,
            key_specification=key_specification,
            z_table=z_table,
            head_name=head_name,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return data_loader


def create_dataloader_from_data(
    config_list: list,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
):
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=config_list,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return data_loader
