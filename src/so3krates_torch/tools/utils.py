import os
from pathlib import Path
import torch
from so3krates_torch.data.atomic_data import AtomicData as So3Data
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools, utils
import h5py
import numpy as np

activation_fn_dict = {
    "silu": torch.nn.SiLU,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "identity": torch.nn.Identity,
}


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


def create_dataloader(
    atoms_list: list,
    batch_size: int,
    r_max: float,
    r_max_lr: float,
    key_specification: mace_data.utils.KeySpecification = mace_data.utils.KeySpecification(),
    shuffle: bool = False,
    drop_last: bool = False,
    z_table: utils.AtomicNumberTable = None,
):

    configs = [
        mace_data.config_from_atoms(atoms, key_specification=key_specification)
        for atoms in atoms_list
    ]
    if z_table is None:
        z_table = utils.AtomicNumberTable([int(z) for z in range(1, 119)])
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            So3Data.from_config(
                config,
                z_table=z_table,
                cutoff=float(r_max),
                cutoff_lr=r_max_lr,
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return data_loader
