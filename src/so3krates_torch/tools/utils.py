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


def load_results_hdf5(path_to_file: str, ensemble_size: int = 1) -> dict:
    loaded_data = {}
    with h5py.File(path_to_file, "r") as f:
        if ensemble_size > 1:
            for key in f.keys():
                loaded_data[key] = {}
                if isinstance(f[key], h5py.Group):
                    for i in range(ensemble_size):
                        model_name = f"model_{i}"
                        loaded_data[key][model_name] = [
                            f[key][model_name][item][()]
                            for item in f[key][model_name].keys()
                        ]

        else:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    loaded_data[key] = [
                        f[key][item][()] for item in f[key].keys()
                    ]
                else:
                    loaded_data[key] = f[key][()]

    return loaded_data


def save_results_hdf5(results: dict, filename: str, is_ensemble: bool = False):
    with h5py.File(filename, "w") as f:
        for k, v in results.items():
            if v is not None:
                if is_ensemble:
                    for i, ensemble_result in enumerate(v):
                        model_id = i
                        grp = f.create_group(f"{k}/model_{model_id}")
                        for j, result in enumerate(ensemble_result):
                            grp.create_dataset(f"item_{j:06d}", data=result)

                else:
                    grp = f.create_group(k)
                    for i, result in enumerate(v):
                        grp.create_dataset(f"item_{i:06d}", data=result)

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
