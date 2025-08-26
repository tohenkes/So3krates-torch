import torch
from typing import Optional, Dict
from mace.modules.loss import (
    weighted_mean_squared_error_energy,
    mean_squared_error_forces,
    weighted_mean_squared_error_dipole,
    reduce_loss
)
from torch_geometric.data import Batch

TensorDict = Dict[str, torch.Tensor]


def weighted_mean_squared_error_hirshfeld(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.hirshfeld_ratios_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(
        self, 
        energy_weight=1.0, 
        forces_weight=1.0,
        dipole_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesHirshfeldLoss(torch.nn.Module):
    def __init__(
        self, 
        energy_weight=1.0, 
        forces_weight=1.0,
        hirshfeld_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "hirshfeld_weight",
            torch.tensor(hirshfeld_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_hirshfeld = weighted_mean_squared_error_hirshfeld(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy + self.forces_weight * loss_forces
            + self.hirshfeld_weight * loss_hirshfeld
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"hirshfeld_weight={self.hirshfeld_weight:.3f})"
        )


class WeightedEnergyForcesDipoleHirshfeldLoss(torch.nn.Module):
    def __init__(
        self, 
        energy_weight=1.0, 
        forces_weight=1.0,
        dipole_weight=1.0,
        hirshfeld_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "hirshfeld_weight",
            torch.tensor(hirshfeld_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp)
        loss_hirshfeld = weighted_mean_squared_error_hirshfeld(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
            + self.hirshfeld_weight * loss_hirshfeld
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"dipole_weight={self.dipole_weight:.3f}, "
            f"hirshfeld_weight={self.hirshfeld_weight:.3f})"
        )
