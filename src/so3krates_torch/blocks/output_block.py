import torch
import torch.nn as nn
from so3krates_torch.tools import scatter
from typing import Callable, Dict, Optional


class AtomicEnergyOutputHead(nn.Module):
    def __init__(
        self,
        features_dim: int,
        energy_regression_dim: Optional[int] = None,
        final_output_features: int = 1,
        layers: int = 2,
        bias: bool = True,
        use_non_linearity: bool = True,
        non_linearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        final_non_linearity: bool = False,
        atomic_type_shifts: Optional[Dict[str, float]] = None,
        learn_atomic_type_shifts: bool = False,
        learn_atomic_type_scales: bool = False,
        num_elements: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if energy_regression_dim is None:
            energy_regression_dim = features_dim

        for _ in range(layers - 1):
            self.layers.append(
                nn.Linear(features_dim, energy_regression_dim, bias=bias)
            )
        self.final_layer = nn.Linear(
            energy_regression_dim, final_output_features, bias=bias
        )
        # make sure non_linearity is not None if use_non_linearity is True
        if use_non_linearity and non_linearity is None:
            raise ValueError(
                "If use_non_linearity is True, non_linearity must be provided."
            )
        self.use_non_linearity = use_non_linearity
        self.non_linearity = non_linearity()
        self.final_non_linearity = final_non_linearity
        if learn_atomic_type_shifts:
            assert (
                atomic_type_shifts is None
            ), "If learn_atomic_type_shifts is True, atomic_type_shifts must be None."
        if learn_atomic_type_shifts or learn_atomic_type_scales:
            assert num_elements is not None, (
                "If learn_atomic_type_shifts or learn_atomic_type_scales is True, "
                "num_elements must be provided."
            )

        self.learn_atomic_type_shifts = learn_atomic_type_shifts
        self.learn_atomic_type_scales = learn_atomic_type_scales

        if self.learn_atomic_type_shifts:
            self.energy_shifts = nn.Linear(
                num_elements, 1, bias=False, dtype=torch.get_default_dtype()
            )
            nn.init.zeros_(self.energy_shifts.weight)
        if self.learn_atomic_type_scales:
            self.energy_scales = nn.Linear(
                num_elements, 1, bias=False, dtype=torch.get_default_dtype()
            )
            nn.init.ones_(self.energy_scales.weight)

        self.use_defined_shifts = False
        if atomic_type_shifts is not None:
            self.use_defined_shifts = True
            # so3lr uses 119 elements
            # starts from 0 index (element zero from mass effect maybe)
            if atomic_type_shifts.get("0") is not None:
                atomic_type_shifts.pop("0")
            if num_elements is not None:
                assert len(atomic_type_shifts) == num_elements
            # sort atomic_type_shifts based on element number (keys)
            atomic_type_shifts = {
                k: atomic_type_shifts[k]
                for k in sorted(atomic_type_shifts.keys())
            }
            self.energy_shifts = nn.Parameter(
                torch.tensor(
                    list(atomic_type_shifts.values()),
                    dtype=torch.get_default_dtype(),
                    requires_grad=False,
                )
            )

    def forward(
        self,
        inv_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            inv_features = layer(inv_features)
            if self.use_non_linearity:
                inv_features = self.non_linearity(inv_features)

        atomic_energies = self.final_layer(inv_features)
        if self.final_non_linearity:
            atomic_energies = self.non_linearity(inv_features)

        if self.learn_atomic_type_shifts:
            atomic_energies += self.energy_shifts(data["node_attrs"])
        if self.learn_atomic_type_scales:
            atomic_energies *= self.energy_scales(data["node_attrs"])

        if self.use_defined_shifts:
            assert (
                atomic_numbers is not None
            ), "If use_defined_shifts is True, atomic_numbers must be provided."
            atomic_indices = atomic_numbers - 1
            atomic_energies += self.energy_shifts[atomic_indices].unsqueeze(1)

        return atomic_energies


class PartialChargesOutputHead(nn.Module):
    def __init__(
        self,
        num_features: int,
        regression_dim: Optional[int] = None,
        activation_fn: torch.nn.Module = torch.nn.Identity,
    ):
        super().__init__()

        self.regression_dim = regression_dim
        self.activation_fn = activation_fn

        # Element-dependent bias embedding (for atomic numbers up to 100)
        self.atomic_embedding = nn.Embedding(
            num_embeddings=100, embedding_dim=1
        )

        # Build the network layers
        if self.regression_dim is not None:
            self.transform_inv_features = nn.Sequential(
                nn.Linear(num_features, regression_dim),
                activation_fn(),
                nn.Linear(regression_dim, 1),
            )
        else:
            self.transform_inv_features = nn.Linear(num_features, 1)

    def forward(
        self,
        inv_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        batch_segments: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:

        # q_ - element-dependent bias
        q_ = self.atomic_embedding(atomic_numbers).squeeze(-1)
        x_ = self.transform_inv_features(inv_features).squeeze(-1)
        x_q = x_ + q_

        total_charge_predicted = scatter.scatter_sum(
            src=x_q, index=batch_segments, dim=0, dim_size=num_graphs
        )  # (num_graphs)

        unique_batches, counts = torch.unique(
            batch_segments, return_counts=True
        )
        number_of_atoms_in_molecule = torch.zeros(
            num_graphs, dtype=counts.dtype, device=inv_features.device
        )
        number_of_atoms_in_molecule[unique_batches] = counts

        charge_conservation = (1 / number_of_atoms_in_molecule) * (
            total_charge - total_charge_predicted
        )

        # Repeat charge conservation for each atom in molecule
        partial_charges = x_q + charge_conservation[batch_segments]
        return partial_charges


class DipoleVecOutputHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        partial_charges: torch.Tensor,
        positions: torch.Tensor,
        batch_segments: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:

        mu_i = positions * partial_charges[:, None]

        # Compute the total dipole moment for each molecule
        total_dipole = scatter.scatter_sum(
            src=mu_i, index=batch_segments, dim=0, dim_size=num_graphs
        )

        return total_dipole


class HirshfeldOutputHead(nn.Module):
    def __init__(
        self,
        num_features: int,
        regression_dim: Optional[int] = None,
        activation_fn: torch.nn.Module = torch.nn.Identity,
    ):
        super().__init__()

        self.regression_dim = regression_dim
        self.activation_fn = activation_fn
        self.num_features = num_features

        self.v_shift_embedding = nn.Embedding(
            num_embeddings=100, embedding_dim=1
        )

        self.q_embedding = nn.Embedding(
            num_embeddings=100, embedding_dim=num_features // 2
        )

        # Build the network layers for k
        if self.regression_dim is not None:
            self.transform_features = nn.Sequential(
                nn.Linear(num_features, regression_dim // 2),
                activation_fn(),
                nn.Linear(regression_dim // 2, num_features // 2),
            )
        else:
            self.transform_features = nn.Linear(
                num_features, num_features // 2
            )

    def forward(
        self,
        inv_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict Hirshfeld ratios from atom-wise features and atomic types.

        Args:
            inv_features: Atomic features, shape: (num_nodes, num_features)
            atomic_numbers: Atomic types, shape: (num_nodes)
            node_mask: Node mask, shape: (num_nodes), optional

        Returns:
            hirshfeld_ratios: Predicted Hirshfeld ratios, shape: (num_nodes)
        """

        v_shift = self.v_shift_embedding(atomic_numbers).squeeze(
            -1
        )  # (num_nodes)

        q = self.q_embedding(atomic_numbers)  # (num_nodes, num_features//2)

        k = self.transform_features(
            inv_features
        )  # (num_nodes, num_features//2)

        qk = (
            q * k / torch.sqrt(torch.tensor(k.shape[-1], dtype=k.dtype))
        ).sum(dim=-1)

        v_eff = v_shift + qk  # (num_nodes)

        hirshfeld_ratios = torch.abs(v_eff)

        return hirshfeld_ratios
