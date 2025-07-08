import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional
from so3krates_torch.tools import scatter

@compile_mode("script")
class EnergyOutputHead(torch.nn.Module):
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
        learn_atomic_type_shifts: bool = False,
        learn_atomic_type_scales: bool = False,
        num_elements: Optional[int] = None,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if energy_regression_dim is None:
            energy_regression_dim = features_dim
        
        for _ in range(layers - 1):
            self.layers.append(
                torch.nn.Linear(features_dim, energy_regression_dim, bias=bias)
            )
        self.final_layer = torch.nn.Linear(
            energy_regression_dim, final_output_features, bias=bias
        )
        # make sure non_linearity is not None if use_non_linearity is True
        if use_non_linearity and non_linearity is None:
            raise ValueError(
                "If use_non_linearity is True, non_linearity must be provided."
            )
        self.use_non_linearity = use_non_linearity
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity
        if (
            learn_atomic_type_shifts or learn_atomic_type_scales
        ):
            assert num_elements is not None, (
                "If learn_atomic_type_shifts or learn_atomic_type_scales is True, "
                "num_elements must be provided."
            )

        self.learn_atomic_type_shifts = learn_atomic_type_shifts
        self.learn_atomic_type_scales = learn_atomic_type_scales
        
        if self.learn_atomic_type_shifts:
            self.energy_shifts = torch.nn.Linear(
                num_elements, 1, bias=False, dtype=torch.get_default_dtype()
            )
            torch.nn.init.zeros_(self.energy_shifts.weight)
        if self.learn_atomic_type_scales:
            self.energy_scales = torch.nn.Linear(
                num_elements, 1, bias=False, dtype=torch.get_default_dtype()
            )
            torch.nn.init.ones_(self.energy_scales.weight)
            

    def forward(
        self,
        inv_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
        num_graphs: int,
            ) -> torch.Tensor:
        
        for layer in self.layers:
            inv_features = layer(inv_features)
            if self.use_non_linearity:
                inv_features = self.non_linearity(inv_features)
                
        atomic_energies = self.final_layer(inv_features)
        if self.final_non_linearity:
            atomic_energies = self.non_linearity(inv_features)

        if self.learn_atomic_type_shifts:
            atomic_energies += self.energy_shifts(data['node_attrs'])
        if self.learn_atomic_type_scales:
            atomic_energies *= self.energy_scales(data['node_attrs'])

        total_energy = scatter.scatter_sum(
            src=atomic_energies,
            index=data["batch"],
            dim=0, dim_size=num_graphs
        )
        return total_energy
