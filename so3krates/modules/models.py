import torch
from typing import Any, Callable, Dict, List, Optional, Type, Union
from e3nn import o3
from e3nn.util.jit import compile_mode



@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)

@compile_mode("script")
class So3krates(torch.nn.Module):
    def __init__(   
        self,
        r_max,
        num_radial_basis,
        max_l,
        features_dim,
        euclidean_variables_dim,
        num_heads,
    ):
        super().__init__()


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
        ) -> Dict[str, Optional[torch.Tensor]]::
        return self.model(x)
