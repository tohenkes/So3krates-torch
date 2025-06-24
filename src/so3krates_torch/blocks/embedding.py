import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional
from so3krates_torch.tools import scatter


@compile_mode("script")
class InvariantEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False):
        super().__init__()
        self.embedding = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x['node_attrs'])

    def reset_parameters(self):
        self.embedding.reset_parameters()
        

@compile_mode("script")
class EuclideanEmbedding(torch.nn.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        max_l: int,
        cutoff_function: torch.nn.Module,
        use_so3: bool = False,
        normalize: bool = True,
        normalization: str = "component",
        
        ):
        super().__init__()
        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_l)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_l, p=1)

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps,
            normalize=normalize,
            normalization=normalization,
        )
        self.inv_avg_num_neighbors = 1. / avg_num_neighbors
        self.cutoff_function = cutoff_function

    def forward(
        self,
        senders,
        receivers,
        lengths: torch.Tensor, 
        vectors: torch.Tensor
        ) -> torch.Tensor:

        edge_embedding = self.spherical_harmonics(
            vectors
        )
        scaled_neighbors = senders * self.cutoff_function(lengths)
        sum_scaled_neighbors = scatter.scatter_sum(
            src=scaled_neighbors,
            index=receivers,
            dim=0,
            dim_size=edge_embedding.shape[0]
        ) * self.inv_avg_num_neighbors
        
        return sum_scaled_neighbors