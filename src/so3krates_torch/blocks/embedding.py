import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional
from so3krates_torch.tools import scatter


@compile_mode("script")
class InvariantEmbedding(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False
    ):
        super().__init__()
        self.embedding = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x["node_attrs"])

    def reset_parameters(self):
        self.embedding.reset_parameters()


@compile_mode("script")
class EuclideanEmbedding(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        receivers: torch.Tensor,
        inv_avg_num_neighbors: float,
    ) -> torch.Tensor:

        scaled_neighbors = sh_vectors * cutoffs
        sum_scaled_neighbors = (
            scatter.scatter_sum(
                src=scaled_neighbors,
                index=receivers,
                dim=0,
            )
            * inv_avg_num_neighbors
        )
        return sum_scaled_neighbors
