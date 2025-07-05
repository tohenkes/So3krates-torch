import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional
from so3krates_torch.tools import scatter


@compile_mode("script")
class InvariantEmbedding(torch.nn.Module):
    '''
    Eq. 10 in https://doi.org/10.1038/s41467-024-50620-6
    '''
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False
    ):
        super().__init__()
        self.embedding = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        return self.embedding(one_hot)

    def reset_parameters(self):
        self.embedding.reset_parameters()


@compile_mode("script")
class EuclideanEmbedding(torch.nn.Module):
    '''
    Eq. 11 in https://doi.org/10.1038/s41467-024-50620-6
    
    In the paper it is said that the initialization is done trough
    Eq. 11, as stated above. However, in the code it is hardcoded to
    be zeros.
    '''
    def __init__(
        self,
        initialization_to_zeros: bool = True,
    ):
        super().__init__()
        self.initialization_to_zeros = initialization_to_zeros
        
    def forward(
        self,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        receivers: torch.Tensor,
        inv_avg_num_neighbors: float,
    ) -> torch.Tensor:

        if self.initialization_to_zeros:
            num_nodes = len(set(receivers.tolist()))
            ev_embedding = torch.zeros(
                (num_nodes, sh_vectors.shape[1]),
                dtype=sh_vectors.dtype,
                device=sh_vectors.device
            )
        else:
            # This is actually Eq. 11 in the paper:
            scaled_neighbors = sh_vectors * cutoffs
            ev_embedding = (
                scatter.scatter_sum(
                    src=scaled_neighbors,
                    index=receivers,
                    dim=0,
                )
                * inv_avg_num_neighbors
            )

        return ev_embedding
