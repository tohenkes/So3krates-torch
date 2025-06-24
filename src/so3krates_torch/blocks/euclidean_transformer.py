import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional


@compile_mode("script")
class EuclideanTransformer(torch.nn.Module):
    # create dummy class
    def __init__(
        self,):
        super().__init__()

    def forward(
        self,
        invariant_features: torch.Tensor,
        euclidean_features: torch.Tensor,
        ) -> torch.Tensor:
        return invariant_features, euclidean_features
        
        
        


@compile_mode("script")
class EuclideanAttentionBlock(torch.nn.Module):
    # create dummy class
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    # create dummy class
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x