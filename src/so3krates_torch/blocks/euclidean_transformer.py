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
        
        self.euclidean_attention_block = EuclideanAttentionBlock()
        self.interaction_block = InteractionBlock()

    def forward(
        self,
        invariant_features: torch.Tensor,
        euclidean_features: torch.Tensor,
        ) -> torch.Tensor:
        
        att_invariant_features, att_euclidean_features = self.euclidean_attention_block(
            invariant_features, euclidean_features
        )
        
        inter_invariant_features, inter_euclidean_features = self.interaction_block(
            att_invariant_features, att_euclidean_features
        )
        return inter_invariant_features, inter_euclidean_features
        
        


@compile_mode("script")
class EuclideanAttentionBlock(torch.nn.Module):
    # create dummy class
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        invariant_features: torch.Tensor,
        euclidean_features: torch.Tensor,
        ) -> torch.Tensor:
        
        # dummy attention
        return invariant_features, euclidean_features
    

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    # create dummy class
    def __init__(self):
        super().__init__()

    def forward(
        self,
        invariant_features: torch.Tensor,
        euclidean_features: torch.Tensor,
        ) -> torch.Tensor:
        
        # dummy interaction
        return invariant_features, euclidean_features