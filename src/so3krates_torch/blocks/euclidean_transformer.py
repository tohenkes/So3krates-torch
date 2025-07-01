import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Any, Callable, Dict, List, Optional, Type, Union
from so3krates_torch.blocks.so3_conv_invariants import (
    SO3ConvolutionInvariants
) 


@compile_mode("script")
class EuclideanTransformer(torch.nn.Module):
    # create dummy class
    def __init__(
        self,
        max_l: int,
        features_dim: int,
        interaction_bias: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        
        self.euclidean_attention_block = EuclideanAttentionBlock()
        self.interaction_block = InteractionBlock(
            max_l=max_l,
            features_dim=features_dim,
            bias=interaction_bias,
            device=device
        )

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        ) -> torch.Tensor:
        
        att_inv_features, att_ev_features = self.euclidean_attention_block(
            inv_features, ev_features
        )
        
        d_inv_features, d_ev_features = self.interaction_block(
            att_inv_features, att_ev_features
        )
        
        inv_features += d_inv_features
        ev_features += d_ev_features
        
        return inv_features, ev_features
        
        


@compile_mode("script")
class EuclideanAttentionBlock(torch.nn.Module):
    # create dummy class
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        ) -> torch.Tensor:
        
        # dummy attention
        return inv_features, ev_features
    

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        max_l: int,
        features_dim: int,
        bias: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
       
       
        self.max_l = max_l
        self.linear_layer = torch.nn.Linear(
            in_features=features_dim + max_l + 1,
            out_features=features_dim + max_l + 1,
            bias=bias
        )
        self.so3_conv_invariants = SO3ConvolutionInvariants(
            max_l=max_l
        )
        # repeat the b_ev_features for each degree
        # e.g. for max_l=2, we have repeats = [1, 3, 5]
        self.repeats = torch.tensor(
            [2 * y + 1 for y in range(self.max_l + 1)],
        ).to(device)
        
    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
    ) -> torch.Tensor:
        
        ev_invariants = self.so3_conv_invariants(
            ev_features, ev_features
        )
        cat_features = torch.concatenate(
            [inv_features, ev_invariants], dim=-1
        )
        # combine features
        transformed_features = self.linear_layer(cat_features)
        # split the features back
        d_inv_features, b_ev_features = torch.split(
            transformed_features,
            [inv_features.shape[-1], ev_invariants.shape[-1]],
            dim=-1
        )
        b_ev_features = torch.repeat_interleave(
            b_ev_features, self.repeats, dim=-1
        )
        d_ev_features = b_ev_features * ev_features
        return d_inv_features, d_ev_features
