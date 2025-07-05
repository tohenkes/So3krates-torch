import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from so3krates_torch.blocks.so3_conv_invariants import SO3ConvolutionInvariants
import math
from so3krates_torch.tools import scatter


@compile_mode("script")
class EuclideanTransformer(torch.nn.Module):
    # create dummy class
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        features_dim: int,
        num_radial_basis: int,
        interaction_bias: bool = True,
        message_normalization: str = "sqrt_num_features",
        avg_num_neighbors: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        filter_net_inv_layers: int = 2,
        filter_net_inv_non_linearity: Type[torch.nn.Module] = torch.nn.SiLU,
        filter_net_ev_layers: int = 2,
        filter_net_ev_non_linearity: Type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()

        self.filter_net_inv = FilterNet(
            degrees=degrees,
            features_dim=features_dim,
            num_radial_basis=num_radial_basis,
            num_layers=filter_net_inv_layers,
            non_linearity=filter_net_inv_non_linearity,
        )
        
        self.filter_net_ev = FilterNet(
            degrees=degrees,
            features_dim=features_dim,
            num_radial_basis=num_radial_basis,
            num_layers=filter_net_ev_layers,
            non_linearity=filter_net_ev_non_linearity,
        )
        
        self.euclidean_attention_block = EuclideanAttentionBlock(
            degrees=degrees,
            num_heads=num_heads,
            features_dim=features_dim,
            filter_net_inv=self.filter_net_inv,
            filter_net_ev=self.filter_net_ev,
            device=device,
            message_normalization=message_normalization,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interaction_block = InteractionBlock(
            degrees=degrees,
            features_dim=features_dim,
            bias=interaction_bias,
            device=device,
        )

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        rbf: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        d_att_inv_features, d_att_ev_features = self.euclidean_attention_block(
            inv_features,
            ev_features,
            rbf=rbf,
            senders=senders,
            receivers=receivers,
            sh_vectors=sh_vectors,
            cutoffs=cutoffs,
        )
        att_inv_features = inv_features + d_att_inv_features
        att_ev_features = ev_features + d_att_ev_features
        d_inv_features, d_ev_features = self.interaction_block(
            att_inv_features, att_ev_features
        )
        # Eq. 26, 27 in https://doi.org/10.1038/s41467-024-50620-6
        new_inv_features = att_inv_features + d_inv_features
        new_ev_features = att_ev_features + d_ev_features
        return new_inv_features, new_ev_features


@compile_mode("script")
class EuclideanAttentionBlock(torch.nn.Module):
    '''
    Fig. 3c) in https://doi.org/10.1038/s41467-024-50620-6
    '''
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        features_dim: int,
        filter_net_inv: callable,
        filter_net_ev: callable,
        message_normalization: str = 'sqrt_num_features',
        qk_non_linearity=None,
        avg_num_neighbors: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.ev_features_dim = torch.sum(
            torch.tensor([2 * y + 1 for y in degrees]
                         )).item()
        self.inv_features_dim = features_dim
        self.so3_conv_invariants = SO3ConvolutionInvariants(degrees=degrees)

        self.filter_net_inv = filter_net_inv
        self.filter_net_ev = filter_net_ev
        

        # query, key, value weights for invariants
        self.inv_heads = num_heads
        self.inv_head_dim = features_dim // num_heads
        self.W_q_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads, self.inv_head_dim, self.inv_head_dim
            )
        )
        self.W_k_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads, self.inv_head_dim, self.inv_head_dim
            )
        )
        self.W_v_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads, self.inv_head_dim, self.inv_head_dim
            )
        )
        # query, key weights for ev features
        # no value weights, as it uses spherical harmonics as values
        self.ev_heads = len(degrees)
        self.ev_head_dim = features_dim // len(degrees)
        self.W_q_ev = torch.nn.Parameter(
            torch.empty(
                self.ev_heads, self.ev_head_dim, self.ev_head_dim
            )
        )
        self.W_k_ev = torch.nn.Parameter(
            torch.empty(
                self.ev_heads, self.ev_head_dim, self.ev_head_dim
            )
        )
        # initialize the weights
        sqrt_5 = math.sqrt(5)
        torch.nn.init.kaiming_uniform_(self.W_q_inv, a=sqrt_5)
        torch.nn.init.kaiming_uniform_(self.W_k_inv, a=sqrt_5)
        torch.nn.init.kaiming_uniform_(self.W_v_inv, a=sqrt_5)
        torch.nn.init.kaiming_uniform_(self.W_q_ev, a=sqrt_5)
        torch.nn.init.kaiming_uniform_(self.W_k_ev, a=sqrt_5)

        # normalization for attention weights
        # Eq. 21 https://doi.org/10.1038/s41467-024-50620-6
        assert message_normalization in [
            'sqrt_num_features',
            'identity',
            'avg_num_neighbors',
        ]
        if message_normalization == 'sqrt_num_features':
            self.att_norm_inv = math.sqrt(self.inv_head_dim)
            self.att_norm_ev = math.sqrt(self.ev_head_dim)
        elif message_normalization == 'identity':
            self.att_norm_inv = 1.0
            self.att_norm_ev = 1.0
        elif message_normalization == 'avg_num_neighbors':
            assert avg_num_neighbors is not None
            self.att_norm_inv = avg_num_neighbors
            self.att_norm_ev = avg_num_neighbors
        
        # non-linearity for query and key weights
        if qk_non_linearity is None:
            self.qk_non_linearity = torch.nn.Identity()
        
        self.degree_repeats = torch.tensor(
            [2 * y + 1 for y in degrees],
        ).to(device)
        
    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        rbf: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        #print(ev_features[0,:5])
        ev_differences = ev_features[senders] - ev_features[receivers]
        ev_differences_invariants = self.so3_conv_invariants(
            ev_differences, ev_differences
        )
        filter_w_inv = self.filter_net_inv(
            rbf,
            ev_differences_invariants
        )
        filter_w_ev = self.filter_net_ev(
            rbf,
            ev_differences_invariants
        )
        # split filter weights into heads
        # now has shape [neighbors, num_heads, inv_head_dim]
        filter_w_inv = filter_w_inv.view(
            -1, self.inv_heads, self.inv_head_dim
        )
        # now has shape [neighbors, len(degrees), ev_head_dim]
        filter_w_ev = filter_w_ev.view(
            -1, self.ev_heads, self.ev_head_dim
        )
        # split features into heads
        # first for invariants; has shape [nodes, num_heads, inv_head_dim] now
        inv_features_inv = inv_features.view(
            -1, self.inv_heads, self.inv_head_dim
        )
        # now for the invariants used for updating the ev features
        # has shape [nodes, len(degrees), ev_head_dim] now
        inv_features_ev = inv_features.view(
            -1, self.ev_heads, self.ev_head_dim
        )
        # computing the queries, keys, and values and immediately
        # selecting receivers (i) and senders (j) (Eq. 21 https://doi.org/10.1038/s41467-024-50620-6)
        q_inv = self.qk_non_linearity(torch.einsum(
            "Hij, NHi -> NHj",
            self.W_q_inv,
            inv_features_inv,
        ))[receivers]

        k_inv = self.qk_non_linearity(torch.einsum(
            "Hij, NHi -> NHj",
            self.W_k_inv,
            inv_features_inv,
        ))[senders]
        
        v_inv = torch.einsum(
            "Hij, NHi -> NHj",
            self.W_v_inv,
            inv_features_inv,
        )[senders]
        
        q_ev = self.qk_non_linearity(torch.einsum(
            "Hij, NHi -> NHj",
            self.W_q_ev,
            inv_features_ev,
        ))[receivers]
        
        k_ev = self.qk_non_linearity(torch.einsum(
            "Hij, NHi -> NHj",
            self.W_k_ev,
            inv_features_ev,
        ))[senders]
        
        # Eq. 21 https://doi.org/10.1038/s41467-024-50620-6
        filtered_k_inv = k_inv * filter_w_inv
        filtered_k_ev = k_ev * filter_w_ev
        
        alpha_inv = (q_inv * filtered_k_inv).sum(-1, keepdim=True) / self.att_norm_inv
        alpha_ev = (q_ev * filtered_k_ev).sum(-1) / self.att_norm_ev
        
        # Eq. 15 in https://doi.org/10.1038/s41467-024-50620-6
        scaled_neighbors_inv = cutoffs[:, None] * alpha_inv * v_inv
        d_h_att_inv_features = scatter.scatter_sum(
            src=scaled_neighbors_inv,
            index=receivers,
            dim=0,
        )
        d_att_inv_features = d_h_att_inv_features.view(-1, self.inv_features_dim)
        
        # already combining heads for ev here
        alpha_ev = torch.repeat_interleave(
            alpha_ev, self.degree_repeats, dim=-1
        )
        # Eq. 26 https://doi.org/10.1038/s41467-024-50620-6
        scaled_neighbors_ev = cutoffs * alpha_ev * sh_vectors
        d_att_ev_features = scatter.scatter_sum(
            src=scaled_neighbors_ev,
            index=receivers,
            dim=0,
        )
        return d_att_inv_features, d_att_ev_features


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    '''
    Fig. 3d) in https://doi.org/10.1038/s41467-024-50620-6
    '''
    def __init__(
        self,
        degrees: List[int],
        features_dim: int,
        bias: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        len_degrees = len(degrees)
        self.linear_layer = torch.nn.Linear(
            in_features=features_dim + len_degrees,
            out_features=features_dim + len_degrees,
            bias=bias,
        )
        self.so3_conv_invariants = SO3ConvolutionInvariants(degrees=degrees)
        # repeat the b_ev_features for each degree
        # e.g. for degrees=[0,1,2], we have repeats = [1, 3, 5]
        self.degree_repeats = torch.tensor(
            [2 * y + 1 for y in degrees],
        ).to(device)

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ev_invariants = self.so3_conv_invariants(ev_features, ev_features)
        
        # Eq. 25 in https://doi.org/10.1038/s41467-024-50620-6
        cat_features = torch.concatenate([inv_features, ev_invariants], dim=-1)
        # combine features
        transformed_features = self.linear_layer(cat_features)
        # split the features back
        d_inv_features, b_ev_features = torch.split(
            transformed_features,
            [inv_features.shape[-1], ev_invariants.shape[-1]],
            dim=-1,
        )
        b_ev_features = torch.repeat_interleave(
            b_ev_features, self.degree_repeats, dim=-1
        )
        # Eq. 24 in https://doi.org/10.1038/s41467-024-50620-6
        d_ev_features = b_ev_features * ev_features
        
        return d_inv_features, d_ev_features


@compile_mode("script")
class FilterNet(torch.nn.Module):
    '''
    Fig. 3e) in https://doi.org/10.1038/s41467-024-50620-6
    '''
    def __init__(
        self,
        degrees: List[int],
        num_radial_basis: int,
        features_dim: int,
        num_layers: int = 2,
        non_linearity: Type[torch.nn.Module] = torch.nn.SiLU
        ):
        super().__init__()
        

        
        assert features_dim % 4 == 0, (
            f"features_dim {features_dim} must be divisible by 4 for the EuclideanTransformer."
        )
        
        self.mlp_rbf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=num_radial_basis,
                out_features=features_dim,
            ),
            non_linearity(),
        )
        self.mlp_ev = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=len(degrees),
                out_features=features_dim // 4,
            ),
            non_linearity(),
        )
        for i in range(num_layers - 1):
            # last layer does not have a non-linearity
            if i == num_layers - 2:
                non_linearity = torch.nn.Identity
            self.mlp_rbf.add_module(
                f"mlp_rbf_layer_{i+1}",
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=features_dim,
                        out_features=features_dim,
                    ),
                    non_linearity(),
                )
            )
            if i == 0:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i+1}",
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            in_features=features_dim // 4,
                            out_features=features_dim,
                        ),
                        non_linearity(),
                    )
                )
            else:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i+1}",
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            in_features=features_dim,
                            out_features=features_dim,
                        ),
                        non_linearity(),
                    )
                )
        
    def forward(
        self,
        rbf: torch.Tensor,
        ev_difference_invariants: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. 20 in https://doi.org/10.1038/s41467-024-50620-6
        rbf_features = self.mlp_rbf(rbf)
        ev_features = self.mlp_ev(ev_difference_invariants)
        
        filter_weights = rbf_features + ev_features
        return filter_weights
