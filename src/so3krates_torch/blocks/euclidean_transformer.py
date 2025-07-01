import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Any, Callable, Dict, List, Optional, Type, Union
from so3krates_torch.blocks.so3_conv_invariants import SO3ConvolutionInvariants
from so3krates_torch.blocks import radial_basis

@compile_mode("script")
class EuclideanTransformer(torch.nn.Module):
    # create dummy class
    def __init__(
        self,
        max_l: int,
        features_dim: int,
        r_max: float,
        num_radial_basis: int,
        radial_basis_fn: str = "gaussian",
        trainable_rbf: bool = False,
        interaction_bias: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.filter_net_inv = FilterNet(
            max_l=max_l,
            features_dim=features_dim,
            r_max=r_max,
            num_radial_basis=num_radial_basis,
            radial_basis_fn=radial_basis_fn,
            trainable_rbf=trainable_rbf,
        )
        
        self.filter_net_ev = FilterNet(
            max_l=max_l,
            features_dim=features_dim,
            r_max=r_max,
            num_radial_basis=num_radial_basis,
            radial_basis_fn=radial_basis_fn,
            trainable_rbf=trainable_rbf,
        )
        
        self.euclidean_attention_block = EuclideanAttentionBlock(
            max_l=max_l,
            r_max=r_max,
            filter_net_inv=self.filter_net_inv,
            filter_net_ev=self.filter_net_ev
        )
        self.interaction_block = InteractionBlock(
            max_l=max_l,
            features_dim=features_dim,
            bias=interaction_bias,
            device=device,
        )

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        sh_vectors: torch.Tensor,
        lengths: torch.Tensor,
        cutoffs: torch.Tensor,
    ) -> torch.Tensor:

        d_att_inv_features, d_att_ev_features = self.euclidean_attention_block(
            inv_features,
            ev_features,
            senders=senders,
            receivers=receivers,
            sh_vectors=sh_vectors,
            lengths=lengths,
            cutoffs=cutoffs,
        )
        att_inv_features = inv_features + d_att_inv_features
        att_ev_features = ev_features + d_att_ev_features

        d_inv_features, d_ev_features = self.interaction_block(
            att_inv_features, att_ev_features
        )

        new_inv_features = att_inv_features + d_inv_features
        new_ev_features = att_ev_features + d_ev_features

        return new_inv_features, new_ev_features


@compile_mode("script")
class EuclideanAttentionBlock(torch.nn.Module):
    # create dummy class
    def __init__(
        self,
        r_max: float,
        max_l: int,
        filter_net_inv: callable,
        filter_net_ev: callable
        ):
        super().__init__()

        
        self.so3_conv_invariants = SO3ConvolutionInvariants(max_l=max_l)
        self.filter_net_inv = filter_net_inv
        self.filter_net_ev = filter_net_ev
        
    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        sh_vectors: torch.Tensor,
        lengths: torch.Tensor,
        cutoffs: torch.Tensor,
    ) -> torch.Tensor:

        ev_differences = ev_features[senders] - ev_features[receivers]
        ev_differences_invariants = self.so3_conv_invariants(
            ev_differences, ev_differences
        )
        filter_w_inv = self.filter_net_inv(
            lengths,
            ev_differences_invariants
        )
        filter_w_ev = self.filter_net_ev(
            lengths,
            ev_differences_invariants
        )
        
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
            bias=bias,
        )
        self.so3_conv_invariants = SO3ConvolutionInvariants(max_l=max_l)
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

        ev_invariants = self.so3_conv_invariants(ev_features, ev_features)
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
            b_ev_features, self.repeats, dim=-1
        )
        d_ev_features = b_ev_features * ev_features
        return d_inv_features, d_ev_features

@compile_mode("script")
class FilterNet(torch.nn.Module):
    # create dummy class
    def __init__(
        self,
        r_max: float,
        max_l: int,
        num_radial_basis: int,
        features_dim: int,
        radial_basis_fn: str = "gaussian",
        trainable_rbf: bool = False,
        num_layers: int = 2,
        non_linearity: Type[torch.nn.Module] = torch.nn.SiLU
        ):
        super().__init__()
        
        radial_basis_fn = radial_basis_fn.lower()
        assert radial_basis_fn in [
            "gaussian",
            "bernstein",
            "bessel",
        ], f"Radial basis '{radial_basis_fn}' is not supported. Choose from 'gaussian', 'bernstein', or 'bessel'."
        
        if radial_basis_fn == "gaussian":
            self.radial_basis_fn = radial_basis.GaussianBasis(
                r_max=r_max,
                num_radial_basis=num_radial_basis,
                trainable=trainable_rbf,
            )
        elif radial_basis_fn == "bernstein":
            self.radial_basis = radial_basis.BernsteinBasis(
                n_rbf=num_radial_basis,
                r_cut=r_max,
                trainable_gamma=trainable_rbf,
            )
        elif radial_basis_fn == "bessel":
            self.radial_basis_fn = radial_basis.BesselBasis(
                n_rbf=num_radial_basis,
                r_cut=r_max,
                trainable_freqs=trainable_rbf,
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
                in_features=max_l + 1,
                out_features=features_dim // 4,
            ),
            non_linearity(),
        )
        for i in range(num_layers - 1):
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
        distances: torch.Tensor,
        ev_difference_invariants: torch.Tensor,
    ) -> torch.Tensor:

        rbf = self.radial_basis_fn(distances)
        rbf_features = self.mlp_rbf(rbf)
        ev_features = self.mlp_ev(ev_difference_invariants)
        
        filter_weights = rbf_features + ev_features
        return filter_weights

        
        