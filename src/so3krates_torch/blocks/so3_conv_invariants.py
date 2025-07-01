import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional


@compile_mode("script")
class SO3ConvolutionInvariants(torch.nn.Module):
    def __init__(
        self,
        max_l: int,
    ):
        super().__init__()
        self.max_l = max_l

        irreps_list = []
        for l in range(max_l + 1):
            standard_parity = "e" if l % 2 == 0 else "o"
            irreps_list.append(o3.Irrep(f"{l}{standard_parity}"))
        self.irreps_in = o3.Irreps(irreps_list)
        self.tensor_product = o3.FullTensorProduct(
            self.irreps_in,
            self.irreps_in,
            filter_ir_out=["0e"],
            internal_weights=False,
        )

    def forward(
        self, ev_features_1: torch.Tensor, ev_features_2: torch.Tensor
    ) -> torch.Tensor:

        return self.tensor_product(ev_features_1, ev_features_2)
