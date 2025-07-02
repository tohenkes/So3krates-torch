import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional


@compile_mode("script")
class InvariantOutputHead(torch.nn.Module):
    def __init__(
        self,
        features_dim: int,
        final_output_features: int = 1,
        layers: int = 2,
        bias: bool = True,
        use_non_linearity: bool = True,
        non_linearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        final_non_linearity: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(layers - 1):
            self.layers.append(
                torch.nn.Linear(features_dim, features_dim, bias=bias)
            )
        self.final_layer = torch.nn.Linear(
            features_dim, final_output_features, bias=bias
        )
        # make sure non_linearity is not None if use_non_linearity is True
        if use_non_linearity and non_linearity is None:
            raise ValueError(
                "If use_non_linearity is True, non_linearity must be provided."
            )
        self.use_non_linearity = use_non_linearity
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if self.use_non_linearity:
                x = self.non_linearity(x)
        x = self.final_layer(x)
        if self.final_non_linearity:
            x = self.non_linearity(x)
        return x
