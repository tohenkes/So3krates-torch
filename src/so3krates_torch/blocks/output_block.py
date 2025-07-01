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
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if self.non_linearity is not None:
                x = self.non_linearity(x)
        x = self.final_layer(x)
        if self.final_non_linearity:
            x = self.non_linearity(x)
        return x
