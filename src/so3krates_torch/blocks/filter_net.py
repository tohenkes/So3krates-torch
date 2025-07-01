import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional
from so3krates_torch.blocks import so3_conv_invariants

@compile_mode("script")
class FilterNet(torch.nn.Module):
    # create dummy class
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x