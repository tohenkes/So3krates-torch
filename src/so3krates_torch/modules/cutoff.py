import torch
from e3nn.util.jit import compile_mode

@compile_mode("script")
class CosineCutoff(torch.nn.Module):
    """
    Equation 17 from so3krates paper 
    https://doi.org/10.1038/s41467-024-50620-6
    """

    r_max: torch.Tensor

    def __init__(
        self,
        r_max: float
        ):
        super().__init__()
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return (
            0.5 * (torch.cos(torch.pi * x / self.r_max) + 1.0)
            if torch.is_tensor(x)
            else 0.5 * (torch.cos(torch.pi * x / self.r_max) + 1.0).item()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max})"
