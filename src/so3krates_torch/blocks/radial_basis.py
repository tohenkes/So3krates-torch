import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Any, Callable, Dict, List, Optional, Type, Union
from torch.nn import Parameter
from torch.nn import Module
import math

@compile_mode("script")
class GaussianBasis(Module):
    def __init__(
        self,
        r_max: float,
        num_radial_basis: int,
        r_0: float = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.r_max = r_max
        self.num_radial_basis = num_radial_basis

        centers = self._init_centers(r_0, r_max, num_radial_basis)
        widths = self._init_widths(centers)

        if trainable:
            self.centers = Parameter(centers)
            self.widths = Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def _init_centers(
        self,
        min: float,
        max: float,
        n: int
    ) -> torch.Tensor:
        return torch.linspace(min, max, n)

    def _init_widths(
        self,
        centers: torch.Tensor
    ) -> torch.Tensor:
        delta = torch.abs(centers[1] - centers[0])
        return torch.full_like(centers, delta)

    def forward(
        self,
        distances: torch.Tensor
    ) -> torch.Tensor:
        d = distances.unsqueeze(-1)  # shape [..., 1]
        return torch.exp(
            -0.5 / (self.widths ** 2) * (d - self.centers) ** 2
        ).squeeze()

@compile_mode("script")
class BernsteinBasis(Module):
    def __init__(
        self,
        n_rbf: int,
        r_cut: float = None,
        gamma_init: float = 0.9448630629184640,
        trainable_gamma: bool = False,
        eps: float = 1e-6,  # For clamping to avoid log(0) or log(1)
    ):
        super().__init__()
        self.n_rbf = n_rbf
        self.r_cut = r_cut
        self.eps = eps

        b = [log_binomial_coefficient(n_rbf - 1, k) for k in range(n_rbf)]
        self.register_buffer("b", torch.tensor(b))

        self.register_buffer("k", torch.arange(n_rbf))
        self.register_buffer("k_rev", torch.arange(n_rbf - 1, -1, -1))

        gamma_tensor = torch.tensor(gamma_init)
        if trainable_gamma:
            self.gamma = Parameter(gamma_tensor)
        else:
            self.register_buffer("gamma", gamma_tensor)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # Compute e^{-gamma * r}, clamp for numerical safety
        exp_r = torch.exp(-self.gamma * distances)
        exp_r = torch.clamp(exp_r, min=self.eps, max=1.0 - self.eps)  # avoid log(0), log(1)

        x = exp_r.unsqueeze(-1)  # shape (..., 1) -> (..., 1, 1)
        k_log_x = self.k * torch.log(x)            # (..., 1, K)
        k_rev_log_1_minus_x = self.k_rev * torch.log(1 - x)  # (..., 1, K)

        log_poly = self.b + k_log_x + k_rev_log_1_minus_x  # broadcasting (..., K)
        return torch.exp(log_poly)  # (..., K)


def log_binomial_coefficient(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

class BesselBasis(Module):
    def __init__(
        self,
        n_rbf: int,
        r_cut: float,
        r_0: float = 0.0,
        trainable_freqs: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_rbf = n_rbf
        self.r_cut = r_cut
        self.r_0 = r_0
        self.eps = eps

        # Initial frequencies: n * pi / r_cut
        initial_freqs = torch.arange(n_rbf) * math.pi / r_cut

        if trainable_freqs:
            self.freqs = Parameter(initial_freqs)  # trainable
        else:
            self.register_buffer("freqs", initial_freqs)  # fixed

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_clamped = torch.clamp(r, min=self.eps)  # prevent divide-by-zero
        r_expanded = r_clamped.unsqueeze(-1)  # (..., 1)

        # sin(freq * r) / r
        sin_term = torch.sin(self.freqs * r_expanded)  # (..., n_rbf)
        output = sin_term / r_expanded  # (..., n_rbf)

        return output
