import torch
from torch.nn import Parameter
from torch.nn import Module
import math


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
            self.register_buffer("centers", centers)  # , persistent=False)
            self.register_buffer("widths", widths)  # , persistent=False)

    def _init_centers(self, min: float, max: float, n: int) -> torch.Tensor:
        return torch.linspace(min, max, n)

    def _init_widths(self, centers: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(centers[1] - centers[0])
        return torch.full_like(centers, delta)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)  # shape [..., 1]
        return torch.exp(
            -0.5 / (self.widths**2) * (d - self.centers) ** 2
        ).squeeze()


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
        exp_r = torch.exp(-self.gamma * distances)
        exp_r = torch.clamp(exp_r, min=self.eps, max=1.0 - self.eps)
        x = exp_r
        k_log_x = self.k * torch.log(x)
        k_rev_log_1_minus_x = self.k_rev * torch.log(1 - x)

        log_poly = self.b + k_log_x + k_rev_log_1_minus_x
        return torch.exp(log_poly)


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

        # sin(freq * r) / r
        sin_term = torch.sin(self.freqs * r_clamped)  # (..., n_rbf)
        output = sin_term / r_clamped  # (..., n_rbf)

        return output


class ComputeRBF(Module):
    def __init__(
        self,
        r_max: float,
        num_radial_basis: int,
        trainable: bool = True,
        radial_basis_fn: str = "gaussian",
    ):
        super().__init__()

        radial_basis_fn = radial_basis_fn.lower()
        assert radial_basis_fn in [
            "gaussian",
            "bernstein",
            "bessel",
        ], f"Radial basis '{radial_basis_fn}' is not supported. Choose from 'gaussian', 'bernstein', or 'bessel'."

        if radial_basis_fn == "gaussian":
            self.radial_basis_fn = GaussianBasis(
                r_max=r_max,
                num_radial_basis=num_radial_basis,
                trainable=trainable,
            )
        elif radial_basis_fn == "bernstein":
            self.radial_basis_fn = BernsteinBasis(
                n_rbf=num_radial_basis,
                r_cut=r_max,
                trainable_gamma=trainable,
            )
        elif radial_basis_fn == "bessel":
            self.radial_basis_fn = BesselBasis(
                n_rbf=num_radial_basis,
                r_cut=r_max,
                trainable_freqs=trainable,
            )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.radial_basis_fn(distances)
