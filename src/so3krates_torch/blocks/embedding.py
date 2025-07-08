import torch
from e3nn.util.jit import compile_mode
from e3nn import o3
from typing import Callable, List, Dict, Optional
from so3krates_torch.tools import scatter
import math

@compile_mode("script")
class InvariantEmbedding(torch.nn.Module):
    '''
    Eq. 10 in https://doi.org/10.1038/s41467-024-50620-6
    '''
    def __init__(
        self, num_elements: int, out_features: int, bias: bool = False
    ):
        super().__init__()
        self.embedding = torch.nn.Linear(
            num_elements,
            out_features, 
            bias=bias,
            dtype=torch.get_default_dtype()
        )

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        return self.embedding(one_hot)

    def reset_parameters(self):
        self.embedding.reset_parameters()


@compile_mode("script")
class EuclideanEmbedding(torch.nn.Module):
    '''
    Eq. 11 in https://doi.org/10.1038/s41467-024-50620-6
    
    In the paper it is said that the initialization is done trough
    Eq. 11, as stated above. However, in the code it is hardcoded to
    be zeros.
    '''
    def __init__(
        self,
        initialization_to_zeros: bool = True,
    ):
        super().__init__()
        self.initialization_to_zeros = initialization_to_zeros
        
    def forward(
        self,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        receivers: torch.Tensor,
        inv_avg_num_neighbors: float,
    ) -> torch.Tensor:

        if self.initialization_to_zeros:
            num_nodes = torch.unique(receivers).numel()
            ev_embedding = torch.zeros(
                (num_nodes, sh_vectors.shape[1]),
                dtype=sh_vectors.dtype,
                device=sh_vectors.device
            )
        else:
            # This is actually Eq. 11 in the paper:
            scaled_neighbors = sh_vectors * cutoffs
            ev_embedding = (
                scatter.scatter_sum(
                    src=scaled_neighbors,
                    index=receivers,
                    dim=0,
                )
                * inv_avg_num_neighbors
            )

        return ev_embedding

@compile_mode("script")
class ChargeSpinEmbedding(torch.nn.Module):
    """
    As introduced in doi.org/10.1038/s41467-021-27504-0.
    """

    def __init__(
            self,
            num_features: int,
            activation_fn: Callable = torch.nn.SiLU,
            num_elements: int = 118,
            ):
        super().__init__()

        self.Wq = torch.nn.Linear(
            in_features=num_elements,
            out_features=num_features,
            bias=False,
        )
        self.Wk = torch.nn.Parameter(
            torch.empty(
                size=(2,num_features)
            )
        )
        self.Wv = torch.nn.Parameter(
            torch.empty(
                size=(2,num_features)
            )
        )
        sqrt_5 = math.sqrt(5)
        torch.nn.init.kaiming_uniform_(self.Wk, a=sqrt_5)
        torch.nn.init.kaiming_uniform_(self.Wv, a=sqrt_5)

        self.sqrt_dim = num_features ** 0.5
        self.activation_fn = activation_fn
        self.mlp = torch.nn.Sequential(
            self.activation_fn(),
            torch.nn.Linear(num_features, num_features, bias=False),
            self.activation_fn(),
            torch.nn.Linear(num_features, num_features, bias=False),
        )
        self.run_residual_mlp = lambda x: x + self.mlp(x)

        self.choose_idx = lambda x: int(x < 0)

    def forward(
            self,
            elements_one_hot: torch.Tensor,
            psi: torch.Tensor,
            batch_segments: torch.Tensor,
            ) -> torch.Tensor:
        """
        Computes the charge-spin embedding as described in
        doi.org/10.1038/s41467-021-27504-0.

        Args:
            elements_one_hot (torch.Tensor): One-hot encoded atomic numbers.
            psi (torch.Tensor): Charge or number of unpaired electrons.
            batch_segments (torch.Tensor): Indices of the batches inside the tensor.

        Returns:
            torch.Tensor: The charge-spin embedding for the input.
        """
        torch.set_printoptions(precision=8, sci_mode=False)
        q = self.Wq(elements_one_hot)
        idx = self.choose_idx(psi.item())
        k = self.Wk[idx].unsqueeze(0)
        v = self.Wv[idx]
        q_x_k = (q * k).sum(dim=-1) / self.sqrt_dim
        y = torch.nn.functional.softplus(q_x_k)
        denominator = scatter.scatter_sum(
            src=y,
            index=batch_segments,
            dim=0,
        )
        
        a = (psi * y / denominator)
        e_psi = self.run_residual_mlp(
            a[:,None] * v[None,:]
        )
        return e_psi



