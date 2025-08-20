###########################################################################################
#
# Taken from MACE package: https://github.com/ACEsuit/mace
#
# Added long-range behavior
#
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, NamedTuple, Optional, Tuple
import torch


def get_symmetric_displacement(
    positions: torch.Tensor,
    cell: Optional[torch.Tensor],
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)

    return positions, cell, displacement


def compute_shifts(
    unit_shifts: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    sender: torch.Tensor,
) -> torch.Tensor:
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return shifts


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:

    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


class InteractionKwargs(NamedTuple):
    lammps_class: Optional[torch.Tensor]
    lammps_natoms: Tuple[int, int] = (0, 0)


class GraphContext(NamedTuple):
    is_lammps: bool
    num_graphs: int
    num_atoms_arange: torch.Tensor
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    vectors: torch.Tensor
    vectors_lr: torch.Tensor
    lengths: torch.Tensor
    lengths_lr: torch.Tensor
    cell: torch.Tensor
    node_heads: torch.Tensor
    interaction_kwargs: InteractionKwargs


def prepare_graph(
    data: Dict[str, torch.Tensor],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    lammps_mliap: bool = False,
    lr: bool = False,
) -> GraphContext:
    if torch.jit.is_scripting():
        lammps_mliap = False

    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    )

    if lammps_mliap:
        raise NotImplementedError("LAMMPS MLIAP support is not implemented")
        n_real, n_total = data["natoms"][0], data["natoms"][1]
        num_graphs = 2
        num_atoms_arange = torch.arange(
            n_real, device=data["node_attrs"].device
        )
        displacement = None
        positions = torch.zeros(
            (int(n_real), 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        cell = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        vectors = data["vectors"].requires_grad_(True)
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        ikw = InteractionKwargs(data["lammps_class"], (n_real, n_total))
    else:
        data["positions"].requires_grad_(True)
        positions = data["positions"]
        cell = data["cell"]
        num_atoms_arange = torch.arange(
            positions.shape[0], device=positions.device
        )
        num_graphs = int(data["ptr"].numel() - 1)
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )

        if compute_virials or compute_stress or compute_displacement:
            p, cell, displacement = get_symmetric_displacement(
                positions=positions,
                cell=cell,
                num_graphs=num_graphs,
                batch=data["batch"],
            )
            data["positions"] = p
            data["shifts"] = compute_shifts(
                unit_shifts=data["unit_shifts"],
                cell=cell,
                batch=data["batch"],
                sender=data["edge_index"][0],
            )
            if lr:
                data["shifts_lr"] = compute_shifts(
                    unit_shifts=data["unit_shifts_lr"],
                    cell=cell,
                    batch=data["batch"],
                    sender=data["edge_index_lr"][0],
                )

        if lr:
            vectors_lr, lengths_lr = get_edge_vectors_and_lengths(
                positions=data["positions"],
                edge_index=data["edge_index_lr"],
                shifts=data["shifts_lr"],
            )
        else:
            vectors_lr, lengths_lr = None, None

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        ikw = InteractionKwargs(None, (0, 0))

    return GraphContext(
        is_lammps=lammps_mliap,
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        vectors_lr=vectors_lr,
        lengths_lr=lengths_lr,
        cell=cell,
        node_heads=node_heads,
        interaction_kwargs=ikw,
    )
