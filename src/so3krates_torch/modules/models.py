import torch
from typing import Any, Callable, Dict, List, Optional, Type, Union
from e3nn.util.jit import compile_mode
from mace.modules.utils import prepare_graph
from so3krates_torch.modules.cutoff import CosineCutoff
from so3krates_torch.modules.spherical_harmonics import RealSphericalHarmonics
from so3krates_torch.blocks import (
    embedding,
    euclidean_transformer,
    output_block,
)
from so3krates_torch.tools import scatter
from mace.modules.utils import get_outputs
from so3krates_torch.blocks import radial_basis

@compile_mode("script")
class So3krates(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_radial_basis: int,
        degrees: List[int],
        features_dim: int,
        num_att_heads: int,
        final_mlp_layers: int,
        atomic_numbers: List[int],
        num_interactions: int,
        num_elements: int,
        avg_num_neighbors: int,
        message_normalization: str = "sqrt_num_features",
        initialize_ev_to_zeros: bool = True,
        radial_basis_fn: str = "gaussian",
        trainable_rbf: bool = False,
        interaction_bias: bool = True,
        cutoff_function: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = CosineCutoff,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions",
            torch.tensor(num_interactions, dtype=torch.int64),
        )

        torch.manual_seed(seed)
        self.cutoff_function = cutoff_function(r_max)
        
        self.spherical_harmonics = RealSphericalHarmonics(
            degrees=degrees,
        )
        
        self.inv_feature_embedding = embedding.InvariantEmbedding(
            in_features=num_elements,
            out_features=features_dim,
            bias=False,
        )
        self.ev_embedding = embedding.EuclideanEmbedding(
            initialization_to_zeros=initialize_ev_to_zeros,
        )
        self.avg_num_neighbors = avg_num_neighbors
        self.inv_avg_num_neighbors = 1.0 / avg_num_neighbors

        self.radial_embedding = radial_basis.ComputeRBF(
            r_max=r_max,
            num_radial_basis=num_radial_basis,
            radial_basis_fn=radial_basis_fn,
            trainable=trainable_rbf,
        )

        self.euclidean_transformers = torch.nn.ModuleList(
            [
                euclidean_transformer.EuclideanTransformer(
                    degrees=degrees,
                    num_heads=num_att_heads,
                    features_dim=features_dim,
                    num_radial_basis=num_radial_basis,
                    interaction_bias=interaction_bias,
                    message_normalization=message_normalization,
                    avg_num_neighbors=avg_num_neighbors,
                    device=device,
                )
                for _ in range(num_interactions)
            ]
        )

        self.output_block = output_block.InvariantOutputHead(
            features_dim=features_dim,
            final_output_features=1,  # TODO: remove hardcoded value
            layers=final_mlp_layers,
            bias=True,
            non_linearity=torch.nn.functional.silu,
            final_non_linearity=False,
            use_non_linearity=True,
        )

    def get_representation(
        self,
        data: Dict[str, torch.Tensor],
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ):
        ######### PROCESSING DATA #########
        ctx = prepare_graph(
            data=data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        self.is_lammps = ctx.is_lammps
        self.num_atoms_arange = ctx.num_atoms_arange
        self.num_graphs = ctx.num_graphs
        self.displacement = ctx.displacement
        self.positions = ctx.positions
        # for some reason the vectors in so3krates are pointing in the
        # opposite direction compared to how its usually done. But this
        # does not matter for the model, as long as we are consistent
        self.vectors = -1. * ctx.vectors
        self.lengths = ctx.lengths
        self.cell = ctx.cell
        self.node_heads = ctx.node_heads
        self.interaction_kwargs = ctx.interaction_kwargs
        self.lammps_natoms = self.interaction_kwargs.lammps_natoms
        self.lammps_class = self.interaction_kwargs.lammps_class

        senders, receivers = data["edge_index"][0], data["edge_index"][1]

        # normalize the vectors to unit length
        self.vectors_unit = self.vectors / self.vectors.norm(dim=-1, keepdim=True)
        sh_vectors = self.spherical_harmonics(self.vectors_unit)
        cutoffs = self.cutoff_function(self.lengths)
        
        ######### EMBEDDING #########
        inv_features = self.inv_feature_embedding(data["node_attrs"])
        ev_features = self.ev_embedding(
            sh_vectors=sh_vectors,
            cutoffs=cutoffs,
            receivers=receivers,
            inv_avg_num_neighbors=self.inv_avg_num_neighbors,
        )
        rbf = self.radial_embedding(self.lengths)
        
        ######### TRANSFORMER #########
        for transformer in self.euclidean_transformers:
            inv_features, ev_features = transformer(
                inv_features=inv_features,
                ev_features=ev_features,
                rbf=rbf,
                senders=senders,
                receivers=receivers,
                sh_vectors=sh_vectors,
                cutoffs=cutoffs,
            )
        return inv_features, ev_features
    
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:


        inv_features, ev_features = self.get_representation(
            data=data,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            lammps_mliap=lammps_mliap,
        )
        
        ######### OUTPUT #########
        node_energies = self.output_block(inv_features)
        total_energy = scatter.scatter_sum(
            src=node_energies, index=data["batch"], dim=0, dim_size=self.num_graphs
        )
        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=self.positions,
            displacement=self.displacement,
            vectors=self.vectors,
            cell=self.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        return {
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "edge_forces": edge_forces,
        }
