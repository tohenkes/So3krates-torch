import torch
from typing import Any, Callable, Dict, List, Optional, Type, Union
from e3nn import o3
from e3nn.util.jit import compile_mode
from mace.modules.utils import prepare_graph
from so3krates_torch.modules.cutoff import CosineCutoff
from so3krates_torch.blocks import embedding, euclidean_transformer, output_block
from so3krates_torch.tools import scatter
from mace.modules.utils import (
    get_outputs
)

@compile_mode("script")
class So3krates(torch.nn.Module):
    def __init__(   
        self,
        r_max,
        num_radial_basis,
        max_l,
        features_dim,
        ev_l,
        num_att_heads,
        final_mlp_layers: int,
        atomic_numbers: List[int],
        num_interactions: int,
        num_elements: int,
        avg_num_neighbors: int,
        use_so3: bool = False,
        cutoff_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = CosineCutoff,
        seed: Optional[int] = None,
        
        
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        
        
        torch.manual_seed(seed)
        self.cutoff_function = cutoff_function(r_max)
        self.invariant_feature_embedding = embedding.InvariantEmbedding(
            in_features=num_elements,
            out_features=features_dim,
            bias=False,
        )
        self.euclidean_embedding = embedding.EuclideanEmbedding(
            avg_num_neighbors=avg_num_neighbors,
            max_l=max_l,
            cutoff_function=self.cutoff_function,
            use_so3=use_so3,
            normalize=True,
            normalization="component",
        )
        self.euclidean_transformers = torch.nn.ModuleList(
            [
                euclidean_transformer.EuclideanTransformer(
                )
                for _ in range(num_interactions)
            ]
        )
        
        self.output_block = output_block.InvariantOutputHead(
            features_dim=features_dim,
            final_output_features=1,
            layers=final_mlp_layers,
            bias=True,
            non_linearity=torch.nn.functional.silu,
        )
            
        
        

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
        
        ctx = prepare_graph(
            data=data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        senders, receivers = data['edge_index'][0], data['edge_index'][1]

        invariant_features = self.invariant_feature_embedding(
            x=data
        )
        euclidean_features = self.euclidean_embedding(
            senders=senders,
            receivers=receivers,
            lengths=lengths,
            vectors=vectors
        )
        
        for transformer in self.euclidean_transformers:
            invariant_features, euclidean_features = transformer(
                invariant_features=invariant_features,
                euclidean_features=euclidean_features,
            )
        
        node_energies = self.output_block(
            invariant_features
        )
        total_energy = scatter.scatter_sum(
            src=node_energies,
            index=data['batch'],
            dim=0,
            dim_size=num_graphs
        )
        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
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
