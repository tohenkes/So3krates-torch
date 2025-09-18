from contextlib import contextmanager
from so3krates_torch.blocks.euclidean_transformer import (
    EuclideanAttentionBlockLORA,
    EuclideanAttentionBlock,
)

@contextmanager
def preserve_grad_state(model):
    """
    Taken from
    https://github.com/ACEsuit/mace/pull/830/
    https://doi.org/10.1038/s41524-025-01727-x
    """
    # save the original requires_grad state for all parameters
    requires_grad_backup = {
        param: param.requires_grad for param in model.parameters()
    }
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad

def model_to_lora(
        model,
        rank: int = 4,
        alpha: int = 8,
        device: str = "cpu",
):
    """
    Convert a model to LoRA (Low-Rank Adaptation) format.
    This function modifies the model in place, replacing the normal euclidean attention blocks
    with their LoRA counterparts.
    """
    for i, transformer in enumerate(model.euclidean_transformers):

        if isinstance(transformer.euclidean_attention_block, EuclideanAttentionBlock):
            degrees = transformer.euclidean_attention_block.degrees
            num_heads = transformer.euclidean_attention_block.num_heads
            features_dim = transformer.euclidean_attention_block.features_dim
            filter_net_inv = transformer.euclidean_attention_block.filter_net_inv
            filter_net_ev = transformer.euclidean_attention_block.filter_net_ev
            message_normalization = transformer.euclidean_attention_block.message_normalization
            qk_non_linearity = transformer.euclidean_attention_block.qk_non_linearity
            avg_num_neighbors = transformer.euclidean_attention_block.avg_num_neighbors


            # Create a new LoRA attention block with the same parameters
            lora_attention_block = EuclideanAttentionBlockLORA(
                degrees=degrees,
                num_heads=num_heads,
                features_dim=features_dim,
                filter_net_inv=filter_net_inv,
                filter_net_ev=filter_net_ev,
                lora_rank=rank,
                lora_alpha=alpha,
                message_normalization=message_normalization,
                qk_non_linearity=qk_non_linearity,
                avg_num_neighbors=avg_num_neighbors,
                device=device
            )

            # Copy weights from the original attention block to the LoRA block
            lora_attention_block.load_state_dict(
                transformer.euclidean_attention_block.state_dict(),
                strict=False,  # Allow missing keys for LoRA-specific parameters
            )

            # Replace the original attention block with the LoRA version
            transformer.euclidean_attention_block = lora_attention_block

    return model

def fuse_lora_weights(model):
    """
    Fuse the LoRA weights into the main model weights.
    This function modifies the model in place, combining the LoRA weights with the original weights.
    """
    for transformer in model.euclidean_transformers:
        if isinstance(transformer.euclidean_attention_block, EuclideanAttentionBlockLORA):
            transformer.euclidean_attention_block.fuse_lora_weights()
    return model


def freeze_model_parameters(
    model,
    keep_trainable_choice: str,
    freeze_shifts: bool = False,
    freeze_scales: bool = False,
    freeze_partial_charges: bool = True,
    freeze_zbl: bool = True,
    freeze_hirshfeld: bool = True,
    freeze_embedding: bool = True,
):
    possible_choices = ["last_layer", "mlp", "qkv", "lora"]
    if keep_trainable_choice not in possible_choices:
        raise ValueError(
            f"Invalid choice '{keep_trainable_choice}'. Must be one of {possible_choices}."
        )

    keep_trainable = []

    # keep output heads trainable (energy, forces, etc.) if not lora
    if keep_trainable_choice != "lora":
        keep_trainable.append("atomic_energy_output_block.layers")
        keep_trainable.append("atomic_energy_output_block.final_layer")

    # Determine which transformer layers to keep trainable based on choice
    num_layers = len(model.euclidean_transformers)

    if keep_trainable_choice == "last_layer":
        # Keep only the last transformer layer trainable
        keep_trainable.append(f"euclidean_transformers.{num_layers-1}")

    elif keep_trainable_choice == "mlp":
        # Keep only the MLP parts of all layers trainable (not attention)
        for i in range(num_layers):
            keep_trainable.append(f"euclidean_transformers.{i}.euclidean_mlp")

    elif keep_trainable_choice == "qkv":
        # Keep only the QKV projection layers trainable
        for i in range(num_layers):
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block"
            )
    elif keep_trainable_choice == "lora":
        # Keep only the LoRA parameters trainable
        for i in range(num_layers):
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.lora_"
            )

    # Handle optional components based on freeze flags
    if not freeze_shifts:
        keep_trainable.append("atomic_energy_output_block.energy_shifts")

    if not freeze_scales:
        keep_trainable.append("atomic_energy_output_block.energy_scales")

    if (
        not freeze_partial_charges
        and hasattr(model, "electrostatic_energy_bool")
        and model.electrostatic_energy_bool
    ):
        keep_trainable.append("partial_charges_output_block")

    if (
        not freeze_zbl
        and hasattr(model, "zbl_repulsion_bool")
        and model.zbl_repulsion_bool
    ):
        keep_trainable.append("zbl_repulsion")

    if (
        not freeze_hirshfeld
        and hasattr(model, "dispersion_energy_bool")
        and model.dispersion_energy_bool
    ):
        keep_trainable.append("hirshfeld_output_block")

    if not freeze_embedding:
        keep_trainable.append("embedding")

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze only the specified ones
    for name, param in model.named_parameters():
        for keep_key in keep_trainable:
            if keep_key in name:
                param.requires_grad = True
                break  # Found match, no need to check other keys
