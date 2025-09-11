from contextlib import contextmanager


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
    possible_choices = ["last_layer", "mlp", "qkv"]
    if keep_trainable_choice not in possible_choices:
        raise ValueError(
            f"Invalid choice '{keep_trainable_choice}'. Must be one of {possible_choices}."
        )

    keep_trainable = []

    # Always keep output heads trainable (energy, forces, etc.)
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
