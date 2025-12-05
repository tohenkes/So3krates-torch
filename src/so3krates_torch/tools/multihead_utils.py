import torch
from so3krates_torch.modules.models import So3krates, SO3LR


def reduce_mh_model_to_sh(
    mh_model_state_dict,
    settings: dict,
    head_idx: int,
    model_choice: str = "so3lr",
    device: str = "cpu",
    dtype: str = "float32"
):
    num_layers = settings.get("final_mlp_layers", 2)
    exclude = [
        "convert_to_multihead",
        "use_multihead",
        "num_output_heads",
        "r_max_lr",
    ]
    settings = {
        k: v for k, v in settings.items() if k not in exclude
    }
    settings["device"] = device
    settings["dtype"] = dtype
    
    if model_choice == "so3lr":
        sh_model = SO3LR(**settings)
    elif model_choice == "so3krates":
        sh_model = So3krates(**settings)
    
    
    sh_model.load_state_dict(mh_model_state_dict, strict=False)
    
    # load energy scales
    mh_energy_scales = mh_model_state_dict["atomic_energy_output_block.energy_scales"].T
    sh_model.atomic_energy_output_block.energy_scales.weight.data = (
        mh_energy_scales.to(dtype=getattr(torch, dtype), device=device)
    )
    
    for layer in range(num_layers-1):
        mh_weight = mh_model_state_dict[
            f"atomic_energy_output_block.layers_weights.{layer}"
            ]
        mh_bias = mh_model_state_dict[
            f"atomic_energy_output_block.layers_bias.{layer}"
            ]
        mh_weight = mh_weight[head_idx].T
        mh_bias = mh_bias[head_idx].T
        # Use .data to replace in-place
        sh_model.atomic_energy_output_block.layers[layer].weight.data = (
                mh_weight.to(dtype=getattr(torch, dtype), device=device)
            )
        sh_model.atomic_energy_output_block.layers[layer].bias.data = (
                mh_bias.to(dtype=getattr(torch, dtype), device=device)
            )
    # Final layer
    mh_weight = mh_model_state_dict[
        f"atomic_energy_output_block.final_layer_weights"
     ]
    mh_bias = mh_model_state_dict[
        f"atomic_energy_output_block.final_layer_bias"
     ]
    mh_weight = mh_weight[head_idx].T
    mh_bias = mh_bias[head_idx].T

    sh_model.atomic_energy_output_block.final_layer.weight.data = (
            mh_weight.to(dtype=getattr(torch, dtype), device=device)
        )
    sh_model.atomic_energy_output_block.final_layer.bias.data = (
            mh_bias.to(dtype=getattr(torch, dtype), device=device)
        )
    return sh_model