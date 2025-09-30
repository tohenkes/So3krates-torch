import argparse
import logging
import yaml
import torch
from ase.io import read
import random
from typing import Tuple
from so3krates_torch.modules.models import SO3LR, MultiHeadSO3LR
from so3krates_torch.tools.utils import (
    create_dataloader_from_list,
    create_data_from_list,
    create_dataloader_from_data,
)
from so3krates_torch.modules.loss import (
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
)
from mace.modules.loss import WeightedEnergyForcesLoss
from mace.data.utils import KeySpecification
from mace.modules.utils import compute_avg_num_neighbors
from mace.tools.utils import MetricsLogger, setup_logger
from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from torch_ema import ExponentialMovingAverage
from so3krates_torch.tools.train import train
from so3krates_torch.tools.finetune import (
    freeze_model_parameters,
    model_to_lora,
    fuse_lora_weights,
)
import os


def setup_config_from_yaml(config_path: str) -> dict:
    """Load and parse configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> None:
    """Setup logging based on configuration."""
    log_level = getattr(logging, config["MISC"].get("log_level", "INFO"))
    setup_logger(
        level=log_level,
        tag=config["GENERAL"]["name_exp"],
        directory=config["GENERAL"]["log_dir"],
    )


def create_model(config: dict, device: torch.device) -> SO3LR:
    """Create and initialize the SO3LR model."""
    arch_config = config["ARCHITECTURE"]

    # Map YAML parameters to model parameters
    model_params = {
        # Base So3krates parameters
        "r_max": arch_config.get("cutoff", 4.5),  # cutoff -> r_max
        "num_radial_basis": arch_config.get("num_radial_basis_fn", 32),
        "degrees": arch_config["degrees"],
        "features_dim": arch_config.get("num_features", 128),
        "num_att_heads": arch_config.get("num_heads", 4),
        "num_interactions": arch_config.get("num_layers", 3),
        "num_elements": 118,  # Default for periodic table
        "energy_regression_dim": arch_config.get("energy_regression_dim", 128),
        "message_normalization": arch_config.get(
            "message_normalization", "avg_num_neighbors"
        ),
        "radial_basis_fn": arch_config.get("radial_basis_fn", "bernstein"),
        "learn_atomic_type_shifts": arch_config.get(
            "energy_learn_atomic_type_shifts", False
        ),
        "learn_atomic_type_scales": arch_config.get(
            "energy_learn_atomic_type_scales", False
        ),
        "layer_normalization_1": arch_config.get(
            "layer_normalization_1", False
        ),
        "layer_normalization_2": arch_config.get(
            "layer_normalization_2", False
        ),
        "residual_mlp_1": arch_config.get("residual_mlp_1", False),
        "residual_mlp_2": arch_config.get("residual_mlp_2", False),
        "use_charge_embed": arch_config.get("use_charge_embed", False),
        "use_spin_embed": arch_config.get("use_spin_embed", False),
        "qk_non_linearity": arch_config.get("qk_non_linearity", "identity"),
        "cutoff_fn": arch_config.get("cutoff_fn", "cosine"),
        "activation_fn": arch_config.get("activation_fn", "silu"),
        "energy_activation_fn": arch_config.get(
            "energy_activation_fn", "silu"
        ),
        "seed": config["GENERAL"].get("seed", 42),
        "device": device,
        "dtype": config["GENERAL"].get("default_dtype", "float32"),
        "layers_behave_like_identity_fn_at_init": arch_config.get(
            "layers_behave_like_identity_fn_at_init", False
        ),
        "output_is_zero_at_init": arch_config.get(
            "output_is_zero_at_init", False
        ),
        "input_convention": arch_config.get("input_convention", "positions"),
        # SO3LR specific parameters
        "zbl_repulsion_bool": arch_config.get("zbl_repulsion_bool", True),
        "electrostatic_energy_bool": arch_config.get(
            "electrostatic_energy_bool", True
        ),
        "electrostatic_energy_scale": arch_config.get(
            "electrostatic_energy_scale", 4.0
        ),
        "dispersion_energy_bool": arch_config.get(
            "dispersion_energy_bool", True
        ),
        "dispersion_energy_scale": arch_config.get(
            "dispersion_energy_scale", 1.2
        ),
        "dispersion_energy_cutoff_lr_damping": arch_config.get(
            "dispersion_energy_cutoff_lr_damping"
        ),
        "r_max_lr": config["ARCHITECTURE"].get("cutoff_lr", None),
        "neighborlist_format": arch_config.get(
            "neighborlist_format_lr", "sparse"
        ),
    }

    if arch_config.get("multihead", False):
        model_params["num_output_heads"] = arch_config.get(
            "num_output_heads", None
        )
        assert (
            model_params["num_output_heads"] is not None
        ), "num_output_heads must be specified when using multihead"
        logging.info(
            f"Creating Multi-Head SO3LR model with {model_params['num_output_heads']} heads"
        )
        model = MultiHeadSO3LR(**model_params)
    else:
        logging.info(f"Createing SO3LR model")
        model = SO3LR(**model_params)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"The model has {total_params} parameters")
    return model


def select_valid_subset(
    data: list,
    valid_ratio: float,
    num_train: int = None,
    num_valid: int = None,
) -> Tuple[list, list]:
    n_valid = int(len(data) * valid_ratio)
    n_train = len(data) - n_valid
    if num_train is not None:
        n_train = min(n_train, num_train)
    if num_valid is not None:
        n_valid = min(n_valid, num_valid)
    random.shuffle(data)

    return data[:n_train], data[n_train : n_train + n_valid]


def setup_data_loaders(config: dict, model: SO3LR) -> tuple:
    """Setup training and validation data loaders."""
    # Key specification for data loading
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy", "dipole": "REF_dipole"},
        arrays_keys={
            "hirshfeld_ratios": "REF_hirsh_ratios",
            "forces": "REF_forces",
        },
    )
    # Create data loaders
    batch_size = config["TRAINING"]["batch_size"]
    valid_batch_size = config["TRAINING"]["valid_batch_size"]
    r_max_lr = config["TRAINING"].get("neighbors_lr_cutoff", 100.0)

    heads = config["TRAINING"].get("heads", None)
    if heads is not None:
        train_data = []
        val_data = {}
        for head_name, head_config in heads.items():
            head_data = read(head_config["path_to_train_data"], index=":")
            head_valid_path = head_config.get("path_to_val_data", None)
            if head_valid_path:
                head_val_data = read(head_valid_path, index=":")
            else:
                valid_ratio = head_config.get("valid_ratio", 0.1)
                num_train = head_config.get("num_train", None)
                num_valid = head_config.get("num_valid", None)
                head_train_data, head_val_data = select_valid_subset(
                    head_data, valid_ratio, num_train, num_valid
                )
            logging.info(
                f"Head {head_name} - Training set size: {len(head_train_data)}"
            )
            logging.info(
                f"Head {head_name} - Validation set size: {len(head_val_data)}"
            )

            head_config_list_train = create_data_from_list(
                head_train_data,
                r_max=model.r_max,
                r_max_lr=r_max_lr,
                key_specification=keyspec,
                head=head_name,
            )
            train_data.extend(head_config_list_train)

            head_config_list_val = create_data_from_list(
                head_val_data,
                r_max=model.r_max,
                r_max_lr=r_max_lr,
                key_specification=keyspec,
                head=head_name,
            )
            val_data[head_name] = create_dataloader_from_data(
                head_config_list_val,
                batch_size=valid_batch_size,
                shuffle=False,
            )

        train_loader = create_dataloader_from_data(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        logging.info(f"Training set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")

        # Compute average number of neighbors
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)
        model.avg_num_neighbors = avg_num_neighbors
        logging.info(f"Average number of neighbors: {avg_num_neighbors:.2f}")
        return train_loader, val_data

    else:
        # Load data
        data_path = config["TRAINING"]["path_to_train_data"]
        logging.info(f"Loading data from {data_path}")
        data = read(data_path, index=":")

        # Split data
        val_data_path = config["TRAINING"].get("path_to_val_data")
        if val_data_path:
            val_data = read(val_data_path, index=":")
            train_data = data
            logging.info(
                f"Using separate validation data from {val_data_path}"
            )
        else:
            valid_ratio = config["TRAINING"].get("valid_ratio", 0.1)
            num_train = config["TRAINING"].get("num_train", None)
            num_valid = config["TRAINING"].get("num_valid", None)
            train_data, val_data = select_valid_subset(
                data, valid_ratio, num_train, num_valid
            )
            logging.info(f"Splitting data with validation ratio {valid_ratio}")

        train_loader = create_dataloader_from_list(
            train_data,
            batch_size=batch_size,
            r_max=model.r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            shuffle=True,
        )

        valid_loader = create_dataloader_from_list(
            val_data,
            batch_size=valid_batch_size,
            r_max=model.r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            shuffle=False,
        )
        logging.info(f"Training set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")

        # Compute average number of neighbors
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)
        model.avg_num_neighbors = avg_num_neighbors
        logging.info(f"Average number of neighbors: {avg_num_neighbors:.2f}")
        return train_loader, {"main": valid_loader}


def setup_loss_function(config: dict) -> torch.nn.Module:
    """Setup loss function based on configuration."""
    loss_config = config["TRAINING"]

    # Get loss weights
    energy_w = loss_config.get("energy_weight", 1.0)
    forces_w = loss_config.get("forces_weight", 1000.0)
    dipole_w = loss_config.get("dipole_weight", 0.0)
    hirshfeld_w = loss_config.get("hirshfeld_weight", 0.0)

    # Check if explicit loss type is specified
    loss_type = loss_config.get("loss_type", "auto")

    if loss_type == "auto":
        # Auto-determine loss function type based on weights
        has_dipole = dipole_w > 0
        has_hirshfeld = hirshfeld_w > 0

        if has_dipole and has_hirshfeld:
            loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
                hirshfeld_weight=hirshfeld_w,
            )
            logging.info(
                "Auto-selected " "WeightedEnergyForcesDipoleHirshfeldLoss"
            )
        elif has_dipole:
            loss_fn = WeightedEnergyForcesDipoleLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
            )
            logging.info("Auto-selected WeightedEnergyForcesDipoleLoss")
        elif has_hirshfeld:
            loss_fn = WeightedEnergyForcesHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                hirshfeld_weight=hirshfeld_w,
            )
            logging.info("Auto-selected WeightedEnergyForcesHirshfeldLoss")
        else:
            # Default to basic energy-forces loss from MACE
            loss_fn = WeightedEnergyForcesLoss(
                energy_weight=energy_w, forces_weight=forces_w
            )
            logging.info("Auto-selected WeightedEnergyForcesLoss")
    else:
        # Explicit loss type specification
        if loss_type == "energy_forces":
            loss_fn = WeightedEnergyForcesLoss(
                energy_weight=energy_w, forces_weight=forces_w
            )
        elif loss_type == "energy_forces_dipole":
            loss_fn = WeightedEnergyForcesDipoleLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
            )
        elif loss_type == "energy_forces_hirshfeld":
            loss_fn = WeightedEnergyForcesHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                hirshfeld_weight=hirshfeld_w,
            )
        elif loss_type == "energy_forces_dipole_hirshfeld":
            loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
                hirshfeld_weight=hirshfeld_w,
            )
        else:
            supported_types = [
                "auto",
                "energy_forces",
                "energy_forces_dipole",
                "energy_forces_hirshfeld",
                "energy_forces_dipole_hirshfeld",
            ]
            raise ValueError(
                f"Unknown loss_type: {loss_type}. "
                f"Supported types: {supported_types}"
            )

        logging.info(f"Explicit loss type: {loss_type}")

    logging.info(
        f"Loss weights: energy={energy_w}, forces={forces_w}, "
        f"dipole={dipole_w}, hirshfeld={hirshfeld_w}"
    )

    return loss_fn


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict,
    use_lora_plus: bool = False,
    lora_B_lr: float = None,
) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    train_config = config["TRAINING"]

    # Setup optimizer
    optimizer_name = train_config.get("optimizer", "adam").lower()
    lr = train_config["lr"]
    weight_decay = train_config.get("weight_decay", 0.0)
    amsgrad = train_config.get("amsgrad", False)

    if optimizer_name == "adam":
        if use_lora_plus:
            assert (
                lora_B_lr is not None
            ), "lora_A_lr must be provided for LoRA+"
            lr = lora_B_lr
            # for LoRA+ adjust learning rate for A and B matrices
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "lora_A" in n
                        ],
                        "lr": lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "lora_B" in n
                        ],
                        "lr": lora_B_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "lora_A" not in n and "lora_B" not in n
                        ]
                    },
                ],
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Setup learning rate scheduler
    scheduler_name = train_config.get("scheduler", "exponential_decay")

    if scheduler_name == "exponential_decay":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=train_config.get("lr_scheduler_gamma", 0.9993)
        )
    elif scheduler_name == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=train_config.get("scheduler_patience", 5),
            factor=train_config.get("lr_factor", 0.85),
            min_lr=1e-6,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logging.info(f"Optimizer: {optimizer_name}, Learning rate: {lr}")
    logging.info(f"Scheduler: {scheduler_name}")

    return optimizer, lr_scheduler


def load_pretrained_model_direct(
    pretrained_path: str, device: torch.device
) -> torch.nn.Module:
    """Load a complete pretrained model directly."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_path}"
        )

    logging.info(f"Loading complete pretrained model from: {pretrained_path}")

    # Load the pretrained model
    loaded_object = torch.load(pretrained_path, map_location=device)

    if isinstance(loaded_object, torch.nn.Module):
        # If it's a complete model object, return it directly
        model = loaded_object.to(device)

        # Verify it's the right type of model
        if not isinstance(model, SO3LR):
            logging.warning(
                f"Loaded model type: {type(model).__name__}, "
                f"expected SO3LR"
            )

        # Ensure model has required attributes for training
        required_attrs = ["r_max"]
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(model, attr)
        ]
        if missing_attrs:
            raise AttributeError(
                f"Loaded model missing required attributes: "
                f"{missing_attrs}"
            )

        logging.info("Loaded complete model object")
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Loaded model has {total_params} parameters")

        # Log some key model attributes for debugging
        if hasattr(model, "r_max"):
            logging.info(f"Model r_max: {model.r_max}")
        if hasattr(model, "features_dim"):
            logging.info(f"Model features_dim: {model.features_dim}")
        if hasattr(model, "num_interactions"):
            logging.info(f"Model num_interactions: {model.num_interactions}")

        return model
    else:
        raise ValueError(
            f"Expected a complete model object, got: " f"{type(loaded_object)}"
        )


def load_pretrained_weights(
    model: torch.nn.Module, pretrained_path: str, device: torch.device
) -> None:
    """Load pretrained model weights into existing model."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_path}"
        )

    logging.info(f"Loading pretrained weights from: {pretrained_path}")

    # Load the pretrained weights
    loaded_object = torch.load(pretrained_path, map_location=device)

    if isinstance(loaded_object, torch.nn.Module):
        # If it's a model object, extract the state dict
        state_dict = loaded_object.state_dict()
        logging.info("Loaded model object, extracting state dict")
    elif isinstance(loaded_object, dict):
        if "model" in loaded_object:
            # Checkpoint format
            state_dict = loaded_object["model"]
            logging.info("Loaded checkpoint format")
        else:
            # Direct state dict
            state_dict = loaded_object
            logging.info("Loaded state dict format")
    else:
        raise ValueError(
            f"Unsupported pretrained model format: " f"{type(loaded_object)}"
        )

    # Load the state dict into our model
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )

    if missing_keys:
        logging.warning(
            f"Missing {len(missing_keys)} keys when loading "
            f"pretrained weights"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected {len(unexpected_keys)} keys when loading "
            f"pretrained weights"
        )

    logging.info("Successfully loaded pretrained weights")


def setup_training_tools(config: dict, model: torch.nn.Module) -> tuple:
    """Setup training tools like EMA, checkpoint handler, and logger."""
    # Setup metrics logger
    train_logger = MetricsLogger(
        directory=config["GENERAL"]["log_dir"],
        tag=config["GENERAL"]["name_exp"] + "_train",
    )
    valid_logger = MetricsLogger(
        directory=config["GENERAL"]["log_dir"],
        tag=config["GENERAL"]["name_exp"] + "_valid",
    )
    logger = {"train": train_logger, "valid": valid_logger}

    # Setup checkpoint handler
    checkpoint_handler = CheckpointHandler(
        directory=config["GENERAL"]["checkpoints_dir"],
        tag=config["GENERAL"]["name_exp"],
    )

    # Setup EMA if enabled
    ema = None
    if config["TRAINING"].get("ema", False):
        ema_decay = config["TRAINING"].get("ema_decay", 0.99)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        logging.info(f"Using EMA with decay: {ema_decay}")

    return logger, checkpoint_handler, ema


def load_checkpoint_if_exists(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_handler: CheckpointHandler,
    ema: ExponentialMovingAverage,
    device: torch.device,
    config: dict,
) -> int:
    """
    Load checkpoint if it exists and return the starting epoch.
    Returns 0 if no checkpoint is found.
    """
    # Check if we should restart from latest checkpoint
    restart_latest = config["MISC"].get("restart_latest", True)

    if not restart_latest:
        logging.info("Checkpoint restart disabled, starting from epoch 0")
        return 0

    # Create checkpoint state
    checkpoint_state = CheckpointState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
    )

    # Try to load latest checkpoint
    try:
        start_epoch = checkpoint_handler.load_latest(
            state=checkpoint_state,
            swa=False,  # Don't load SWA checkpoint initially
            device=device,
            strict=False,
        )

        if start_epoch is not None:
            logging.info(f"Loaded checkpoint from epoch {start_epoch}")

            # Load EMA state if available and EMA is enabled
            if ema is not None:
                try:
                    # Try to load EMA state from checkpoint
                    io_handler = checkpoint_handler.io
                    latest_path = io_handler._get_latest_checkpoint_path(
                        swa=False
                    )
                    if latest_path:
                        checkpoint_data = torch.load(
                            latest_path, map_location=device
                        )
                        if "ema" in checkpoint_data:
                            ema.load_state_dict(checkpoint_data["ema"])
                            logging.info("Loaded EMA state from checkpoint")
                except Exception as e:
                    logging.warning(f"Could not load EMA state: {e}")

            return start_epoch + 1  # Start from next epoch
        else:
            logging.info("No checkpoint found, starting from epoch 0")
            return 0

    except Exception as e:
        logging.warning(f"Error loading checkpoint: {e}")
        logging.info("Starting from epoch 0")
        return 0


def setup_finetuning(
    config: dict, model: torch.nn.Module, device_name: str
) -> None:
    # check that a pre-trained model is provided
    choice = config["TRAINING"].get("finetune_choice", None)
    if choice is not None:
        pretrained_model_given = (
            config["TRAINING"].get("pretrained_weights") is not None
            or config["TRAINING"].get("pretrained_model") is not None
        )
        assert pretrained_model_given, (
            "Finetuning requires a pretrained model. "
            "Please provide 'pretrained_weights' or 'pretrained_model' in the config."
        )
        logging.info(f"Setting up finetuning with choice: {choice}")

        lora_rank = config["TRAINING"].get("lora_rank", 4)
        lora_alpha = config["TRAINING"].get("lora_alpha", 2.0 * lora_rank)

        if choice == "lora" or choice == "lora+mlp":
            model = model_to_lora(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                freeze_A=config["TRAINING"].get("lora_freeze_A", False),
                device=device_name,
                seed=config["GENERAL"].get("seed", 42),
            )
            logging.info("Converted model to LoRA format")

        elif choice == "dora":
            model = model_to_lora(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                use_dora=True,
                device=device_name,
                scaling_to_one=config["TRAINING"].get(
                    "dora_scaling_to_one", True
                ),
            )
            logging.info("Converted model to DoRA format")

        elif choice == "vera":
            model = model_to_lora(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                use_vera=True,
                device=device_name,
                scaling_to_one=config["TRAINING"].get(
                    "vera_scaling_to_one", True
                ),
            )
            logging.info("Converted model to VeRA format")

        if choice != "naive":
            freeze_model_parameters(
                model,
                choice,
                freeze_lora_A=config["TRAINING"].get("lora_freeze_A", False),
            )
        # log number of trainable params, absolute and percentage
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logging.info(f"Total model parameters: {total_params}")
        logging.info(f"Trainable model parameters: {trainable_params}")
        logging.info(
            f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
        )


def run_training(config: dict) -> None:
    """
    Execute the complete training pipeline.

    Args:
        config: Configuration dictionary containing all training parameters
    """
    # Setup logging
    setup_logging(config)

    # Get pretrained model settings from config
    pretrained_weights = config["TRAINING"].get("pretrained_weights", None)
    pretrained_model = config["TRAINING"].get("pretrained_model", None)
    no_checkpoint = config["MISC"].get("no_checkpoint", False)

    # Validate pretrained settings
    if pretrained_weights and pretrained_model:
        raise ValueError(
            "Cannot specify both 'pretrained_weights' and "
            "'pretrained_model' in config. Use "
            "'pretrained_weights' for weights only or "
            "'pretrained_model' for complete model."
        )

    # Override checkpoint loading if specified in config
    if no_checkpoint:
        config["MISC"]["restart_latest"] = False

    # Setup device
    device_name = config["MISC"].get("device", "cuda")
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")

    # Handle model creation and pretrained loading
    if pretrained_model:
        # Load complete pretrained model (ignores config architecture)
        model = load_pretrained_model_direct(pretrained_model, device)
        logging.info(
            "Using complete pretrained model, ignoring config "
            "architecture settings"
        )
    else:
        # Create model from config
        model = create_model(config, device)

        # Load pretrained weights if specified
        if pretrained_weights:
            load_pretrained_weights(model, pretrained_weights, device)

    # Setup finetuning if specified
    setup_finetuning(config, model, device_name)

    # Setup data loaders
    train_loader, valid_loaders = setup_data_loaders(config, model)

    # Setup loss function
    loss_fn = setup_loss_function(config)

    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, config)

    # Setup training tools
    logger, checkpoint_handler, ema = setup_training_tools(config, model)

    # Load checkpoint if exists (unless pretrained model was loaded)
    start_epoch = 0
    if not pretrained_weights and not pretrained_model:
        # Skip checkpoint if any pretrained model was loaded
        start_epoch = load_checkpoint_if_exists(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            ema=ema,
            device=device,
            config=config,
        )

    logging.info("Model, data loaders, and training components initialized")
    if start_epoch > 0:
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("Starting training from scratch")

    # Setup training parameters
    max_num_epochs = config["TRAINING"]["num_epochs"]
    eval_interval = config["TRAINING"].get("eval_interval", 1)
    patience = config["TRAINING"].get("patience", 50)
    max_grad_norm = config["TRAINING"].get("clip_grad", 10.0)
    log_wandb = config["MISC"].get("log_wandb", False)
    save_all_checkpoints = config["MISC"].get("keep_checkpoints", False)

    # Setup output arguments for model evaluation
    output_args = {
        "forces": True,  # Always compute forces for training
        "virials": config["GENERAL"].get("compute_stress", False),
        "stress": config["GENERAL"].get("compute_stress", False),
    }

    # Get error logging type
    log_errors = config["MISC"].get("error_table", "PerAtomMAE")

    logging.info("Starting training loop...")
    # Start training
    train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        start_epoch=start_epoch,
        max_num_epochs=max_num_epochs,
        patience=patience,
        checkpoint_handler=checkpoint_handler,
        logger_train=logger["train"],
        logger_valid=logger["valid"],
        eval_interval=eval_interval,
        output_args=output_args,
        device=device,
        log_errors=log_errors,
        ema=ema,
        max_grad_norm=max_grad_norm,
        log_wandb=log_wandb,
        distributed=False,  # Single GPU training for now
        save_all_checkpoints=save_all_checkpoints,
        plotter=None,  # No plotting for now
        distributed_model=None,
        train_sampler=None,
        rank=0,
    )
    logging.info("Training completed successfully!")

    # TODO: fuse model weights if using LoRA before saving
    if config["TRAINING"].get("finetune_choice", None) in [
        "dora",
        "lora",
        "vera",
    ]:
        logging.info("Fusing LoRA weights into base model for saving...")
        model = fuse_lora_weights(model)
        logging.info("LoRA weights fused successfully.")
    # TODO: use EMA weights if enabled before saving

    # save the model in the working directory
    final_model_path = f'{config["GENERAL"]["name_exp"]}.pth'
    torch.save(model.state_dict(), final_model_path)
    torch.save(model, final_model_path.replace(".pth", ".model"))


def main():
    parser = argparse.ArgumentParser(description="Train SO3LR model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    # Load configuration
    config = setup_config_from_yaml(args.config)

    # Run the training pipeline
    run_training(config)


if __name__ == "__main__":
    main()
