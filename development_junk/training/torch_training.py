from so3krates_torch.tools.train import (
    take_step
)
from so3krates_torch.modules.models import So3krates
from mace.tools import utils, scripts_utils, torch_geometric
from mace.modules import loss
from mace import data
import torch
from mace.modules.utils import compute_avg_num_neighbors
z_table = utils.AtomicNumberTable(
            [int(z) for z in range(1, 119)]
        )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
import time

r_max = 4.5
batch_size = 100
keyspec = data.KeySpecification(
    info_keys={
        "energy": "energy",
        "total_charge":"total_charge",
        "total_spin":"total_spin",
        "stress":"stress",
        "head":"default"
        }, 
    arrays_keys={
        "charges": "Qs",
        "forces": "forces",
        }
)
collections, atomic_energies_dict = scripts_utils.get_dataset_from_xyz(
    work_dir='.',
    train_path="des15k_10percent.extxyz",
    valid_path=None,
    valid_fraction=0.1, 
    seed=seed,
    key_specification=keyspec
)
train_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
        ) for config in collections.train
    ],
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
valid_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
        ) for config in collections.valid
    ],
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

avg_num_neighbors = compute_avg_num_neighbors(
    train_loader
)

degrees = [1,2,3,4]
model = So3krates(
    r_max=5.0,
    num_radial_basis=32,
    degrees=degrees,
    features_dim=132,
    num_att_heads=4,
    final_mlp_layers=2, 
    num_interactions=3,
    num_elements=len(z_table),
    avg_num_neighbors=avg_num_neighbors,
    message_normalization='avg_num_neighbors',
    seed=42,
    device=device,
    trainable_rbf=True,
    learn_atomic_type_shifts=True,
    learn_atomic_type_scales=True,
    energy_regression_dim=128,  # This is the energy regression dimension
    layer_normalization_1=True,
    layer_normalization_2=True,
    residual_mlp_1=True,
    residual_mlp_2=False,
    use_charge_embed=True,
    use_spin_embed=True,
).to(device).train()

loss_fn = loss.WeightedEnergyForcesLoss(
    energy_weight=1.0,
    forces_weight=10.
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    amsgrad=True
)
output_args = {
    "forces": True,
    "stress": False,
    "virials": False,
}

total_time = time.perf_counter()
epoch_times = []
step_times = []
step = 0
for epoch in range(5):
    epoch_time = time.perf_counter()
    for batch in train_loader:
        iteration_time = time.perf_counter()
        loss_val, loss_dict = take_step(
            model=model,
            loss_fn=loss_fn,
            batch=batch.to(device),
            optimizer=optimizer,
            device=device,
            ema=None,
            max_grad_norm=10,
            output_args=output_args
        )
        iteration_time = time.perf_counter() - iteration_time
        step_times.append(iteration_time)
        print(f"Step {step+1} completed in {iteration_time:.4f} seconds.")
        step += 1
        
    epoch_time = time.perf_counter() - epoch_time
    epoch_times.append(epoch_time)
    print(f"Epoch {epoch+1} completed in {epoch_time:.4f} seconds.")
    print(f"{loss_val.item():.4f}")
total_time = time.perf_counter() - total_time
print(f"Total training time: {total_time:.4f} seconds.")
print(f"Average time per epoch: {sum(epoch_times) / len(epoch_times):.4f} seconds.")
print(f"Average time per iteration: {sum(step_times) / len(step_times):.4f} seconds.")
print(f"Iterations per second: {len(step_times) / sum(step_times):.4f}")
print(f"Epochs per second: {len(epoch_times) / sum(epoch_times):.4f}")
