import argparse
from so3krates_torch.tools.eval import evaluate_model, ensemble_prediction
from so3krates_torch.tools.utils import save_results_hdf5
from ase.io import read
import torch
from pathlib import Path


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument("--output_file", type=str, default="results.h5")
    argparser.add_argument("--ensemble_size", type=int, default=1)
    argparser.add_argument("--device", type=str, default="cuda")
    argparser.add_argument("--batch_size", type=int, default=5)
    argparser.add_argument("--model_type", type=str, default="so3lr")
    argparser.add_argument("--r_max_lr", type=float, default=12.0)
    argparser.add_argument(
        "--multispecies", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_dipole", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_stress", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_hirshfeld", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_partial_charges", action="store_true", default=False
    )
    argparser.add_argument(
        "--dispersion_energy_cutoff_lr_damping", type=float, default=2.0
    )
    argparser.add_argument("--dtype", type=str, default="float32")
    argparser.add_argument("--return_att", action="store_true", default=False)
    args = argparser.parse_args()

    # check if path ends with .model or is dir
    if args.model_path.endswith(".model"):
        model_paths = [args.model_path]
    else:
        model_paths = list(Path(args.model_path).glob("*.model"))

    models = []
    for model_path in model_paths:
        model = torch.load(
            model_path, map_location=args.device, weights_only=False
        )
        models.append(model)

    data = read(args.data_path, index=":")

    if len(models) == 1:
        model = models[0]
        result = evaluate_model(
            atoms_list=data,
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            model_type=args.model_type,
            r_max_lr=args.r_max_lr,
            dispersion_energy_cutoff_lr_damping=args.dispersion_energy_cutoff_lr_damping,
            dtype=args.dtype,
            multi_species=args.multispecies,
            compute_stress=args.compute_stress,
            compute_dipole=args.compute_dipole,
            compute_hirshfeld=args.compute_hirshfeld,
            compute_partial_charges=args.compute_partial_charges,
            return_att=args.return_att,
        )
    else:
        result = ensemble_prediction(
            models=models,
            atoms_list=data,
            device=args.device,
            model_type=args.model_type,
            dtype=args.dtype,
            batch_size=args.batch_size,
            multi_species=args.multispecies,
            r_max_lr=args.r_max_lr,
            dispersion_energy_cutoff_lr_damping=args.dispersion_energy_cutoff_lr_damping,
            compute_stress=args.compute_stress,
            compute_dipole=args.compute_dipole,
            compute_hirshfeld=args.compute_hirshfeld,
            compute_partial_charges=args.compute_partial_charges,
        )

    save_results_hdf5(result, args.output_file, is_ensemble=len(models) > 1)
