from so3krates_torch.tools.eval import test_ensemble
from so3krates_torch.tools.utils import ensemble_from_folder
import torch
import argparse
import numpy as np
from prettytable import PrettyTable
import logging


def main():
    parser = argparse.ArgumentParser(description="Test ensemble of models")
    parser.add_argument(
        "--models", type=str, help="Path to models", required=True
    )
    parser.add_argument(
        "--data", type=str, help="Path to dataset", required=True
    )
    parser.add_argument(
        "--output_args",
        type=str,
        nargs="+",
        help="List of output arguments",
        default=["energy", "forces"],
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size", default=16
    )
    parser.add_argument("--device", type=str, help="Device", default="cpu")
    parser.add_argument(
        "--save",
        type=str,
        help="Path to folder where save results",
        default="./",
    )
    parser.add_argument(
        "--return_predictions",
        type=bool,
        help="Return predictions",
        default=False,
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file",
        default="test_ensemble.log",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to file where save results",
        default="ensemble_test_results.npz",
    )
    parser.add_argument(
        "--r_max_lr",
        type=float,
        help="Max radius for long-range potentials",
        default=12.0,
    )
    parser.add_argument(
        "--dispersion_energy_cutoff_lr_damping",
        type=float,
        help="Dispersion energy cutoff for long-range potentials",
        default=2.0,
    )
    parser.add_argument(
        "--energy_key", type=str, help="Energy key", default="REF_energy"
    )
    parser.add_argument(
        "--forces_key", type=str, help="Forces key", default="REF_forces"
    )
    parser.add_argument(
        "--stress_key", type=str, help="Stress key", default="REF_stress"
    )
    parser.add_argument(
        "--virials_key", type=str, help="Virials key", default="REF_virials"
    )
    parser.add_argument(
        "--dipole_key", type=str, help="Dipole key", default="REF_dipoles"
    )
    parser.add_argument(
        "--charges_key", type=str, help="Charges key", default="REF_charges"
    )
    parser.add_argument(
        "--hirshfeld_key",
        type=str,
        help="Hirshfeld key",
        default="REF_hirsh_ratios",
    )

    args = parser.parse_args()
    # setup logger to save to 'test_ensemble.log'
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info("Starting test.")
    ensemble = ensemble_from_folder(
        path_to_models=args.models, device=args.device, dtype=torch.float32
    )

    for model in ensemble.values():
        if args.r_max_lr is not None:
            model.r_max_lr = args.r_max_lr
        if args.dispersion_energy_cutoff_lr_damping is not None:
            model.dispersion_energy_cutoff_lr_damping = (
                args.dispersion_energy_cutoff_lr_damping
            )

    possible_args = [
        "energy",
        "forces",
        "stress",
        "virials",
        "dipole",
        "hirshfeld_ratios",
    ]
    output_args = {}
    for arg in args.output_args:
        arg = arg.lower()
        if arg not in possible_args:
            raise ValueError("Invalid output argument")
        if arg == "energy":
            output_args["energy"] = True
        if arg == "forces":
            output_args["forces"] = True
        if arg == "stress":
            output_args["stress"] = True
        if arg == "virials":
            output_args["virials"] = True
        if arg == "dipole":
            output_args["dipole"] = True
        if arg == "hirshfeld_ratios":
            output_args["hirshfeld_ratios"] = True

    for arg in possible_args:
        if arg not in output_args:
            output_args[arg] = False

    (avg_ensemble_metrics, ensemble_metrics) = test_ensemble(
        ensemble=ensemble,
        path_to_data=args.data,
        batch_size=args.batch_size,
        output_args=output_args,
        device=args.device,
        return_predictions=args.return_predictions,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
        r_max_lr=args.r_max_lr,
        log=True,  # Enable logging for detailed output
    )
    results = {
        "avg_ensemble_metrics": avg_ensemble_metrics,
        "ensemble_metrics": ensemble_metrics,
    }
    np.savez(args.save + args.results_file, **results)

    table = PrettyTable()
    table.field_names = [
        "Model",
        "MAE E [meV/Atom]",
        "RMSE E [meV/Atom]",
        "MAE F [meV/(A*atom)]",
        "RMSE F [meV/(A*atom)]",
    ]
    for i in ensemble.keys():
        table.add_row(
            [
                i,
                f"{ensemble_metrics[i]['mae_e_per_atom'] * 10e2:.1f}",
                f"{ensemble_metrics[i]['rmse_e_per_atom'] * 10e2:.1f}",
                f"{ensemble_metrics[i]['mae_f'] * 10e2:.1f}",
                f"{ensemble_metrics[i]['rmse_f'] * 10e2:.1f}",
            ]
        )
    table.add_row(
        [
            "Average",
            f"{avg_ensemble_metrics['mae_e_per_atom'] * 10e2:.1f}",
            f"{avg_ensemble_metrics['rmse_e_per_atom'] * 10e2:.1f}",
            f"{avg_ensemble_metrics['mae_f'] * 10e2:.1f}",
            f"{avg_ensemble_metrics['rmse_f'] * 10e2:.1f}",
        ]
    )

    print(table)
    logging.info(table)


if __name__ == "__main__":
    main()
