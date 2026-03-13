"""Benchmark and correctness comparison for So3krates-torch data loaders.

Compares three data-loading modes:
  eager        -- original: all graphs built at startup (baseline)
  lazy         -- atoms loaded at init, graphs built on __getitem__
  lazy-file    -- fully lazy: frames read from disk on __getitem__
  cached-cold  -- first run of cached mode (populating disk cache)
  cached-warm  -- second run of cached mode (reading from disk cache)

Usage
-----
  python examples/benchmark_dataloaders.py \\
      --data examples/training/md17_ethanol_small.xyz \\
      --r_max 5.0 \\
      --batch_size 5 \\
      --epochs 3
"""

import argparse
import os
import shutil
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from ase.io import read as ase_read
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric

from so3krates_torch.data.datasets import DiskCache, XYZDataset
from so3krates_torch.tools.utils import (
    create_dataloader_from_dataset,
    create_dataloader_from_list,
)

# Default key spec matching training pipeline
_KEYSPEC = KeySpecification(
    info_keys={
        "energy": "REF_energy",
        "dipole": "REF_dipole",
        "total_charge": "charge",
    },
    arrays_keys={
        "hirshfeld_ratios": "REF_hirsh_ratios",
        "forces": "REF_forces",
    },
)


@dataclass
class BenchmarkResult:
    mode: str
    startup_s: float
    peak_ram_mb: float
    frames_per_s: float
    atoms_per_s: float
    batch_latency_mean_ms: float
    batch_latency_p95_ms: float
    n_batches: int
    n_frames: int
    n_epochs: int
    errors: List[str] = field(default_factory=list)


def _build_eager_loader(data_path, r_max, batch_size):
    """Original eager pipeline: read all atoms, build all graphs upfront."""
    atoms_list = ase_read(data_path, index=":")
    loader = create_dataloader_from_list(
        atoms_list,
        batch_size=batch_size,
        r_max=r_max,
        r_max_lr=None,
        key_specification=_KEYSPEC,
        shuffle=False,
    )
    return loader, len(atoms_list)


def _build_dataset_loader(
    data_path, r_max, batch_size, lazy, cache=None
):
    """New map-style pipeline."""
    ds = XYZDataset(
        file_path=data_path,
        cutoff=r_max,
        cutoff_lr=None,
        key_specification=_KEYSPEC,
        cache=cache,
        lazy=lazy,
    )
    loader = create_dataloader_from_dataset(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, len(ds)


def run_benchmark(
    mode: str,
    data_path: str,
    r_max: float,
    batch_size: int,
    n_epochs: int,
    device: str,
    cache_dir: Optional[str] = None,
) -> BenchmarkResult:
    """Run one benchmark pass and return a BenchmarkResult."""
    print(f"\n[{mode}] building loader...", flush=True)
    tracemalloc.start()
    t0 = time.perf_counter()

    if mode == "eager":
        loader, n_frames = _build_eager_loader(data_path, r_max, batch_size)
    elif mode == "lazy":
        loader, n_frames = _build_dataset_loader(
            data_path, r_max, batch_size, lazy=False
        )
    elif mode == "lazy-file":
        loader, n_frames = _build_dataset_loader(
            data_path, r_max, batch_size, lazy=True
        )
    elif mode in ("cached-cold", "cached-warm"):
        cache = DiskCache(
            cache_dir=cache_dir,
            cutoff=r_max,
            source_file=data_path,
        )
        loader, n_frames = _build_dataset_loader(
            data_path, r_max, batch_size, lazy=False, cache=cache
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    startup_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / 1024 / 1024

    print(
        f"[{mode}] startup={startup_s:.3f}s  "
        f"peak_ram={peak_ram_mb:.1f}MB  "
        f"running {n_epochs} epochs...",
        flush=True,
    )

    batch_times = []
    total_atoms = 0
    dev = torch.device(device)

    for epoch in range(n_epochs):
        for batch in loader:
            t_batch = time.perf_counter()
            batch = batch.to(dev)
            batch_times.append(time.perf_counter() - t_batch)
            total_atoms += int(batch.positions.shape[0])

    n_batches = len(batch_times)
    total_frames = n_frames * n_epochs
    elapsed_s = sum(batch_times)
    frames_per_s = total_frames / elapsed_s if elapsed_s > 0 else 0.0
    atoms_per_s = total_atoms / elapsed_s if elapsed_s > 0 else 0.0

    times_ms = np.array(batch_times) * 1000
    return BenchmarkResult(
        mode=mode,
        startup_s=startup_s,
        peak_ram_mb=peak_ram_mb,
        frames_per_s=frames_per_s,
        atoms_per_s=atoms_per_s,
        batch_latency_mean_ms=float(np.mean(times_ms)),
        batch_latency_p95_ms=float(np.percentile(times_ms, 95)),
        n_batches=n_batches,
        n_frames=n_frames,
        n_epochs=n_epochs,
    )


def check_correctness(data_path, r_max, batch_size):
    """Assert that eager and lazy modes produce identical AtomicData."""
    print("\n[correctness] comparing eager vs lazy...", flush=True)
    eager_loader, _ = _build_eager_loader(data_path, r_max, batch_size)
    lazy_loader, _ = _build_dataset_loader(
        data_path, r_max, batch_size, lazy=False
    )

    for i, (b_eager, b_lazy) in enumerate(zip(eager_loader, lazy_loader)):
        for attr in ("positions", "energy", "forces", "atomic_numbers"):
            t_e = getattr(b_eager, attr, None)
            t_l = getattr(b_lazy, attr, None)
            if t_e is None or t_l is None:
                continue
            if not torch.allclose(t_e.float(), t_l.float(), atol=1e-5):
                raise AssertionError(
                    f"Batch {i}: mismatch in '{attr}'\n"
                    f"  eager max diff = {(t_e - t_l).abs().max():.2e}"
                )
    print("[correctness] PASSED — eager and lazy outputs are identical.")


def print_table(results: List[BenchmarkResult]) -> None:
    header = (
        f"{'Mode':<16} {'Startup(s)':>10} {'RAM(MB)':>9} "
        f"{'Frames/s':>10} {'Atoms/s':>10} "
        f"{'Batch mean(ms)':>15} {'Batch p95(ms)':>14}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.mode:<16} {r.startup_s:>10.3f} {r.peak_ram_mb:>9.1f} "
            f"{r.frames_per_s:>10.1f} {r.atoms_per_s:>10.0f} "
            f"{r.batch_latency_mean_ms:>15.2f} "
            f"{r.batch_latency_p95_ms:>14.2f}"
        )
    print(sep)

    # Highlight speedups relative to eager baseline
    eager = next((r for r in results if r.mode == "eager"), None)
    if eager:
        print(f"\nSpeedup relative to eager (frames/s):")
        for r in results:
            if r.mode == "eager":
                continue
            speedup = r.frames_per_s / eager.frames_per_s
            print(f"  {r.mode:<16} {speedup:+.2f}x")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(here, "training", "md17_ethanol_small.xyz")

    parser = argparse.ArgumentParser(
        description="Benchmark So3krates-torch data loaders"
    )
    parser.add_argument(
        "--data",
        default=default_data,
        help="Path to XYZ/ExtXYZ dataset file",
    )
    parser.add_argument(
        "--r_max", type=float, default=5.0, help="Short-range cutoff (Å)"
    )
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of full passes for throughput measurement",
    )
    parser.add_argument(
        "--device", default="cuda", help="torch device (cpu or cuda)"
    )
    parser.add_argument(
        "--skip_correctness",
        action="store_true",
        help="Skip correctness check",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    print(f"Dataset : {args.data}")
    print(f"r_max   : {args.r_max}")
    print(f"batch   : {args.batch_size}")
    print(f"epochs  : {args.epochs}")
    print(f"device  : {args.device}")

    if not args.skip_correctness:
        check_correctness(args.data, args.r_max, args.batch_size)

    cache_dir = tempfile.mkdtemp(prefix="so3cache_bench_")
    try:
        modes = [
            ("eager", {}),
            ("lazy", {}),
            ("lazy-file", {}),
            ("cached-cold", {"cache_dir": cache_dir}),
            ("cached-warm", {"cache_dir": cache_dir}),
        ]

        results = []
        for mode, kwargs in modes:
            result = run_benchmark(
                mode=mode,
                data_path=args.data,
                r_max=args.r_max,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                device=args.device,
                **kwargs,
            )
            results.append(result)

        print_table(results)

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
