#!/usr/bin/env python3
"""
validate_all.py - Find all checkpoints, run validation on each, and rank by CER.

Usage (from model/ directory):
    uv run python scripts/validate_all.py

Optional args:
    --logs-dir    Path to logs directory (default: logs/)
    --model       Model config name (default: gru_ctc)
    --pattern     Checkpoint filename pattern (default: last.ckpt)
    --output      CSV output file (default: scripts/validation_results.csv)

Example:
    uv run python scripts/validate_all.py --pattern "*.ckpt" --model gru_ctc
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def find_checkpoints(logs_dir: Path, pattern: str) -> list[Path]:
    ckpts = sorted(logs_dir.rglob(pattern))
    if not ckpts:
        print(f"No checkpoints found matching '{pattern}' under {logs_dir}")
    return ckpts


def parse_params_from_path(ckpt_path: Path) -> dict:
    """Extract hyperparameters from the job directory name."""
    # Walk up to find a directory with job params encoded in its name
    for parent in ckpt_path.parents:
        name = parent.name
        if "module." in name or "model=" in name:
            params = {}
            for part in name.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k.strip()] = v.strip()
            return params
    return {}


def run_validation(ckpt_path: Path, model: str) -> dict:
    """Run validation for a single checkpoint and parse metrics from output."""
    # Hydra requires the checkpoint value to be double-quoted inside single quotes
    # to handle = and , characters in the path
    ckpt_str = f'checkpoint="{ckpt_path}"'

    cmd = [
        "uv", "run", "python", "-m", "emg2qwerty.train",
        "user=single_user",
        f"model={model}",
        "train=false",
        "cluster=basic",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=40",
        ckpt_str,
    ]

    print(f"\n  Running: {ckpt_path}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per validation run
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}

    def extract(key):
        m = re.search(rf"'{key}':\s*([\d.]+)", output)
        return float(m.group(1)) if m else None

    metrics = {
        "val_CER":   extract("val/CER"),
        "test_CER":  extract("test/CER"),
        "val_IER":   extract("val/IER"),
        "val_DER":   extract("val/DER"),
        "val_SER":   extract("val/SER"),
        "val_loss":  extract("val/loss"),
        "test_loss": extract("test/loss"),
    }

    if metrics["val_CER"] is not None:
        print(f"  val_CER={metrics['val_CER']:.2f}  test_CER={metrics['test_CER']:.2f}  val_loss={metrics['val_loss']:.4f}")
    else:
        print("  Could not parse metrics — run may have failed")
        print("  Last 20 lines of output:")
        for line in output.splitlines()[-20:]:
            print(f"    {line}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate all checkpoints and rank by CER")
    parser.add_argument("--logs-dir", default="logs", help="Path to logs directory")
    parser.add_argument("--model", default="gru_ctc", help="Model config name")
    parser.add_argument("--pattern", default="last.ckpt", help="Checkpoint filename glob pattern")
    parser.add_argument("--output", default="scripts/validation_results.csv", help="CSV output path")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)

    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        sys.exit(1)

    # Find all checkpoints
    ckpts = find_checkpoints(logs_dir, args.pattern)
    print(f"Found {len(ckpts)} checkpoint(s)\n")

    results = []

    for i, ckpt in enumerate(ckpts, 1):
        print(f"[{i}/{len(ckpts)}] Validating checkpoint:")
        params = parse_params_from_path(ckpt)
        metrics = run_validation(ckpt, args.model)

        results.append({
            "checkpoint": str(ckpt),
            "hidden_size": params.get("module.hidden_size", "?"),
            "num_layers":  params.get("module.num_layers", "?"),
            "lr":          params.get("module.optimizer.lr", "?"),
            "dropout":     params.get("module.dropout", "?"),
            **metrics,
        })

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", "hidden_size", "num_layers", "lr", "dropout",
                  "val_CER", "test_CER", "val_IER", "val_DER", "val_SER",
                  "val_loss", "test_loss"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print ranked summary
    completed = [r for r in results if r.get("val_CER") is not None]
    incomplete = [r for r in results if r.get("val_CER") is None]

    print(f"\nCompleted: {len(completed)}/{len(results)}")
    if incomplete:
        print(f"Failed:    {len(incomplete)}")
        for r in incomplete:
            print(f"  {r['checkpoint']}")

    if not completed:
        print("\nNo completed runs to summarize.")
        return

    completed.sort(key=lambda r: r["val_CER"])

    print(f"\n{'Rank':<5} {'hidden':>8} {'layers':>7} {'lr':>8} {'dropout':>8} "
          f"{'val_CER':>9} {'test_CER':>10} {'val_loss':>10}")
    print("-" * 75)
    for rank, r in enumerate(completed, 1):
        print(f"{rank:<5} {r['hidden_size']:>8} {r['num_layers']:>7} {r['lr']:>8} {r['dropout']:>8} "
              f"{r['val_CER']:>9.2f} {r['test_CER']:>10.2f} {r['val_loss']:>10.4f}")

    best = completed[0]
    print(f"\nBest checkpoint (val_CER={best['val_CER']:.2f}):")
    print(f"  hidden={best['hidden_size']}, layers={best['num_layers']}, "
          f"lr={best['lr']}, dropout={best['dropout']}")
    print(f"  {best['checkpoint']}")


if __name__ == "__main__":
    main()
