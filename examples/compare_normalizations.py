"""Compare normalisation methods and classifiers on the same folds.

Cross-validates the detection stage once per (normalisation, model) pair, on
identical subject-grouped folds and with the same seed, so that two rows differ
only in the setting they name.  Four of the six normalisation methods are
compared; ``cmn`` and ``cvn`` are left out because they are the two halves of
``cmvn``.

Runs on synthetic audio and needs no corpus::

    python examples/compare_normalizations.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from voice_pathology_detection import (
    Config,
    build_index,
    configure_logging,
    extract_corpus,
    run_detection,
    set_global_seed,
    write_synthetic_corpus,
)
from voice_pathology_detection.data.labels import CLASS_NAMES

#: Normalisation methods compared, in the order they are run.
STRATEGIES: tuple[str, ...] = ("none", "cvn_block", "cmvn", "cmvn_per_utt")

#: Models compared, with settings that suit a few thousand frames.
MODELS: dict[str, dict[str, object]] = {
    "mlp": {
        "hidden_layer_sizes": [64, 64],
        "max_iter": 250,
        "batch_size": 64,
        "learning_rate_init": 0.01,
        "momentum": 0.9,
        "early_stopping": False,
    },
    "svm": {"kernel": "rbf", "C": 1.0},
    "gmm": {"n_components": 2, "priors": "uniform"},
}


def main() -> int:
    """Run the comparison and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=8, help="Recordings per class.")
    parser.add_argument("--folds", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the whole comparison.")
    parser.add_argument(
        "--out-csv", type=Path, default=None, help="Optional path for the table as CSV."
    )
    args = parser.parse_args()

    configure_logging("WARNING")
    set_global_seed(args.seed)

    with tempfile.TemporaryDirectory(prefix="voice-pathology-detection-compare-") as workspace:
        root = Path(workspace)
        write_synthetic_corpus(
            root,
            dict.fromkeys(CLASS_NAMES, args.per_class),
            sr=16000,
            duration=1.0,
            seed=args.seed,
        )

        base = Config().override(
            {
                "paths.data_root": str(root),
                "paths.cache_dir": str(root / "cache"),
                "seed": args.seed,
                "split.n_splits": args.folds,
                "split.val_fraction": 0.35,
                "mfcc.sample_rate": 16000,
                "mfcc.frame_ms": 25.0,
                "mfcc.hop_ms": 10.0,
                "mfcc.high_freq": 7800.0,
                "trim.mode": "none",
            }
        )

        # The features depend only on the front end, so they are extracted once
        # and reused: every row below sees exactly the same frames.
        index = build_index(base, expected_counts=dict.fromkeys(CLASS_NAMES, args.per_class))
        features = extract_corpus(index, base, cache_dir=base.paths.cache_dir)
        print(
            f"{features.n_frames} frame(s) of {features.n_dim} dimension(s) "
            f"from {features.n_files} recording(s)\n"
        )

        rows: list[dict[str, object]] = []
        for model_name, params in MODELS.items():
            for strategy in STRATEGIES:
                cfg = base.override(
                    {
                        "normalization.method": strategy,
                        "detection.model.name": model_name,
                        "detection.model.params": params,
                    }
                )
                result = run_detection(features, cfg)
                rows.append(
                    {
                        "model": model_name,
                        "normalization": strategy,
                        "uar": result.subject_summary["uar"]["mean"],
                        "uar_std": result.subject_summary["uar"]["std"],
                        "sensitivity": result.subject_summary["sensitivity"]["mean"],
                        "specificity": result.subject_summary["specificity"]["mean"],
                        "roc_auc": result.subject_summary["roc_auc"]["mean"],
                        "threshold": sum(result.thresholds) / len(result.thresholds),
                    }
                )
                print(
                    f"  {model_name:<5} {strategy:<14} "
                    f"UAR {rows[-1]['uar']:.4f} +/- {rows[-1]['uar_std']:.4f}"
                )

    table = pd.DataFrame(rows).sort_values("uar", ascending=False, ignore_index=True)
    print("\nDetection stage, subject level, identical folds and seed:\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")

    best = table.iloc[0]
    print(
        f"\nBest here: {best['model']} with {best['normalization']} "
        f"(UAR {best['uar']:.4f}). On synthetic data this ranking is an "
        "illustration, not a finding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
