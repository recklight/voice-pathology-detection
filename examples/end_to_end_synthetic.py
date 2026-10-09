"""End-to-end demonstration on synthetic audio.

Synthesises a small corpus of sustained vowels, indexes it, extracts cepstral
features, cross-validates the two-stage cascade and the single-stage four-way
classifier, and writes the metrics and figures of each run.  No real recording
is involved.

The material is synthetic and tiny, so the numbers say nothing about real
voices.  What it does show is that every step runs from audio to report and
that a run repeats: the same seed gives the same answer, and everything needed
to audit a run lands in its own directory.

Run it with::

    python examples/end_to_end_synthetic.py --out-dir results/example
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from voice_pathology_detection import (
    Config,
    build_index,
    configure_logging,
    extract_corpus,
    run_cascade,
    run_flat,
    save_run,
    set_global_seed,
    write_synthetic_corpus,
)
from voice_pathology_detection.data.labels import CLASS_NAMES
from voice_pathology_detection.eval.metrics import folds_to_frame
from voice_pathology_detection.eval.plots import plot_confusion_matrix, plot_fold_errorbar, plot_roc


def build_config(data_root: Path, out_dir: Path, sample_rate: int, seed: int) -> Config:
    """Assemble the configuration for the demonstration.

    The model settings differ from the package defaults on purpose: the
    defaults suit the hundreds of thousands of frames a real corpus yields,
    while a few thousand synthetic frames need a larger step size, more epochs
    and no early-stopping hold-out.
    """
    network = {
        "hidden_layer_sizes": [64, 64],
        "max_iter": 300,
        "batch_size": 64,
        "learning_rate_init": 0.01,
        "momentum": 0.9,
        "early_stopping": False,
    }
    return Config().override(
        {
            "paths.data_root": str(data_root),
            "paths.results_dir": str(out_dir),
            "paths.cache_dir": str(data_root / "cache"),
            "seed": seed,
            "split.n_splits": 3,
            "split.val_fraction": 0.35,
            "mfcc.sample_rate": sample_rate,
            "mfcc.frame_ms": 25.0,
            "mfcc.hop_ms": 10.0,
            "mfcc.high_freq": min(7800.0, sample_rate / 2.0 - 100.0),
            "trim.mode": "none",
            "normalization.method": "cmvn",
            "detection.model.name": "mlp",
            "detection.model.params": dict(network),
            "detection.threshold_method": "youden",
            "disorder.model.name": "mlp",
            "disorder.model.params": dict(network),
            "flat.model.name": "mlp",
            "flat.model.params": dict(network),
        }
    )


def main() -> int:
    """Run the demonstration and report where the artefacts landed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/example"),
        help="Directory the run folders are written to.",
    )
    parser.add_argument("--per-class", type=int, default=10, help="Recordings per class.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sampling rate, Hz.")
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds per recording.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the whole run.")
    args = parser.parse_args()

    configure_logging("INFO")
    set_global_seed(args.seed)

    with tempfile.TemporaryDirectory(prefix="voice-pathology-detection-example-") as workspace:
        root = Path(workspace)

        print("\n1. Synthesising a corpus")
        corpus = write_synthetic_corpus(
            root,
            dict.fromkeys(CLASS_NAMES, args.per_class),
            sr=args.sample_rate,
            duration=args.duration,
            seed=args.seed,
        )
        print(f"   wrote {args.per_class * len(CLASS_NAMES)} recording(s) to {corpus}")

        cfg = build_config(root, args.out_dir, args.sample_rate, args.seed)

        print("\n2. Indexing and extracting features")
        index = build_index(cfg, expected_counts=dict.fromkeys(CLASS_NAMES, args.per_class))
        features = extract_corpus(index, cfg, cache_dir=cfg.paths.cache_dir)
        print(
            f"   {features.n_frames} frame(s) of {features.n_dim} dimension(s) "
            f"from {features.n_files} recording(s)"
        )

        print("\n3. Cross-validating the two-stage cascade")
        cascade = run_cascade(features, cfg)
        cascade_dir = save_run(
            cfg.paths.results_dir,
            cfg,
            cascade.to_dict(),
            run_id="example-cascade",
            tables={"predictions": cascade.predictions},
            title="Cascade on synthetic audio",
        )
        names = dict(CLASS_NAMES)
        plot_confusion_matrix(
            cascade.pooled_overall,
            cascade_dir / "figures" / "confusion_four_way.png",
            names=names,
            title="Cascade: merged four-way decision",
        )
        plot_roc(
            cascade.predictions["y_true_binary"],
            cascade.predictions["score"],
            cascade_dir / "figures" / "detection_roc.png",
            title="Cascade: detection ROC",
        )

        print("\n4. Cross-validating the single-stage four-way classifier")
        flat = run_flat(features, cfg)
        flat_dir = save_run(
            cfg.paths.results_dir,
            cfg,
            flat.to_dict(),
            run_id="example-flat",
            tables={
                "predictions": flat.predictions,
                "folds_subject": folds_to_frame(
                    [outcome.subject_metrics for outcome in flat.folds]
                ),
            },
            title="Single-stage four-way classifier on synthetic audio",
        )
        plot_confusion_matrix(
            flat.pooled_subject,
            flat_dir / "figures" / "confusion_four_way.png",
            names=names,
            title="Single stage: four-way decision",
        )
        plot_fold_errorbar(
            folds_to_frame([outcome.subject_metrics for outcome in flat.folds]),
            flat_dir / "figures" / "folds.png",
            title="Single stage: subject level across folds",
        )

    print("\n5. Results")
    print(f"{'route':<24}{'UAR':>10}{'accuracy':>12}{'sensitivity':>14}")
    print("-" * 60)
    print(
        f"{'cascade (two stage)':<24}"
        f"{cascade.overall_summary['uar']['mean']:>10.4f}"
        f"{cascade.overall_summary['accuracy']['mean']:>12.4f}"
        f"{cascade.detection_summary['sensitivity']['mean']:>14.4f}"
    )
    print(
        f"{'flat (single stage)':<24}"
        f"{flat.subject_summary['uar']['mean']:>10.4f}"
        f"{flat.subject_summary['accuracy']['mean']:>12.4f}"
        f"{flat.subject_summary['sensitivity']['mean']:>14.4f}"
    )
    print(f"\nDetection threshold per fold: {[round(value, 4) for value in cascade.thresholds]}")
    print(f"Subjects forwarded per fold:  {cascade.forwarded_per_fold}")
    print(f"\nArtefacts:\n  {cascade_dir}\n  {flat_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
