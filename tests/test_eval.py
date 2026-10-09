"""Aggregation, threshold selection, metrics, reports and figures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from voice_pathology_detection.config import Config
from voice_pathology_detection.eval.aggregate import aggregate_frames, frame_offsets, split_by_group
from voice_pathology_detection.eval.metrics import (
    confusion_frame,
    evaluate,
    folds_to_frame,
    predictive_values,
    summarise_folds,
)
from voice_pathology_detection.eval.plots import (
    plot_confusion_matrix,
    plot_fold_errorbar,
    plot_roc,
    plot_spectrogram_grid,
    plot_threshold_curve,
)
from voice_pathology_detection.eval.report import (
    NumpyJSONEncoder,
    new_run_id,
    render_markdown,
    save_run,
    write_json,
)
from voice_pathology_detection.eval.threshold import (
    ThresholdChoice,
    roc_table,
    select_threshold,
    threshold_sweep,
)


# -- aggregation --------------------------------------------------------
def test_offsets_mark_the_start_of_every_recording() -> None:
    assert frame_offsets([2, 3, 1]).tolist() == [0, 2, 5]


def test_an_empty_recording_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        frame_offsets([2, 0, 1])


def test_splitting_by_group_recovers_the_recordings() -> None:
    parts = split_by_group(np.arange(6), [2, 4])
    assert [p.tolist() for p in parts] == [[0, 1], [2, 3, 4, 5]]


def test_mean_probability_averages_over_the_frames_of_a_recording() -> None:
    proba = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.3, 0.7]])
    result = aggregate_frames(proba, [2, 2], mode="mean_proba")
    assert np.allclose(result.proba, [[0.7, 0.3], [0.2, 0.8]])
    assert result.pred.tolist() == [0, 1]
    assert np.allclose(result.score(1), [0.3, 0.8])


def test_majority_vote_counts_hard_decisions_only() -> None:
    # Two of the three frames vote for class 0, but the mean posterior favours
    # class 1, so the two rules disagree on this recording.
    proba = np.array([[0.51, 0.49], [0.51, 0.49], [0.0, 1.0]])
    votes = aggregate_frames(proba, [3], mode="majority_vote")
    means = aggregate_frames(proba, [3], mode="mean_proba")
    assert votes.pred.tolist() == [0]
    assert means.pred.tolist() == [1]
    assert np.allclose(votes.ratio, [[2 / 3, 1 / 3]])


def test_the_ratio_rule_thresholds_the_share_of_positive_frames() -> None:
    proba = np.array([[0.2, 0.8]] * 3 + [[0.9, 0.1]] + [[0.2, 0.8]] + [[0.9, 0.1]] * 3)
    strict = aggregate_frames(proba, [4, 4], mode="ratio_threshold", threshold=0.9)
    lenient = aggregate_frames(proba, [4, 4], mode="ratio_threshold", threshold=0.25)
    assert strict.pred.tolist() == [0, 0]
    assert lenient.pred.tolist() == [1, 1]
    assert strict.threshold == 0.9


def test_the_ratio_rule_needs_a_threshold_and_two_classes() -> None:
    proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="requires a threshold"):
        aggregate_frames(proba, [2], mode="ratio_threshold")
    with pytest.raises(ValueError, match="must not be negative"):
        aggregate_frames(proba, [2], mode="ratio_threshold", threshold=-0.1)
    with pytest.raises(ValueError, match="only defined for two classes"):
        aggregate_frames(
            np.full((2, 3), 1 / 3),
            [2],
            mode="ratio_threshold",
            threshold=0.5,
            classes=[1, 2, 3],
        )


def test_aggregation_validates_its_inputs() -> None:
    proba = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="unknown aggregation mode"):
        aggregate_frames(proba, [1], mode="median")
    with pytest.raises(ValueError, match="must not contain negative"):
        aggregate_frames(np.array([[-0.1, 1.1]]), [1])
    with pytest.raises(ValueError, match="group_sizes cover"):
        aggregate_frames(proba, [2])
    with pytest.raises(ValueError, match="at least two columns"):
        aggregate_frames(np.array([[1.0]]), [1])


def test_custom_class_labels_are_carried_through() -> None:
    proba = np.array([[0.1, 0.2, 0.7]])
    result = aggregate_frames(proba, [1], classes=[1, 2, 3])
    assert result.pred.tolist() == [3]
    assert result.classes.tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="not among the classes"):
        result.score(0)


# -- thresholds ---------------------------------------------------------
def test_youden_finds_the_perfect_split() -> None:
    choice = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], method="youden")
    assert choice.sensitivity == 1.0
    assert choice.specificity == 1.0
    assert 0.2 < choice.threshold <= 0.8
    assert choice.auc == 1.0


def test_a_tuned_threshold_sits_halfway_between_the_classes() -> None:
    # Every threshold in (0.2, 0.8] separates these scores; the upper end would
    # call a new subject scoring 0.75 normal although it ranks with the others.
    choice = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], method="youden")
    assert choice.threshold == pytest.approx(0.5)

    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    target = select_threshold(y, score, method="target_sensitivity", target=1.0)
    assert target.threshold == pytest.approx(0.3)
    assert target.sensitivity == 1.0
    assert target.specificity == pytest.approx(2 / 3)


def test_calling_nothing_positive_stays_above_every_vote_ratio() -> None:
    # With every score at 1 the best point is the "nothing positive" row, whose
    # threshold lies just above 1; the ratio rule has to accept it.
    choice = select_threshold([0, 1, 1], [1.0, 1.0, 1.0], method="target_specificity", target=1.0)
    assert choice.threshold > 1.0
    assert choice.specificity == 1.0

    proba = np.array([[0.0, 1.0]] * 4)
    result = aggregate_frames(proba, [2, 2], mode="ratio_threshold", threshold=choice.threshold)
    assert result.pred.tolist() == [0, 0]


def test_a_sensitivity_target_is_met_when_it_can_be() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    choice = select_threshold(y, score, method="target_sensitivity", target=1.0)
    assert choice.sensitivity == 1.0
    assert choice.threshold <= 0.4


def test_a_specificity_target_is_met_when_it_can_be() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    choice = select_threshold(y, score, method="target_specificity", target=1.0)
    assert choice.specificity == 1.0


def test_an_unreachable_target_falls_back_to_the_best_point() -> None:
    y = np.array([0, 1])
    score = np.array([0.5, 0.5])  # no threshold separates them
    choice = select_threshold(y, score, method="target_sensitivity", target=1.0)
    assert np.isfinite(choice.threshold)


def test_a_fixed_threshold_is_scored_as_given() -> None:
    choice = select_threshold([0, 1], [0.2, 0.9], method="fixed", fixed=0.5)
    assert choice.threshold == 0.5
    assert choice.sensitivity == 1.0
    assert choice.specificity == 1.0
    assert choice.method == "fixed"


def test_threshold_selection_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        select_threshold([0, 1], [0.1, 0.9], method="guess")
    with pytest.raises(ValueError, match="needs both classes"):
        select_threshold([1, 1], [0.1, 0.9])
    with pytest.raises(ValueError, match="requires the threshold"):
        select_threshold([0, 1], [0.1, 0.9], method="fixed")
    with pytest.raises(ValueError, match="requires a target"):
        select_threshold([0, 1], [0.1, 0.9], method="target_sensitivity")
    with pytest.raises(ValueError, match="but score has"):
        select_threshold([0, 1], [0.1])


def test_the_roc_table_replaces_the_infinite_first_threshold() -> None:
    table = roc_table([0, 1], [0.2, 0.9])
    assert np.all(np.isfinite(table["threshold"].to_numpy()))
    assert table["threshold"].iloc[0] == np.nextafter(0.9, np.inf)


def test_the_sweep_is_monotone_in_the_expected_directions() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 30)
    score = np.clip(y + rng.normal(0, 0.3, 60), 0, 1)
    sweep = threshold_sweep(y, score)
    assert np.all(np.diff(sweep["sensitivity"].to_numpy()) <= 1e-12)
    assert np.all(np.diff(sweep["specificity"].to_numpy()) >= -1e-12)
    assert np.all(np.diff(sweep["n_positive"].to_numpy()) <= 0)


def test_a_threshold_choice_serialises_to_json() -> None:
    choice = ThresholdChoice(0.5, 0.9, 0.8, 0.7, 0.95, "youden", target=0.9)
    payload = choice.to_dict()
    assert payload["threshold"] == 0.5
    assert payload["method"] == "youden"
    assert json.dumps(payload)


# -- metrics ------------------------------------------------------------
def test_uar_matches_the_balanced_accuracy_of_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200)
    y_pred = rng.integers(0, 4, 200)
    report = evaluate(y_true, y_pred, labels=[0, 1, 2, 3])
    assert report["uar"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_uar_averages_only_the_classes_that_occur() -> None:
    # Class 2 is only ever predicted: it costs class 0 half its recall, and it
    # is listed with a recall of 0 without entering the mean.
    with pytest.warns(UserWarning):
        expected = balanced_accuracy_score([0, 0, 1, 1], [0, 2, 1, 1])
    report = evaluate([0, 0, 1, 1], [0, 2, 1, 1], labels=[0, 1, 2])
    assert report["uar"] == pytest.approx(expected) == pytest.approx(0.75)
    assert report["per_class_recall"] == {0: 0.5, 1: 1.0, 2: 0.0}


def test_several_classes_can_share_the_positive_side() -> None:
    # Four-way labels, pathological (1, 2, 3) against normal (0).
    report = evaluate([0, 0, 1, 2, 3, 3], [0, 1, 1, 3, 0, 2], positive_label=(1, 2, 3))
    assert report["sensitivity"] == pytest.approx(3 / 4)
    assert report["specificity"] == pytest.approx(1 / 2)
    assert report["ppv"] == pytest.approx(3 / 4)


def test_screening_metrics_can_be_left_undefined() -> None:
    report = evaluate([1, 2, 3], [1, 3, 3], labels=[1, 2, 3], positive_label=None)
    for key in ("sensitivity", "specificity", "ppv", "npv"):
        assert np.isnan(report[key])
    assert report["uar"] == pytest.approx(2 / 3)


def test_uar_ignores_the_class_imbalance_that_accuracy_rewards() -> None:
    y_true = np.array([0] * 90 + [1] * 10)
    majority = np.zeros(100, dtype=int)
    report = evaluate(y_true, majority, labels=[0, 1])
    assert report["accuracy"] == pytest.approx(0.9)
    assert report["uar"] == pytest.approx(0.5)


def test_predictive_values_condition_on_the_prediction() -> None:
    ppv, npv = predictive_values([1, 1, 0, 0], [1, 0, 0, 0])
    assert ppv == pytest.approx(1.0)
    assert npv == pytest.approx(2 / 3)


def test_predictive_values_are_undefined_without_both_predictions() -> None:
    ppv, npv = predictive_values([1, 0], [0, 0])
    assert np.isnan(ppv)
    assert npv == pytest.approx(0.5)


def test_predictive_values_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="but y_pred has"):
        predictive_values([1, 0], [1])
    with pytest.raises(ValueError, match="empty inputs"):
        predictive_values([], [])


def test_a_report_carries_every_documented_key() -> None:
    report = evaluate([0, 1, 1, 0], [0, 1, 0, 0], np.array([0.1, 0.9, 0.4, 0.2]))
    expected = {
        "level",
        "n",
        "labels",
        "accuracy",
        "uar",
        "macro_f1",
        "per_class_recall",
        "per_class_precision",
        "confusion_matrix",
        "support",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "roc_auc",
    }
    assert expected <= set(report)
    assert report["n"] == 4
    assert json.dumps(report, cls=NumpyJSONEncoder)


def test_the_roc_area_handles_binary_and_multiclass_scores() -> None:
    binary = evaluate([0, 1], [0, 1], np.array([[0.9, 0.1], [0.2, 0.8]]), labels=[0, 1])
    assert binary["roc_auc"] == pytest.approx(1.0)

    proba = np.eye(3)[[0, 1, 2]]
    multi = evaluate([0, 1, 2], [0, 1, 2], proba, labels=[0, 1, 2])
    assert multi["roc_auc"] == pytest.approx(1.0)

    assert np.isnan(evaluate([0, 1], [0, 1])["roc_auc"])


def test_evaluation_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="but y_pred has"):
        evaluate([0, 1], [0])
    with pytest.raises(ValueError, match="empty predictions"):
        evaluate([], [])


def test_folds_are_summarised_with_mean_and_spread() -> None:
    folds = [{"uar": 0.6, "accuracy": 0.7}, {"uar": 0.8, "accuracy": 0.9}]
    summary = summarise_folds(folds, keys=["uar", "accuracy", "ppv"])
    assert summary["uar"]["mean"] == pytest.approx(0.7)
    assert summary["uar"]["std"] == pytest.approx(0.1)
    assert summary["uar"]["n"] == 2
    assert np.isnan(summary["ppv"]["mean"])
    assert summary["ppv"]["n"] == 0


def test_undefined_entries_are_skipped_rather_than_poisoning_the_mean() -> None:
    folds = [{"uar": 0.6}, {"uar": float("nan")}, {"uar": 0.8}]
    summary = summarise_folds(folds, keys=["uar"])
    assert summary["uar"]["mean"] == pytest.approx(0.7)
    assert summary["uar"]["n"] == 2


def test_summarising_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty list of folds"):
        summarise_folds([])


def test_reports_render_as_tables() -> None:
    folds = [{"fold": 0, "level": "subject", "n": 10, "uar": 0.5}]
    table = folds_to_frame(folds, keys=["uar"])
    assert table.loc[0, "uar"] == 0.5

    report = evaluate([0, 1], [0, 1], labels=[0, 1])
    matrix = confusion_frame(report, names={0: "Normal", 1: "Pathological"})
    assert list(matrix.index) == ["Normal", "Pathological"]


# -- reports ------------------------------------------------------------
def test_a_run_directory_holds_config_metrics_and_report(tmp_path) -> None:
    payload = {"task": "demo", "subject_summary": {"uar": {"mean": 0.6, "std": 0.1, "n": 2}}}
    run_dir = save_run(
        tmp_path, Config(), payload, run_id="r1", tables={"predictions": pd.DataFrame({"a": [1]})}
    )
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "predictions.csv").is_file()

    restored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert restored["task"] == "demo"
    assert Config.from_yaml(run_dir / "config.yaml").seed == 0


def test_the_encoder_understands_numpy_and_paths(tmp_path) -> None:
    payload = {
        "i": np.int64(3),
        "f": np.float64(0.5),
        "b": np.bool_(True),
        "a": np.arange(3),
        "p": tmp_path,
    }
    restored = json.loads(json.dumps(payload, cls=NumpyJSONEncoder))
    assert restored["i"] == 3
    assert restored["a"] == [0, 1, 2]
    assert restored["b"] is True


def test_markdown_renders_the_sections_it_is_given() -> None:
    text = render_markdown(
        {
            "task": "detection",
            "labels": [0, 1],
            "subject_summary": {"uar": {"mean": 0.62, "std": 0.03, "n": 5}},
            "pooled_subject": {
                "n": 200,
                "uar": 0.61,
                "accuracy": 0.9,
                "confusion_matrix": [[40, 10], [5, 145]],
            },
            "thresholds": [0.4, 0.5],
        },
        title="Detection",
    )
    assert "# Detection" in text
    assert "0.6200" in text
    assert "Confusion matrix" in text
    assert "Decision threshold" in text


def test_a_run_identifier_is_time_stamped() -> None:
    assert new_run_id("x").startswith("x-")
    assert len(new_run_id()) > len("run-")


def test_runs_started_in_the_same_second_get_their_own_folders(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timezone

    from voice_pathology_detection.eval import report

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 10, 8, 10, 15, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(report, "datetime", FrozenClock)
    first = new_run_id("demo", tmp_path)
    second = new_run_id("demo", tmp_path)
    assert first == "demo-20261008-101500"
    assert second == "demo-20261008-101500-2"
    assert (tmp_path / first).is_dir() and (tmp_path / second).is_dir()


def test_json_is_written_with_directories_created(tmp_path) -> None:
    path = write_json(tmp_path / "deep" / "m.json", {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_undefined_numbers_are_written_as_null(tmp_path) -> None:
    payload = {
        "mean": float("nan"),
        "values": [np.float64(np.inf), 0.5],
        "nested": {"x": np.nan, "array": np.array([1.0, np.nan])},
    }
    text = write_json(tmp_path / "m.json", payload).read_text(encoding="utf-8")

    def refuse(token: str) -> float:
        raise AssertionError(f"{token} is not valid JSON")

    restored = json.loads(text, parse_constant=refuse)
    assert restored == {
        "mean": None,
        "values": [None, 0.5],
        "nested": {"x": None, "array": [1.0, None]},
    }


# -- figures ------------------------------------------------------------
def test_every_figure_is_written(tmp_path, synthetic_corpus) -> None:
    report = evaluate([0, 1, 1, 2], [0, 1, 2, 2], labels=[0, 1, 2])
    assert plot_confusion_matrix(report, tmp_path / "cm.png").is_file()
    assert plot_confusion_matrix(report, tmp_path / "cm_raw.png", normalize=False).is_file()

    sweep = threshold_sweep([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert plot_threshold_curve(sweep, tmp_path / "th.png", chosen=0.5).is_file()

    table = folds_to_frame([{"fold": 0, "uar": 0.6}, {"fold": 1, "uar": 0.7}], keys=["uar"])
    assert plot_fold_errorbar(table, tmp_path / "bars.png", metrics=("uar",)).is_file()

    assert plot_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], tmp_path / "roc.png").is_file()

    recordings = sorted((synthetic_corpus / "train").rglob("*.wav"))[:2]
    out = plot_spectrogram_grid(
        recordings, tmp_path / "spec.png", n_fft=256, noverlap=200, max_freq=3500.0
    )
    assert out.is_file()


def test_spectrogram_frames_sit_at_the_middle_of_their_window(tmp_path, monkeypatch) -> None:
    from matplotlib.axes import Axes
    from scipy.io import wavfile

    drawn: list[tuple[np.ndarray, np.ndarray]] = []
    original = Axes.pcolormesh

    def record(self, times, frequencies, values, **kwargs):
        drawn.append((np.asarray(times), np.asarray(values)))
        return original(self, times, frequencies, values, **kwargs)

    monkeypatch.setattr(Axes, "pcolormesh", record)
    sr = 8000
    tone = tmp_path / "tone.wav"
    wavfile.write(tone, sr, (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32))
    silence = tmp_path / "silence.wav"
    wavfile.write(silence, sr, np.zeros(sr, dtype=np.float32))

    plot_spectrogram_grid([tone, silence], tmp_path / "s.png", n_fft=256, noverlap=128)
    times, _ = drawn[0]
    # Frame t spans samples [128 t, 128 t + 256), so its middle is at 128 t + 128.
    assert times[:3] == pytest.approx(np.array([128, 256, 384]) / sr)
    # A silent recording sits at the floor of the dynamic range, not at the peak.
    assert np.all(drawn[1][1] == -80.0)


def test_figures_report_what_they_cannot_draw(tmp_path) -> None:
    with pytest.raises(KeyError, match="lacks the column"):
        plot_threshold_curve(pd.DataFrame({"threshold": [0.5]}), tmp_path / "x.png")
    with pytest.raises(KeyError, match="none of the metrics"):
        plot_fold_errorbar(pd.DataFrame({"fold": [0]}), tmp_path / "y.png", metrics=("uar",))
    with pytest.raises(ValueError, match="both classes"):
        plot_roc([1, 1], [0.2, 0.9], tmp_path / "z.png")
    with pytest.raises(ValueError, match="at least one recording"):
        plot_spectrogram_grid([], tmp_path / "w.png")
    with pytest.raises(ValueError, match="must be smaller than"):
        plot_spectrogram_grid(["a.wav"], tmp_path / "v.png", n_fft=256, noverlap=256)
