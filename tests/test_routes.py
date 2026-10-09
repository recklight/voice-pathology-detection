"""The feature set, the stage loop and the four training routes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voice_pathology_detection.config import Config, ModelConfig
from voice_pathology_detection.data.synthetic import make_frame_dataset
from voice_pathology_detection.features.pipeline import FeatureSet, extract_corpus, extract_file
from voice_pathology_detection.routes.cascade import run_cascade
from voice_pathology_detection.routes.detection import binary_view, fit_detection, run_detection
from voice_pathology_detection.routes.disorder import (
    fit_disorder,
    pathological_subset,
    run_disorder,
)
from voice_pathology_detection.routes.flat import fit_flat, run_flat
from voice_pathology_detection.routes.stage import (
    StageError,
    StageModel,
    fit_stage,
    frame_posteriors,
    run_stage,
)


# -- the feature set ----------------------------------------------------
def test_the_feature_set_reports_consistent_bookkeeping(feature_set: FeatureSet) -> None:
    assert feature_set.n_frames == feature_set.X.shape[0]
    assert feature_set.group_sizes.sum() == feature_set.n_frames
    assert feature_set.frame_labels.size == feature_set.n_frames
    assert feature_set.frame_subjects.size == feature_set.n_frames
    assert feature_set.file_labels.size == feature_set.n_files


def test_frames_of_one_recording_can_be_recovered(feature_set: FeatureSet) -> None:
    frames = feature_set.frames_of(2)
    assert frames.shape[0] == feature_set.group_sizes[2]
    assert np.array_equal(frames, feature_set.X[feature_set.frame_file == 2])


def test_selecting_files_renumbers_the_frame_map(feature_set: FeatureSet) -> None:
    subset = feature_set.select([1, 3, 5])
    assert subset.n_files == 3
    assert subset.frame_file.min() == 0
    assert subset.frame_file.max() == 2
    assert subset.group_sizes.sum() == subset.n_frames
    assert np.array_equal(subset.frames_of(0), feature_set.frames_of(1))


def test_a_boolean_mask_selects_the_same_recordings(feature_set: FeatureSet) -> None:
    mask = np.zeros(feature_set.n_files, dtype=bool)
    mask[[0, 4]] = True
    assert np.array_equal(feature_set.select(mask).X, feature_set.select([0, 4]).X)


def test_selection_keeps_the_original_file_order(feature_set: FeatureSet) -> None:
    # Positions given out of order must not scramble the frame grouping.
    shuffled = feature_set.select([5, 1, 3])
    ordered = feature_set.select([1, 3, 5])
    assert shuffled.files["subject_id"].tolist() == ordered.files["subject_id"].tolist()
    assert np.array_equal(shuffled.X, ordered.X)


def test_selection_validates_its_indices(feature_set: FeatureSet) -> None:
    with pytest.raises(IndexError, match="out of range"):
        feature_set.select([feature_set.n_files])
    with pytest.raises(IndexError, match="boolean mask of length"):
        feature_set.select(np.ones(3, dtype=bool))
    with pytest.raises(IndexError, match="outside an index"):
        feature_set.frames_of(feature_set.n_files)


def test_the_frame_matrix_can_be_replaced_in_place(feature_set: FeatureSet) -> None:
    replaced = feature_set.with_features(np.zeros((feature_set.n_frames, 3)))
    assert replaced.n_dim == 3
    assert np.array_equal(replaced.frame_file, feature_set.frame_file)
    with pytest.raises(ValueError, match="expected"):
        feature_set.with_features(np.zeros((2, 3)))


def test_a_non_monotonic_frame_map_is_refused() -> None:
    files = pd.DataFrame({"subject_id": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError, match="not monotonic"):
        FeatureSet(np.zeros((3, 2)), np.array([1, 0, 1]), files)


def test_a_mismatched_frame_map_is_refused() -> None:
    files = pd.DataFrame({"subject_id": ["a"], "label": [0]})
    with pytest.raises(ValueError, match="frame_file has"):
        FeatureSet(np.zeros((3, 2)), np.array([0, 0]), files)
    with pytest.raises(ValueError, match="index holds only"):
        FeatureSet(np.zeros((2, 2)), np.array([0, 5]), files)


def test_frame_counts_that_contradict_the_frame_map_are_refused() -> None:
    files = pd.DataFrame({"subject_id": ["a", "b"], "label": [0, 1], "n_frames": [3, 2]})
    with pytest.raises(ValueError, match="disagrees with frame_file"):
        FeatureSet(np.zeros((5, 2)), np.array([0, 0, 1, 1, 1]), files)


def test_a_feature_set_survives_a_round_trip(feature_set: FeatureSet, tmp_path) -> None:
    path = feature_set.save(tmp_path / "features.npz")
    restored = FeatureSet.load(path)
    assert np.allclose(restored.X, feature_set.X)
    assert restored.files["subject_id"].tolist() == feature_set.files["subject_id"].tolist()


def test_numeric_looking_identifiers_survive_a_round_trip(tmp_path) -> None:
    files = pd.DataFrame({"subject_id": ["001", "002", "010"], "label": [0, 1, 0]})
    path = FeatureSet(np.zeros((3, 2)), np.arange(3), files).save(tmp_path / "f.npz")
    assert FeatureSet.load(path).files["subject_id"].tolist() == ["001", "002", "010"]


def test_loading_reports_what_is_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no such feature archive"):
        FeatureSet.load(tmp_path / "absent.npz")


# -- extraction ---------------------------------------------------------
def test_a_corpus_is_extracted_and_cached(corpus_config: Config) -> None:
    from voice_pathology_detection.data.dataset import build_index

    index = build_index(corpus_config)
    first = extract_corpus(index, corpus_config, cache_dir=corpus_config.paths.cache_dir)
    assert first.n_files == 12
    assert first.n_dim == corpus_config.mfcc.n_dim
    assert first.group_sizes.min() > 0

    cached = list(corpus_config.paths.cache_dir.glob("*.npy"))
    assert len(cached) == 12

    second = extract_corpus(index, corpus_config, cache_dir=corpus_config.paths.cache_dir)
    assert np.allclose(first.X, second.X)


def test_a_single_recording_can_be_extracted(corpus_config: Config) -> None:
    path = next((corpus_config.paths.train_root / "Normal").glob("*.wav"))
    features = extract_file(path, corpus_config.mfcc, corpus_config.trim)
    assert features.shape[1] == corpus_config.mfcc.n_dim


def test_unreadable_recordings_are_skipped_or_raised(corpus_config: Config, tmp_path) -> None:
    from voice_pathology_detection.data.dataset import build_index

    index = build_index(corpus_config)
    broken = index.copy()
    broken.loc[0, "path"] = str(tmp_path / "missing.wav")

    skipped = extract_corpus(broken, corpus_config, skip_errors=True)
    assert skipped.n_files == 11

    with pytest.raises(FileNotFoundError):
        extract_corpus(broken, corpus_config, skip_errors=False)


def test_extraction_validates_the_index(corpus_config: Config) -> None:
    with pytest.raises(ValueError, match="needs a 'path' column"):
        extract_corpus(pd.DataFrame({"label": [0]}), corpus_config)
    with pytest.raises(ValueError, match="empty index"):
        extract_corpus(pd.DataFrame({"path": [], "label": []}), corpus_config)


def test_a_corpus_of_only_unreadable_files_is_reported(corpus_config: Config, tmp_path) -> None:
    index = pd.DataFrame(
        {
            "path": [str(tmp_path / "a.wav")],
            "subject_id": ["a"],
            "label": [0],
            "file_name": ["a.wav"],
        }
    )
    with pytest.raises(ValueError, match="none of the"):
        extract_corpus(index, corpus_config)


# -- posteriors ---------------------------------------------------------
def test_posteriors_are_aligned_to_the_requested_label_order() -> None:
    from voice_pathology_detection.models.registry import build_model

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (40, 3)), rng.normal(2, 0.5, (40, 3))])
    y = np.array([1] * 40 + [3] * 40)
    model = build_model("gmm", {"n_components": 1}).fit(X, y)

    proba = frame_posteriors(model, X, (1, 2, 3))
    assert proba.shape == (80, 3)
    # Class 2 never occurred in training, so its column must be empty.
    assert np.allclose(proba[:, 1], 0.0)
    assert proba[:40, 0].mean() > proba[:40, 2].mean()


def test_an_estimator_without_posteriors_falls_back_to_its_decision_function() -> None:
    from voice_pathology_detection.models.registry import build_model

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (30, 3)), rng.normal(2, 0.5, (30, 3))])
    y = np.array([0] * 30 + [1] * 30)
    model = build_model("linear_svm").fit(X, y)

    proba = frame_posteriors(model, X, (0, 1))
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# -- the stage loop -----------------------------------------------------
def test_a_stage_scores_every_subject_exactly_once(
    feature_set: FeatureSet, small_config: Config
) -> None:
    result = run_stage(
        binary_view(feature_set),
        small_config,
        task="detection",
        labels=(0, 1),
        model=ModelConfig(name="gmm", params={"n_components": 1}),
        tune_threshold=True,
    )
    predictions = result.predictions
    assert len(predictions) == feature_set.n_files
    assert predictions["subject_id"].is_unique
    assert set(predictions["fold"]) == set(range(small_config.split.n_splits))


def test_a_stage_reports_both_levels_and_every_fold(
    feature_set: FeatureSet, small_config: Config
) -> None:
    result = run_stage(
        binary_view(feature_set),
        small_config,
        task="detection",
        labels=(0, 1),
        model=ModelConfig(name="gmm", params={"n_components": 1}),
        tune_threshold=True,
    )
    assert len(result.folds) == small_config.split.n_splits
    assert result.pooled_frame["level"] == "frame"
    assert result.pooled_subject["level"] == "subject"
    assert 0.0 <= result.subject_summary["uar"]["mean"] <= 1.0
    assert 0.0 <= result.frame_summary["uar"]["mean"] <= 1.0
    assert len(result.thresholds) == small_config.split.n_splits
    assert all(0.0 <= t <= 1.0 for t in result.thresholds)
    assert result.to_dict()["task"] == "detection"


def test_the_threshold_comes_from_held_out_data_only(
    feature_set: FeatureSet, small_config: Config
) -> None:
    result = run_stage(
        binary_view(feature_set),
        small_config,
        task="detection",
        labels=(0, 1),
        model=ModelConfig(name="gmm", params={"n_components": 1}),
        tune_threshold=True,
        threshold_method="youden",
    )
    for outcome in result.folds:
        assert outcome.threshold is not None
        assert outcome.threshold.method == "youden"
        assert np.isfinite(outcome.threshold.auc)


@pytest.fixture
def overlapping_set() -> FeatureSet:
    """Four classes that overlap, so every fold makes a different set of mistakes."""
    X, frame_file, files = make_frame_dataset(7, 9, 6, separation=0.3, seed=3)
    return FeatureSet(X, frame_file, files)


def test_frame_metrics_are_pooled_over_every_evaluated_frame(
    overlapping_set: FeatureSet, small_config: Config
) -> None:
    result = run_detection(overlapping_set, small_config.override({"split.n_splits": 3}))
    pooled = result.pooled_frame
    matrix = np.asarray(pooled["confusion_matrix"], dtype=float)

    assert pooled["n"] == overlapping_set.n_frames == matrix.sum()
    assert (
        matrix.tolist()
        == np.sum(
            [outcome.frame_metrics["confusion_matrix"] for outcome in result.folds], axis=0
        ).tolist()
    )
    # The scalars belong to that matrix, not to an average of the folds.
    assert pooled["uar"] == pytest.approx(np.mean(np.diag(matrix) / matrix.sum(axis=1)))
    assert pooled["sensitivity"] == pytest.approx(matrix[1, 1] / matrix[1].sum())


def test_a_fixed_threshold_is_used_verbatim(feature_set: FeatureSet, small_config: Config) -> None:
    result = run_stage(
        binary_view(feature_set),
        small_config,
        task="detection",
        labels=(0, 1),
        model=ModelConfig(name="gmm", params={"n_components": 1}),
        tune_threshold=True,
        threshold_method="fixed",
        fixed_threshold=0.42,
    )
    assert result.thresholds == [0.42] * small_config.split.n_splits


def test_a_single_class_feature_set_is_refused(small_config: Config) -> None:
    X, frame_file, files = make_frame_dataset(4, 5, 6, labels=(0,), seed=0)
    with pytest.raises(StageError, match="at least two classes"):
        run_stage(
            FeatureSet(X, frame_file, files),
            small_config,
            task="detection",
            labels=(0, 1),
            model=ModelConfig(name="gmm"),
        )


def test_threshold_tuning_needs_a_two_class_stage(
    feature_set: FeatureSet, small_config: Config
) -> None:
    with pytest.raises(StageError, match="only defined for two classes"):
        run_stage(
            feature_set,
            small_config,
            task="flat",
            labels=(0, 1, 2, 3),
            model=ModelConfig(name="gmm"),
            tune_threshold=True,
        )


# -- the routes ---------------------------------------------------------
def test_detection_collapses_the_four_classes(
    feature_set: FeatureSet, small_config: Config
) -> None:
    binary = binary_view(feature_set)
    assert sorted(set(binary.file_labels.tolist())) == [0, 1]
    assert binary.n_frames == feature_set.n_frames

    result = run_detection(feature_set, small_config)
    assert result.labels == (0, 1)
    assert set(result.predictions["y_pred"].unique()) <= {0, 1}


def test_the_disorder_stage_drops_the_normal_recordings(
    feature_set: FeatureSet, small_config: Config
) -> None:
    subset = pathological_subset(feature_set)
    assert sorted(set(subset.file_labels.tolist())) == [1, 2, 3]

    result = run_disorder(feature_set, small_config)
    assert result.labels == (1, 2, 3)
    assert set(result.predictions["y_true"].unique()) <= {1, 2, 3}


def test_the_disorder_stage_needs_pathological_material(small_config: Config) -> None:
    X, frame_file, files = make_frame_dataset(4, 5, 6, labels=(0,), seed=0)
    with pytest.raises(StageError, match="no pathological recording"):
        pathological_subset(FeatureSet(X, frame_file, files))

    X, frame_file, files = make_frame_dataset(4, 5, 6, labels=(0, 1), seed=0)
    with pytest.raises(StageError, match="at least two classes"):
        pathological_subset(FeatureSet(X, frame_file, files))


def test_the_single_stage_route_predicts_all_four_classes(
    feature_set: FeatureSet, small_config: Config
) -> None:
    result = run_flat(feature_set, small_config)
    assert result.labels == (0, 1, 2, 3)
    assert len(result.predictions) == feature_set.n_files
    assert np.asarray(result.pooled_subject["confusion_matrix"]).shape == (4, 4)


def test_four_way_sensitivity_is_the_recall_of_pathological_voices(
    overlapping_set: FeatureSet, small_config: Config
) -> None:
    result = run_flat(overlapping_set, small_config)
    truth = result.predictions["y_true"].to_numpy() != 0
    called = result.predictions["y_pred"].to_numpy() != 0
    assert result.pooled_subject["sensitivity"] == pytest.approx(called[truth].mean())
    assert result.pooled_subject["specificity"] == pytest.approx((~called[~truth]).mean())


def test_the_disorder_stage_reports_no_screening_metrics(
    feature_set: FeatureSet, small_config: Config
) -> None:
    result = run_disorder(feature_set, small_config)
    assert np.isnan(result.pooled_subject["sensitivity"])
    assert result.subject_summary["specificity"]["n"] == 0


def test_the_single_stage_route_takes_its_model_from_its_own_section(
    feature_set: FeatureSet, small_config: Config
) -> None:
    # A model name no registry knows: the flat route must never look it up.
    cfg = small_config.override({"disorder.model.name": "absent"})
    assert run_flat(feature_set, cfg).labels == (0, 1, 2, 3)


def test_the_cascade_merges_both_stages(feature_set: FeatureSet, small_config: Config) -> None:
    result = run_cascade(feature_set, small_config)
    predictions = result.predictions

    assert len(predictions) == feature_set.n_files
    assert predictions["subject_id"].is_unique
    assert set(predictions["y_pred"].unique()) <= {0, 1, 2, 3}
    # A subject the detector cleared is always Normal; a forwarded one never is.
    cleared = ~predictions["forwarded"].to_numpy()
    assert np.all(predictions["y_pred"].to_numpy()[cleared] == 0)
    assert np.all(predictions["y_pred"].to_numpy()[~cleared] != 0)

    assert len(result.thresholds) == small_config.split.n_splits
    assert len(result.forwarded_per_fold) == small_config.split.n_splits
    assert np.asarray(result.pooled_overall["confusion_matrix"]).shape == (4, 4)
    assert result.to_dict()["task"] == "cascade"

    # The merged decision is scored pathological against normal, the same way
    # as the detection stage.
    truth = predictions["y_true"].to_numpy() != 0
    assert result.pooled_overall["sensitivity"] == pytest.approx(
        predictions["forwarded"].to_numpy()[truth].mean()
    )
    assert result.pooled_overall["sensitivity"] == pytest.approx(
        result.pooled_detection["sensitivity"]
    )


def test_the_cascade_keeps_a_fixed_threshold_of_zero(
    feature_set: FeatureSet, small_config: Config
) -> None:
    cfg = small_config.override({"detection.threshold_method": "fixed", "detection.threshold": 0.0})
    result = run_cascade(feature_set, cfg)
    assert result.thresholds == [0.0] * cfg.split.n_splits
    assert sum(result.forwarded_per_fold) == feature_set.n_files


def test_the_cascade_separates_well_formed_synthetic_classes() -> None:
    X, frame_file, files = make_frame_dataset(
        n_files_per_class=8, n_frames=12, n_dim=10, separation=3.0, seed=0
    )
    cfg = Config().override(
        {
            "split.n_splits": 2,
            "split.val_fraction": 0.4,
            "normalization.method": "cmvn",
            "detection.model.name": "gmm",
            "detection.model.params": {"n_components": 1},
            "disorder.model.name": "gmm",
            "disorder.model.params": {"n_components": 1},
        }
    )
    result = run_cascade(FeatureSet(X, frame_file, files), cfg)
    assert result.overall_summary["uar"]["mean"] > 0.6
    assert result.detection_summary["sensitivity"]["mean"] > 0.6


# -- fitted stages ------------------------------------------------------
def test_a_fitted_stage_predicts_and_round_trips(
    feature_set: FeatureSet, small_config: Config, tmp_path
) -> None:
    model = fit_detection(feature_set, small_config)
    assert model.task == "detection"
    assert model.labels == (0, 1)
    assert model.threshold is not None

    aggregate = model.predict_subjects(feature_set)
    assert aggregate.pred.size == feature_set.n_files
    assert model.predict_frames(feature_set.X, group_sizes=feature_set.group_sizes).shape == (
        feature_set.n_frames,
        2,
    )

    path = model.save(tmp_path / "detection.joblib")
    restored = StageModel.load(path)
    assert np.array_equal(restored.predict_subjects(feature_set).pred, aggregate.pred)


def test_every_route_can_be_fitted_on_all_data(
    feature_set: FeatureSet, small_config: Config
) -> None:
    assert fit_disorder(feature_set, small_config).labels == (1, 2, 3)
    assert fit_flat(feature_set, small_config).labels == (0, 1, 2, 3)
    assert (
        fit_stage(
            feature_set,
            small_config,
            task="flat",
            labels=(0, 1, 2, 3),
            model=ModelConfig(name="gmm", params={"n_components": 1}),
        ).task
        == "flat"
    )


@pytest.mark.parametrize("aggregation", ["majority_vote", "ratio_threshold"])
def test_the_saved_threshold_is_chosen_on_the_score_it_is_applied_to(
    aggregation: str, small_config: Config, monkeypatch
) -> None:
    # With ten frames per recording a vote ratio is a multiple of 0.1; the
    # threshold of a counting rule has to be chosen on those, not on posteriors.
    from voice_pathology_detection.routes import stage

    seen: list[np.ndarray] = []
    choose = stage.select_threshold

    def recording(y, score, **kwargs):
        seen.append(np.asarray(score))
        return choose(y, score, **kwargs)

    monkeypatch.setattr(stage, "select_threshold", recording)
    X, frame_file, files = make_frame_dataset(8, 10, 6, separation=0.4, seed=1)
    fit_detection(
        FeatureSet(X, frame_file, files),
        small_config.override({"detection.aggregation": aggregation}),
    )
    assert len(seen) == 1
    assert np.allclose(seen[0] * 10, np.round(seen[0] * 10))


def test_a_fitted_detector_applies_its_threshold(
    overlapping_set: FeatureSet, small_config: Config
) -> None:
    model = fit_detection(overlapping_set, small_config)
    for threshold in (0.1, 0.5, 0.9):
        model.threshold = threshold
        aggregate = model.predict_subjects(binary_view(overlapping_set))
        assert np.array_equal(aggregate.pred, (aggregate.score(1) >= threshold).astype(int))


def test_ratio_aggregation_survives_a_threshold_above_every_ratio(small_config: Config) -> None:
    # A specificity of 1 is often only reachable by calling nobody positive;
    # that threshold sits just above the largest vote ratio, i.e. above 1 when
    # some subject had every frame voted pathological.  Which seeds get there
    # depends on how scikit-learn assigns the folds, so several are tried.
    cfg = small_config.override(
        {
            "detection.aggregation": "ratio_threshold",
            "detection.threshold_method": "target_specificity",
            "detection.target": 1.0,
        }
    )
    above_one = []
    for seed in range(6):
        X, frame_file, files = make_frame_dataset(8, 10, 6, separation=0.2, seed=seed)
        result = run_detection(FeatureSet(X, frame_file, files), cfg)
        assert len(result.predictions) == 32
        above_one.append(max(result.thresholds) > 1.0)
    assert any(above_one)


def test_loading_a_missing_model_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no such model file"):
        StageModel.load(tmp_path / "absent.joblib")


# -- end to end on audio ------------------------------------------------
def test_the_cascade_runs_on_synthesised_audio(corpus_config: Config) -> None:
    from voice_pathology_detection.data.dataset import build_index

    index = build_index(corpus_config)
    features = extract_corpus(index, corpus_config)
    result = run_cascade(features, corpus_config)

    assert len(result.predictions) == 12
    assert np.isfinite(result.overall_summary["uar"]["mean"])
    assert set(result.predictions["y_pred"].unique()) <= {0, 1, 2, 3}
