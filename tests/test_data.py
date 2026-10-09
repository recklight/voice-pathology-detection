"""Labels, trimming, corpus indexing, splits and synthetic material."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voice_pathology_detection.config import TrimConfig
from voice_pathology_detection.data.dataset import (
    DatasetError,
    attach_binary_labels,
    build_index,
    finalise_index,
    read_demographics,
    scan_class_directories,
    summarise_index,
)
from voice_pathology_detection.data.labels import (
    ALL_LABELS,
    CLASS_NAMES,
    BinaryClass,
    VoiceClass,
    check_class_counts,
    class_name,
    is_pathological,
    parse_label,
    to_binary,
)
from voice_pathology_detection.data.splits import (
    SplitError,
    check_no_group_leakage,
    subject_folds,
    train_val_split,
)
from voice_pathology_detection.data.synthetic import make_frame_dataset, synthesize_voice
from voice_pathology_detection.data.trim import trim_audio, trim_energy_vad, trim_fixed_edges

SR = 8000


# -- labels -------------------------------------------------------------
def test_the_four_classes_carry_the_expected_codes() -> None:
    assert VoiceClass.NORMAL == 0
    assert VoiceClass.NEOPLASM == 1
    assert VoiceClass.PHONOTRAUMA == 2
    assert VoiceClass.VOCAL_PALSY == 3
    assert ALL_LABELS == (0, 1, 2, 3)


def test_only_normal_stays_normal_in_the_binary_task() -> None:
    assert to_binary([0, 1, 2, 3]).tolist() == [0, 1, 1, 1]
    assert is_pathological([0, 3]).tolist() == [False, True]


def test_an_unknown_code_is_refused_by_the_binary_mapping() -> None:
    with pytest.raises(ValueError, match="unknown code"):
        to_binary([0, 9])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Normal", 0),
        ("neoplasm", 1),
        ("Vocal palsy", 3),
        ("vocal_palsy", 3),
        ("VOCAL-PALSY", 3),
        ("2", 2),
        (2, 2),
        (np.int64(1), 1),
        (np.float64(1.0), 1),
        (3.0, 3),
        ("2.0", 2),
    ],
)
def test_folder_names_and_codes_resolve_to_the_same_class(value: object, expected: int) -> None:
    assert parse_label(value) == expected


@pytest.mark.parametrize("value", ["tumour", 9, "", "-1", 1.5, float("nan")])
def test_an_unrecognised_class_is_refused(value: object) -> None:
    with pytest.raises(ValueError):
        parse_label(value)


def test_display_names_cover_both_tasks() -> None:
    assert class_name(2) == "Phonotrauma"
    assert class_name(int(to_binary([2])[0]), binary=True) == "Pathological"
    with pytest.raises(ValueError, match="unknown label"):
        class_name(2, binary=True)
    assert BinaryClass.PATHOLOGICAL == 1


def test_a_class_count_mismatch_can_be_fatal_or_advisory(caplog) -> None:
    counts = {0: 50, 1: 40, 2: 60, 3: 49}
    with pytest.raises(ValueError, match="Vocal palsy: found 49, expected 50"):
        check_class_counts(counts)
    check_class_counts(counts, strict=False)  # only warns
    check_class_counts({0: 50, 1: 40, 2: 60, 3: 50})


# -- trimming -----------------------------------------------------------
def test_fixed_trimming_removes_the_requested_edges() -> None:
    result = trim_fixed_edges(np.ones(SR * 6), SR, edge_seconds=2.0, min_seconds=4.5)
    assert result.applied
    assert (result.start, result.end) == (2 * SR, 4 * SR)
    assert result.n_samples == 2 * SR
    assert result.duration(SR) == pytest.approx(2.0)


def test_a_short_recording_is_left_alone() -> None:
    result = trim_fixed_edges(np.ones(SR), SR, edge_seconds=2.0, min_seconds=4.5)
    assert not result.applied
    assert result.n_samples == SR


def test_trimming_that_would_empty_a_recording_is_refused() -> None:
    result = trim_fixed_edges(np.ones(SR * 5), SR, edge_seconds=3.0, min_seconds=1.0)
    assert not result.applied


def test_the_energy_detector_finds_a_tone_inside_silence() -> None:
    tone = np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    padded = np.concatenate([np.zeros(SR), tone, np.zeros(SR)])
    result = trim_energy_vad(padded, SR)
    assert result.applied
    assert result.start > SR // 2
    assert result.end < padded.size - SR // 2


def test_the_mode_none_returns_the_signal_untouched() -> None:
    result = trim_audio(np.ones(100), SR, TrimConfig(mode="none"))
    assert not result.applied
    assert result.n_samples == 100


def test_a_trim_leaving_too_little_audio_is_rolled_back() -> None:
    cfg = TrimConfig(mode="fixed_edges", edge_seconds=2.4, min_seconds=1.0, min_output_seconds=1.0)
    result = trim_audio(np.ones(SR * 5), SR, cfg)
    assert not result.applied
    assert result.n_samples == SR * 5


def test_trimming_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="sr must be positive"):
        trim_audio(np.ones(10), 0)
    with pytest.raises(ValueError, match="empty waveform"):
        trim_audio(np.zeros(0), SR)


# -- synthetic material -------------------------------------------------
@pytest.mark.parametrize("label", list(CLASS_NAMES))
def test_every_class_synthesises_a_bounded_waveform(label: int) -> None:
    audio = synthesize_voice(label, sr=SR, duration=0.4, rng=np.random.default_rng(0))
    assert audio.shape == (int(0.4 * SR),)
    assert np.all(np.isfinite(audio))
    assert np.abs(audio).max() <= 1.0


def test_synthesis_is_reproducible_given_a_seed() -> None:
    first = synthesize_voice(1, sr=SR, duration=0.3, rng=np.random.default_rng(3))
    second = synthesize_voice(1, sr=SR, duration=0.3, rng=np.random.default_rng(3))
    assert np.array_equal(first, second)


def test_synthesis_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="sr must be positive"):
        synthesize_voice(0, sr=0)
    with pytest.raises(ValueError, match="duration must be positive"):
        synthesize_voice(0, sr=SR, duration=0.0)


def test_the_frame_generator_reports_consistent_bookkeeping() -> None:
    X, frame_file, files = make_frame_dataset(3, 5, 7, labels=(0, 2), seed=0)
    assert X.shape == (2 * 3 * 5, 7)
    assert files.shape[0] == 6
    assert frame_file.max() == 5
    assert files["n_frames"].sum() == X.shape[0]
    assert sorted(files["label"].unique().tolist()) == [0, 2]


# -- corpus indexing ----------------------------------------------------
def test_a_class_folder_layout_is_indexed(synthetic_corpus) -> None:
    index = scan_class_directories(synthetic_corpus / "train")
    assert len(index) == 12
    assert index["label"].value_counts().to_dict() == {0: 3, 1: 3, 2: 3, 3: 3}
    assert index["subject_id"].is_unique
    assert set(index["label_name"]) == set(CLASS_NAMES.values())


def test_indexing_a_missing_directory_explains_what_to_do(tmp_path) -> None:
    with pytest.raises(DatasetError, match=r"Point paths\.data_root"):
        scan_class_directories(tmp_path / "absent")


def test_an_empty_directory_is_reported(tmp_path) -> None:
    (tmp_path / "Normal").mkdir()
    with pytest.raises(DatasetError, match="contains no"):
        scan_class_directories(tmp_path)


def test_files_outside_a_diagnosis_folder_are_reported(tmp_path, synthetic_corpus) -> None:
    stray = synthetic_corpus / "train" / "unsorted"
    stray.mkdir()
    (stray / "x.wav").write_bytes((synthetic_corpus / "train" / "Normal" / "001.wav").read_bytes())
    with pytest.raises(DatasetError, match="not inside a recognised diagnosis folder"):
        scan_class_directories(synthetic_corpus / "train")


def test_unlabelled_material_can_be_indexed(synthetic_corpus) -> None:
    index = scan_class_directories(
        synthetic_corpus / "train" / "Normal", require_labels=False, split="test"
    )
    assert (index["label"] == -1).all()
    assert (index["label_name"] == "Unknown").all()
    assert (index["split"] == "test").all()


def test_duplicate_subject_identifiers_are_refused() -> None:
    frame = pd.DataFrame({"path": ["a.wav", "b.wav"], "subject_id": ["s", "s"], "label": [0, 1]})
    with pytest.raises(DatasetError, match="must be unique"):
        finalise_index(frame)


def test_an_index_without_a_label_column_is_refused() -> None:
    with pytest.raises(DatasetError, match="required column"):
        finalise_index(pd.DataFrame({"path": ["a.wav"]}))


def test_binary_labels_are_attached_and_unknowns_preserved() -> None:
    frame = pd.DataFrame({"path": ["a", "b"], "subject_id": ["a", "b"], "label": [-1, 2]})
    out = attach_binary_labels(frame)
    assert out["label_binary"].tolist() == [-1, 1]
    assert out["label_name"].tolist() == ["Unknown", "Phonotrauma"]


def test_demographics_are_read_and_applied(tmp_path, synthetic_corpus) -> None:
    flat = tmp_path / "flat" / "train"
    flat.mkdir(parents=True)
    names = []
    for source in sorted((synthetic_corpus / "train").rglob("*.wav")):
        target = flat / f"{source.parent.name}_{source.name}"
        target.write_bytes(source.read_bytes())
        names.append(target.name)

    table = tmp_path / "flat" / "meta.csv"
    rows = [
        {"File name": name, "Diagnosis(0-3)": parse_label(name.split("_")[0])} for name in names
    ]
    pd.DataFrame(rows).to_csv(table, index=False)

    demographics = read_demographics(table)
    assert set(demographics.columns) >= {"file_name", "label"}

    from voice_pathology_detection.config import Config

    cfg = Config().override(
        {"paths.data_root": str(tmp_path / "flat"), "paths.demographics": "meta.csv"}
    )
    index = build_index(cfg)
    assert len(index) == 12
    assert index["label"].value_counts().to_dict() == {0: 3, 1: 3, 2: 3, 3: 3}


def test_a_demographics_table_missing_a_recording_is_reported(tmp_path, synthetic_corpus) -> None:
    from voice_pathology_detection.config import Config

    table = synthetic_corpus / "train" / "meta.csv"
    pd.DataFrame([{"File name": "nobody.wav", "Diagnosis": 0}]).to_csv(table, index=False)
    cfg = Config().override(
        {"paths.data_root": str(synthetic_corpus), "paths.demographics": "train/meta.csv"}
    )
    with pytest.raises(DatasetError, match="missing from"):
        build_index(cfg)


def _flat_corpus(tmp_path, synthetic_corpus, names: list[str]):
    """Copy the synthetic recordings into one flat folder under the given stems."""
    flat = tmp_path / "flat" / "train"
    flat.mkdir(parents=True)
    sources = sorted((synthetic_corpus / "train").rglob("*.wav"))
    labels = []
    for name, source in zip(names, sources, strict=True):
        (flat / f"{name}.wav").write_bytes(source.read_bytes())
        labels.append(parse_label(source.parent.name))
    return tmp_path / "flat", labels


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_zero_padded_file_names_in_a_table_keep_their_zeros(
    tmp_path, synthetic_corpus, suffix: str
) -> None:
    from voice_pathology_detection.config import Config

    names = [f"{i:03d}" for i in range(1, 13)]
    root, labels = _flat_corpus(tmp_path, synthetic_corpus, names)
    table = pd.DataFrame({"File name": names, "Diagnosis": labels})
    if suffix == ".csv":
        table.to_csv(root / f"Demographics{suffix}", index=False)
    else:
        table.to_excel(root / f"Demographics{suffix}", index=False)

    cfg = Config().override(
        {"paths.data_root": str(root), "paths.demographics": f"Demographics{suffix}"}
    )
    index = build_index(cfg, expected_counts={0: 3, 1: 3, 2: 3, 3: 3}, strict_counts=True)
    assert sorted(index["file_name"]) == [f"{name}.wav" for name in names]


def test_a_blank_diagnosis_is_reported_by_file_name(tmp_path) -> None:
    table = tmp_path / "meta.csv"
    pd.DataFrame({"File name": ["001", "002", "003"], "Diagnosis": [0, 2, None]}).to_csv(
        table, index=False
    )
    with pytest.raises(DatasetError, match=r"no diagnosis for 1 file.*003"):
        read_demographics(table)


def test_whole_number_diagnoses_written_as_floats_are_read(tmp_path) -> None:
    table = tmp_path / "meta.csv"
    pd.DataFrame({"File name": ["a", "b"], "Diagnosis": [0.0, 3.0]}).to_csv(table, index=False)
    assert read_demographics(table)["label"].tolist() == [0, 3]


def test_a_speaker_column_groups_the_recordings_of_one_person(tmp_path, synthetic_corpus) -> None:
    from voice_pathology_detection.config import Config

    names = [f"{i:03d}" for i in range(1, 13)]
    root, labels = _flat_corpus(tmp_path, synthetic_corpus, names)
    # 001 and 002 are the same person; 012 has no speaker in the table.
    speakers = ["p1", "p1", *[f"p{i}" for i in range(3, 13)]]
    speakers[-1] = None
    pd.DataFrame({"File name": names, "Diagnosis": labels, "Subject ID": speakers}).to_csv(
        root / "meta.csv", index=False
    )
    cfg = Config().override({"paths.data_root": str(root), "paths.demographics": "meta.csv"})
    index = build_index(cfg)

    by_file = dict(zip(index["file_name"], index["speaker_id"], strict=True))
    assert by_file["001.wav"] == by_file["002.wav"] == "p1"
    assert by_file["012.wav"] == "012"
    assert index["subject_id"].is_unique

    folds = subject_folds(index, n_splits=2, seed=0, group_column="speaker_id")
    check_no_group_leakage(index, folds, group_column="speaker_id")
    first, second = (index.index[index["file_name"] == name][0] for name in ("001.wav", "002.wav"))
    assert all((first in fold.test) == (second in fold.test) for fold in folds)


def test_two_speaker_columns_in_one_table_are_refused(tmp_path) -> None:
    table = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "File name": ["001", "002"],
            "Diagnosis": [0, 1],
            "Speaker": ["p1", "p2"],
            "Subject ID": ["p1", "p2"],
        }
    ).to_csv(table, index=False)
    with pytest.raises(DatasetError, match=r"'Speaker' and 'Subject ID' name the same field"):
        read_demographics(table)


def test_a_demographics_table_without_the_needed_columns_is_refused(tmp_path) -> None:
    table = tmp_path / "meta.csv"
    pd.DataFrame([{"who": "a", "what": 1}]).to_csv(table, index=False)
    with pytest.raises(DatasetError, match="lacks the column"):
        read_demographics(table)


def test_the_index_summary_counts_files_per_class(synthetic_corpus) -> None:
    index = scan_class_directories(synthetic_corpus / "train")
    summary = summarise_index(index)
    assert set(summary["label_name"]) == set(CLASS_NAMES.values())
    assert summary["n_files"].sum() == 12


# -- splits -------------------------------------------------------------
def _index(n_per_class: int = 6, n_classes: int = 4) -> pd.DataFrame:
    rows = [
        {"subject_id": f"c{label}_s{i}", "label": label}
        for label in range(n_classes)
        for i in range(n_per_class)
    ]
    return pd.DataFrame(rows)


def test_folds_never_split_a_subject() -> None:
    frame = _index()
    folds = subject_folds(frame, n_splits=3, seed=0)
    assert len(folds) == 3
    check_no_group_leakage(frame, folds)
    covered = np.concatenate([fold.test for fold in folds])
    assert sorted(covered.tolist()) == list(range(len(frame)))


def test_a_repeated_subject_is_kept_inside_one_fold() -> None:
    frame = _index()
    frame.loc[1, "subject_id"] = frame.loc[0, "subject_id"] + "x"
    frame["subject_id"] = frame["subject_id"].str.replace(r"_s[01]$", "_shared", regex=True)
    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    folds = subject_folds(frame, n_splits=2, seed=0)
    check_no_group_leakage(frame, folds)


def test_the_split_is_reproducible() -> None:
    frame = _index()
    first = subject_folds(frame, n_splits=3, seed=5)
    second = subject_folds(frame, n_splits=3, seed=5)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.test, b.test)


def test_a_different_seed_changes_the_split() -> None:
    frame = _index()
    a = subject_folds(frame, n_splits=3, seed=0)
    b = subject_folds(frame, n_splits=3, seed=1)
    assert any(not np.array_equal(x.test, y.test) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n_splits": 1}, "at least 2"),
        ({"n_splits": 40}, "cannot fill"),
    ],
)
def test_an_infeasible_split_is_reported(kwargs: dict, fragment: str) -> None:
    with pytest.raises(SplitError, match=fragment):
        subject_folds(_index(), **kwargs)


def test_a_single_class_cannot_be_stratified() -> None:
    with pytest.raises(SplitError, match="at least two classes"):
        subject_folds(_index(n_classes=1), n_splits=2)


def test_a_missing_column_is_named() -> None:
    with pytest.raises(SplitError, match="lacks the column"):
        subject_folds(pd.DataFrame({"label": [0, 1]}), n_splits=2)


def test_the_inner_split_partitions_the_training_rows() -> None:
    frame = _index()
    fold = subject_folds(frame, n_splits=2, seed=0)[0]
    fit_rows, val_rows = train_val_split(frame, fold.train, val_fraction=0.3, seed=0)
    assert np.intersect1d(fit_rows, val_rows).size == 0
    assert sorted(np.concatenate([fit_rows, val_rows]).tolist()) == sorted(fold.train.tolist())
    assert val_rows.size > 0


def test_the_inner_split_keeps_subjects_together() -> None:
    frame = _index()
    fold = subject_folds(frame, n_splits=2, seed=0)[0]
    fit_rows, val_rows = train_val_split(frame, fold.train, val_fraction=0.3, seed=0)
    shared = np.intersect1d(
        frame["subject_id"].to_numpy()[fit_rows], frame["subject_id"].to_numpy()[val_rows]
    )
    assert shared.size == 0


def test_the_inner_split_counts_speakers_rather_than_files() -> None:
    # The four normal files are one speaker, so the normal class has nothing to hold out.
    frame = pd.DataFrame(
        {
            "speaker_id": ["a", "a", "a", "a", "b", "c", "d", "e"],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    with pytest.raises(SplitError, match="cannot hold out"):
        train_val_split(frame, np.arange(len(frame)), val_fraction=0.5, group_column="speaker_id")


def test_the_inner_split_refuses_an_impossible_fraction() -> None:
    with pytest.raises(SplitError, match="val_fraction"):
        train_val_split(_index(), np.arange(24), val_fraction=0.95)
    with pytest.raises(SplitError, match=r"\(0, 0\.5\]"):
        train_val_split(_index(), np.arange(24), val_fraction=0.6)


@pytest.mark.parametrize(("fraction", "share"), [(0.2, 1 / 5), (0.3, 1 / 3), (0.4, 1 / 2)])
def test_the_held_out_share_is_one_over_the_rounded_fold_count(
    fraction: float, share: float
) -> None:
    frame = _index(n_per_class=30, n_classes=2)
    _, val_rows = train_val_split(frame, np.arange(len(frame)), val_fraction=fraction, seed=0)
    assert val_rows.size / len(frame) == pytest.approx(share, abs=0.02)


def test_the_inner_split_refuses_empty_rows() -> None:
    with pytest.raises(SplitError, match="empty set"):
        train_val_split(_index(), np.asarray([], dtype=int))


def test_a_fold_cannot_list_a_row_on_both_sides() -> None:
    from voice_pathology_detection.data.splits import Fold

    with pytest.raises(SplitError, match="both the training"):
        Fold(index=0, train=np.array([0, 1]), test=np.array([1, 2]))


def test_a_fold_describes_its_two_sides() -> None:
    frame = _index()
    folds = subject_folds(frame, n_splits=2, seed=0)
    assert folds[0].sizes[0] + folds[0].sizes[1] == len(frame)
    assert set(folds[0].to_dict()) == {"fold", "n_train", "n_test", "train", "test"}
