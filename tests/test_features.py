"""The front end: coefficient order, band limiting, deltas and normalisation."""

from __future__ import annotations

import numpy as np
import pytest
from speechdsp import mfcc as speechdsp_mfcc

from voice_pathology_detection.config import MFCCConfig, NormalizationConfig
from voice_pathology_detection.features.deltas import append_deltas, regression_delta, split_blocks
from voice_pathology_detection.features.mfcc import (
    C0_SCALE,
    FeatureError,
    band_limit,
    extract_features,
    extract_static,
    n_fft_for,
)
from voice_pathology_detection.features.normalize import (
    BlockVarianceNormalizer,
    GlobalCepstralNormalizer,
    IdentityNormalizer,
    PerUtteranceNormalizer,
    make_normalizer,
)

SR = 8000


# -- coefficient layout -------------------------------------------------
def test_static_features_have_the_configured_shape(tone, mfcc_cfg) -> None:
    static = extract_static(tone, SR, mfcc_cfg)
    assert static.ndim == 2
    assert static.shape[1] == mfcc_cfg.n_static == 13
    assert static.shape[0] > 10


def test_c0_is_the_last_column_and_carries_the_htk_scaling(tone) -> None:
    cfg = MFCCConfig(sample_rate=SR, frame_ms=25.0, hop_ms=10.0, low_freq=None, high_freq=None)
    ours = extract_static(tone, SR, cfg)
    reference = speechdsp_mfcc(
        tone,
        SR,
        n_mfcc=13,
        n_mels=cfg.n_mels,
        frame_ms=25.0,
        hop_ms=10.0,
        n_fft=cfg.resolved_n_fft,
        preemph=cfg.preemph,
        lifter=cfg.lifter,
        append_energy=False,
    )
    # c1..c12 are carried over untouched, c0 moves to the end and is rescaled.
    assert np.allclose(ours[:, :12], reference[:, 1:])
    assert np.allclose(ours[:, -1], C0_SCALE * reference[:, 0])


def test_full_feature_vector_is_three_blocks_of_static_coefficients(tone, mfcc_cfg) -> None:
    static = extract_static(tone, SR, mfcc_cfg)
    full = extract_features(tone, SR, mfcc_cfg)
    assert full.shape == (static.shape[0], 39)
    assert np.allclose(full[:, :13], static)


def test_deltas_can_be_switched_off(tone, mfcc_cfg) -> None:
    cfg = MFCCConfig(sample_rate=SR, frame_ms=25.0, hop_ms=10.0, high_freq=3800.0, use_deltas=False)
    assert extract_features(tone, SR, cfg).shape[1] == 13


def test_a_pure_tone_concentrates_energy_in_the_low_quefrencies(mfcc_cfg) -> None:
    t = np.arange(SR) / SR
    static = extract_static(np.sin(2 * np.pi * 300 * t), SR, mfcc_cfg)
    magnitudes = np.abs(static[:, :12]).mean(axis=0)
    assert magnitudes[:4].mean() > magnitudes[8:].mean()


# -- validation ---------------------------------------------------------
def test_a_recording_shorter_than_one_frame_is_refused(mfcc_cfg) -> None:
    with pytest.raises(FeatureError, match="shorter than"):
        extract_static(np.zeros(10), SR, mfcc_cfg)


def test_an_empty_waveform_is_refused(mfcc_cfg) -> None:
    with pytest.raises(FeatureError, match="empty waveform"):
        extract_static(np.zeros(0), SR, mfcc_cfg)


def test_a_non_positive_sampling_rate_is_refused(tone, mfcc_cfg) -> None:
    with pytest.raises(FeatureError, match="sr must be positive"):
        extract_static(tone, 0, mfcc_cfg)


def test_a_non_finite_sample_is_refused(tone, mfcc_cfg) -> None:
    # The zero-phase band filter would spread it over every frame.
    broken = tone.copy()
    broken[100] = np.nan
    with pytest.raises(FeatureError, match="1 non-finite sample"):
        extract_static(broken, SR, mfcc_cfg)


def test_a_recording_with_a_nan_is_skipped_by_name(corpus_config) -> None:
    from scipy.io import wavfile

    from voice_pathology_detection.data.dataset import build_index
    from voice_pathology_detection.features.pipeline import extract_corpus

    index = build_index(corpus_config)
    broken = np.sin(2 * np.pi * 220 * np.arange(SR) / SR).astype(np.float32)
    broken[100] = np.nan
    wavfile.write(index.loc[0, "path"], SR, broken)

    features = extract_corpus(index, corpus_config)
    assert features.n_files == len(index) - 1
    assert index.loc[0, "subject_id"] not in features.subjects
    assert np.all(np.isfinite(features.X))


def test_n_fft_is_raised_when_it_cannot_hold_the_frame() -> None:
    cfg = MFCCConfig(sample_rate=SR, frame_ms=25.0, n_fft=256)
    # 25 ms at 44100 Hz is far longer than the configured 256-point transform.
    assert n_fft_for(44100, cfg) == 2048
    assert n_fft_for(SR, cfg) == 256


# -- band limiting ------------------------------------------------------
def test_band_limiting_removes_energy_outside_the_band() -> None:
    t = np.arange(SR) / SR
    low, high = np.sin(2 * np.pi * 30 * t), np.sin(2 * np.pi * 1000 * t)
    filtered = band_limit(low + high, SR, (200.0, 3000.0))
    spectrum = np.abs(np.fft.rfft(filtered))
    frequencies = np.fft.rfftfreq(filtered.size, 1 / SR)
    assert spectrum[np.argmin(np.abs(frequencies - 30))] < 0.02 * spectrum.max()
    assert spectrum[np.argmin(np.abs(frequencies - 1000))] > 0.5 * spectrum.max()


def test_band_limiting_is_a_no_op_for_the_full_band(tone) -> None:
    assert band_limit(tone, SR, None) is tone
    assert band_limit(tone, SR, (0.0, SR / 2.0)) is tone


def test_a_waveform_too_short_to_filter_is_passed_through() -> None:
    short = np.ones(5)
    assert np.array_equal(band_limit(short, SR, (200.0, 3000.0)), short)


# -- dynamic coefficients ----------------------------------------------
def test_the_regression_recovers_the_slope_of_a_ramp() -> None:
    ramp = np.arange(20, dtype=float)[:, None] * 3.0
    velocity = regression_delta(ramp, width=7)
    assert np.allclose(velocity[4:-4], 3.0)


def test_acceleration_of_a_ramp_vanishes_in_the_interior() -> None:
    ramp = np.arange(30, dtype=float)[:, None]
    full = append_deltas(ramp, delta_width=7, accel_width=5)
    assert full.shape == (30, 3)
    assert np.allclose(full[8:-8, 2], 0.0, atol=1e-12)


def test_a_constant_sequence_has_no_dynamics() -> None:
    constant = np.full((12, 4), 2.5)
    full = append_deltas(constant)
    assert np.allclose(full[:, 4:], 0.0)


@pytest.mark.parametrize("width", [2, 4, 1, 0])
def test_even_or_tiny_regression_windows_are_refused(width: int) -> None:
    with pytest.raises(ValueError, match="odd integer"):
        regression_delta(np.zeros((5, 2)), width=width)


def test_a_one_dimensional_input_is_refused() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        regression_delta(np.zeros(5))


def test_split_blocks_exposes_the_three_parts_of_a_vector() -> None:
    features = np.arange(39 * 2, dtype=float).reshape(2, 39)
    blocks = split_blocks(features, 13)
    assert blocks.shape == (2, 3, 13)
    assert np.array_equal(blocks[0, 1], features[0, 13:26])


def test_split_blocks_refuses_a_dimension_that_does_not_divide() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        split_blocks(np.zeros((2, 40)), 13)


# -- normalisation ------------------------------------------------------
def test_block_normalisation_gives_every_block_unit_spread() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 5.0, size=(20, 39))
    out = BlockVarianceNormalizer(block_size=13).fit_transform(X)
    assert np.allclose(split_blocks(out, 13).std(axis=2), 1.0, atol=1e-6)


def test_block_normalisation_is_per_frame_and_carries_no_state() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 6))
    normalizer = BlockVarianceNormalizer(block_size=3).fit(X)
    # A frame transformed alone matches the same frame inside the batch.
    alone = normalizer.transform(X[3:4])
    assert np.allclose(alone, normalizer.transform(X)[3:4])


def test_block_normalisation_survives_a_constant_block() -> None:
    X = np.hstack([np.ones((4, 3)), np.arange(12.0).reshape(4, 3)])
    out = BlockVarianceNormalizer(block_size=3).fit_transform(X)
    assert np.all(np.isfinite(out))


def test_variance_divisor_differs_from_the_standard_deviation() -> None:
    X = np.array([[0.0, 2.0, 4.0, 0.0, 2.0, 4.0]])
    by_std = BlockVarianceNormalizer(block_size=3, divisor="std").fit_transform(X)
    by_var = BlockVarianceNormalizer(block_size=3, divisor="var").fit_transform(X)
    assert not np.allclose(by_std, by_var)


def test_global_normalisation_uses_training_statistics_only() -> None:
    rng = np.random.default_rng(2)
    train = rng.normal(5.0, 2.0, size=(200, 4))
    shifted = rng.normal(50.0, 2.0, size=(50, 4))

    normalizer = GlobalCepstralNormalizer(mode="cmvn").fit(train)
    assert np.allclose(normalizer.transform(train).mean(axis=0), 0.0, atol=1e-10)
    # Had the statistics been re-estimated on the shifted data, this would be
    # centred too; it must not be.
    assert np.all(np.abs(normalizer.transform(shifted).mean(axis=0)) > 5.0)


def test_global_normalisation_refuses_a_dimension_change() -> None:
    normalizer = GlobalCepstralNormalizer().fit(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="fitted on 3 feature"):
        normalizer.transform(np.zeros((5, 4)))


def test_global_normalisation_must_be_fitted_first() -> None:
    normalizer = GlobalCepstralNormalizer()
    with pytest.raises(ValueError, match="must be fitted"):
        normalizer.transform(np.zeros((2, 2)))


def test_per_utterance_normalisation_removes_a_per_recording_gain() -> None:
    base = np.arange(12.0).reshape(6, 2)
    stacked = np.vstack([base, base * 7.0 + 3.0])
    out = PerUtteranceNormalizer(mode="cmvn").fit_transform(stacked, group_sizes=[6, 6])
    assert np.allclose(out[:6], out[6:])


def test_per_utterance_normalisation_needs_group_sizes() -> None:
    normalizer = PerUtteranceNormalizer().fit(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="needs group_sizes"):
        normalizer.transform(np.zeros((4, 2)))


def test_group_sizes_must_partition_the_matrix() -> None:
    normalizer = PerUtteranceNormalizer().fit(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="out of step"):
        normalizer.transform(np.zeros((4, 2)), group_sizes=[3, 3])


def test_identity_normalisation_returns_a_copy() -> None:
    X = np.ones((3, 2))
    out = IdentityNormalizer().fit_transform(X)
    assert np.array_equal(out, X)
    assert out is not X


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", IdentityNormalizer),
        ("cvn_block", BlockVarianceNormalizer),
        ("cmn", GlobalCepstralNormalizer),
        ("cvn", GlobalCepstralNormalizer),
        ("cmvn", GlobalCepstralNormalizer),
        ("cmvn_per_utt", PerUtteranceNormalizer),
    ],
)
def test_the_factory_covers_every_configured_method(method: str, expected: type) -> None:
    assert isinstance(make_normalizer(NormalizationConfig(method=method)), expected)
