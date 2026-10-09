# voice-pathology-detection

[![CI](https://github.com/recklight/voice-pathology-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/voice-pathology-detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [繁體中文](README.zh-TW.md)

Research code that detects pathological voice in recordings of the sustained vowel `/a:/` and classifies the type of disorder.

This is research code: it has not been clinically validated, it is not a medical device, and its output must not be used as the basis for a diagnosis.

It answers two questions:

1. Detection (stage 1): is the voice normal or pathological?
2. Disorder type (stage 2): if it is pathological, does it fall under Neoplasm (tumor), Phonotrauma (vocal fold trauma) or Vocal palsy (vocal fold paralysis)?

Every frame of a recording gives 13 mel-frequency cepstral coefficients (MFCCs), and appending the first- and second-order deltas brings that to 39 dimensions. After cepstral normalization, a neural network classifies every frame, and the frame scores are aggregated into one decision per subject. For detecting pathological voice with cepstral vectors and a deep network, see Fang et al. (2019). The four classes are defined as in the FEMH Voice Data Challenge 2018 dataset (see [References](#references)).

The shared signal-processing and evaluation functions (MFCC, deltas, STFT spectrograms, endpoint detection and UAR) come from [`speechdsp`](https://pypi.org/project/speechdsp/) on PyPI rather than being reimplemented here.

---

## Contents

- [Overview](#overview)
- [Method](#method)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [CLI reference](#cli-reference)
- [Project layout](#project-layout)
- [Preparing the data](#preparing-the-data)
- [Configuration](#configuration)
- [Results and metrics](#results-and-metrics)
- [Design notes](#design-notes)
- [Development and testing](#development-and-testing)
- [References](#references)

---

## Overview

[![voice-pathology-detection workflow](docs/images/workflow.png)](https://raw.githack.com/recklight/voice-pathology-detection/master/docs/workflow.html?theme=dark)

Interactive diagram: [view it online](https://raw.githack.com/recklight/voice-pathology-detection/master/docs/workflow.html?theme=dark), or open [`docs/workflow.html`](docs/workflow.html) in a browser on your machine. You can zoom, search for a node, or trace the path between any two steps, e.g. to see where a step's input comes from.

Roughness, breathiness, unstable pitch and broken harmonic structure all show up in the short-time cepstrum, so the pipeline is built on frame-level cepstral features. It runs in seven stages:

| Stage | What it does | Module |
| --- | --- | --- |
| 1. Trimming | Removes the non-phonation parts at the start and end of a recording | `voice_pathology_detection.data.trim` |
| 2. Feature extraction | 13 cepstral coefficients per 30 ms frame, plus Δ and ΔΔ for 39 dimensions | `voice_pathology_detection.features.mfcc` |
| 3. Normalization | Cepstral mean/variance normalization, with statistics estimated on the training folds only | `voice_pathology_detection.features.normalize` |
| 4. Classification | Frame-level classification (neural network, DBN, SVM, GMM) | `voice_pathology_detection.models` |
| 5. Aggregation | Combines the frame scores into one decision per subject | `voice_pathology_detection.eval.aggregate` |
| 6. Threshold | Chooses the operating point on held-out data | `voice_pathology_detection.eval.threshold` |
| 7. Evaluation | Cross-validated metrics, figures and predictions written to disk | `voice_pathology_detection.eval` |

There are two routes to compare:

- `cascade` (two stages): detection runs first, whatever it calls pathological goes on to a three-class classifier, and the two results are merged into a four-class label 0/1/2/3.
- `flat` (one stage, four classes): a single classifier outputs all four classes directly.

Both routes run on identical folds with the same evaluation and reports, so comparing their numbers shows whether splitting the task into two stages helps at all.

---

## Method

### Trimming the recordings

Clinical recordings often have the doctor and the patient talking before and after the sustained vowel. That talk is not phonation and should stay out of the analysis. There are two ways to trim it:

- `fixed_edges`: cuts `edge_seconds` seconds from the start and from the end, but only from recordings longer than `min_seconds`, so a short recording of the prompted vowel alone is not cut away entirely.
- `energy_vad`: finds the phonation with dual-threshold endpoint detection on the short-time energy (`speechdsp.endpoint_detect`), without assuming anything about the length of the recording.

If the trimmed signal is shorter than `min_output_seconds`, the trim is thrown away: the untrimmed signal is kept and a warning is logged, so a wrong parameter cannot cut away a whole recording.

### Cepstral features

Each frame is pre-emphasized with $y[n] = x[n] - \alpha\,x[n-1]$ ($\alpha = 0.97$), Hamming-windowed and turned into a power spectrum, which then goes through $M = 23$ triangular filters spaced evenly on the mel scale:

$$
m = 1127 \ln\left(1 + \frac{f}{700}\right)
$$

The log filterbank energies go through a discrete cosine transform, which gives the cepstral coefficients:

$$
c_k = \sqrt{\frac{2}{M}} \sum_{m=1}^{M} \log(E_m) \cos\left(\frac{k\pi}{M}\left(m - \tfrac{1}{2}\right)\right),
\qquad k = 0, 1, \ldots, 12
$$

Sinusoidal cepstral liftering then evens out the dynamic range of the coefficients:

$$
\hat{c}_k = \left(1 + \frac{L}{2}\sin\frac{\pi k}{L}\right) c_k, \qquad L = 22
$$

The output follows HTK's `MFCC_0` layout. The columns are ordered $[c_1, \ldots, c_{12}, c_0]$: the zeroth coefficient is the last column of each 13-dimensional block, not the first. $c_0$ gets the same $\sqrt{2/M}$ scale factor as the other coefficients. An orthonormal DCT-II uses $\sqrt{1/M}$ for the zeroth term, a factor of $\sqrt{2}$ less, and `features/mfcc.py` puts that factor back.

Before the front end, a zero-phase Butterworth filter band-limits the signal without adding phase distortion. The defaults are `low_freq: 64` and `high_freq: null`, which is a 64 Hz high-pass and nothing else; to turn the high-pass off, write `low_freq: null` in the config file.

### Dynamic features

A static cepstrum describes a single 30 ms slice. How tremor and roughness rise and fall over time shows up only in how the coefficients change. The deltas are computed by regression:

$$
d_t = \frac{\sum_{n=1}^{N} n \,(c_{t+n} - c_{t-n})}{2\sum_{n=1}^{N} n^2}
$$

where $N = (\text{width} - 1)/2$. The first-order deltas use a window width of 7 ($N = 3$) by default. The second-order (acceleration) deltas apply the same regression to the first-order deltas, not a second difference to the static coefficients, with a default width of 5 ($N = 2$). Padding the edges with copies of the first and last frames keeps the number of frames unchanged. The final feature layout is:

```
[c1..c12, c0 | Δc1..Δc12, Δc0 | ΔΔc1..ΔΔc12, ΔΔc0]     39 dimensions
```

### Cepstral normalization

"Cepstral normalization" covers three different operations, and once they are mixed up the numbers stop being comparable. I keep them apart here, as six settings selected with `normalization.method`:

| `normalization.method` | Operation | State |
| --- | --- | --- |
| `none` | Nothing is done | None |
| `cvn_block` | Per frame: divides each 13-dimensional block (static / first-order / second-order) by the standard deviation of the coefficients in that block | Stateless |
| `cmn` / `cvn` / `cmvn` | Along time: zero mean / unit variance for each coefficient separately | Stateful, `fit` on the training folds only |
| `cmvn_per_utt` | Same as above, but the statistics come from each recording itself | Stateless |

The three `cvn_block` blocks are `0:13 / 13:26 / 26:39`, and the divisor is the standard deviation, which keeps the features dimensionless; to divide by the variance, set `normalization.divisor: var`. `cmvn` is the standard method in the literature. Its statistics are estimated on the training folds only and applied unchanged to the validation and test folds, so no statistic from an evaluation fold reaches the model.

### Classifiers

| Name | Description |
| --- | --- |
| `mlp` | Sigmoid multilayer perceptron, SGD with cross-entropy; 5 layers × 200 neurons by default for stage 1, 4 layers × 100 for stage 2 and for the single-stage four-class route |
| `dbn` | Deep belief network: unsupervised layer-wise pre-training with restricted Boltzmann machines, then supervised fine-tuning |
| `torch_dbn` | PyTorch version of the same network (optional extra `dl`), with fewer options than `dbn`; see below |
| `svm` | RBF-kernel support vector machine, with standardization in front |
| `linear_svm` | Linear support vector machine |
| `gmm` | One Gaussian mixture model per class, combined into posteriors with Bayes' theorem |

`dbn` pre-trains layer by layer with restricted Boltzmann machines trained by contrastive divergence (CD-k). The hidden-unit probabilities of each layer become the input of the layer above, the trained weights go into the hidden layers of a feedforward network, and backpropagation fine-tunes it. `pretrain: false` trains exactly the same topology with the same seed from random initialization, so comparing the two runs measures directly what pre-training adds.

`torch_dbn` is for running large corpora on a GPU and supports only some of the `dbn` options. Pre-training is always contrastive divergence with a single Gibbs step, and fine-tuning always runs `max_iter` epochs of SGD with momentum, with no early stopping and no L2 penalty. It accepts `hidden_layer_sizes`, `pretrain`, `pretrain_epochs`, `pretrain_learning_rate`, `max_iter`, `batch_size`, `learning_rate_init`, `momentum` and `device`. The `dbn`-only parameters `cd_steps`, `update`, `alpha` and `early_stopping` raise an error, so delete those keys from `configs/dbn_pretrained.yaml` before changing the model name to `torch_dbn`.

`dbn` can also replace the gradient step of contrastive divergence with a maximum-entropy update rule (`update: maxent`). The parameters are treated as the Lagrange multipliers of a maximum-entropy problem with a Gaussian prior of variance $\sigma^2$, and each step uses Newton's method to solve the stationarity condition of the Gaussian-prior update in Chen and Rosenfeld (2000):

$$
f(\gamma) = F\sigma^2 e^{\gamma S} + \gamma + \lambda - \mu\sigma^2 = 0,
\qquad
f'(\gamma) = F S \sigma^2 e^{\gamma S} + 1
$$

Here $F$ is the model expectation, $\mu$ the data expectation, $\lambda$ the current parameter and $S = \sum_i F_i$. The derivative has a closed form, Newton's method converges within a few iterations, and all parameters are solved at once, vectorized in NumPy. The prior pulls large updates towards zero, which can damp the heavy oscillation of contrastive divergence in the first few epochs.

### From frames to subjects

The model looks at 30 ms frames, but the question is about a person. An aggregation step sits in between, and which rule it uses changes the result:

| `aggregation` | Rule |
| --- | --- |
| `mean_proba` | Averages the posteriors over all frames of the recording and picks the class with the highest average (default) |
| `majority_vote` | Counts the hard decisions and takes the class with the most votes |
| `ratio_threshold` | Binary tasks only: pathological once the share of pathological frames reaches the threshold |

The mean posterior is the default because it keeps how confident each frame is, while a hard vote only sees which class each frame went to.

### Decision threshold

The threshold is set on a validation set split off inside the training folds; the evaluation fold has no part in it:

- `youden`: maximizes Youden's J, i.e. $\text{sensitivity} + \text{specificity} - 1$. The default when both kinds of error cost the same.
- `target_sensitivity` / `target_specificity`: among the thresholds that reach the given sensitivity (or specificity), takes the one with the best specificity (or sensitivity).
- `fixed`: uses the value in the config file unchanged, to reproduce a particular setup.

On the validation set, every threshold between the lowest score called positive and the highest score called negative gives the same result. I put the threshold at the midpoint of the two rather than right at the lowest positive score, so a new subject whose score is slightly lower is still called pathological. What the threshold is compared with depends on the aggregation rule: the mean posterior of the pathological class for `mean_proba`, and the share of pathological frames for `majority_vote` and `ratio_threshold`.

The size of the validation set comes from `split.val_fraction`. The training folds go through one more grouped, stratified k-fold split and one of those folds is held out, with k = round(1 / val_fraction). So 0.2 holds out a fifth, 0.3 a third, 0.4 and 0.5 both a half, and values above 0.5 are rejected. If a class has fewer than k groups in the training folds, k drops to that number of groups and the held-out share grows with it. When the training folds are too small to give a validation set that contains both classes, that fold falls back to a fixed threshold of 0.5 and logs a warning.

The threshold chosen in each fold is written to `metrics.json`, where you can see straight away how far the folds differ.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/recklight/voice-pathology-detection.git
cd voice-pathology-detection

# 1. Install the package (the shared speechdsp package comes from PyPI with it)
pip install -e .

# 2. For development and running the tests
pip install -e ".[dev]"

# 3. Optional: the PyTorch DBN (everything else works exactly the same without it)
pip install -e ".[dl]"
```

Main dependencies: `speechdsp>=0.2.0`, `numpy>=1.24`, `scipy>=1.10`, `scikit-learn>=1.2`, `pandas>=1.5`, `matplotlib>=3.6`, `pyyaml`, `typer`, `joblib`, `openpyxl`. The code works with both NumPy 1.x and 2.x: besides testing against the latest releases, CI has a job that runs the full suite on the minimum versions above.

---

## Quick start

You can run the whole pipeline end to end without any data:

```bash
voice-pathology-detection demo --out-dir results/demo
```

This synthesizes a small batch of artificial vowels in a temporary folder, runs the two-stage cascade, and writes the config, metrics, predictions and figures to `results/demo/demo-<timestamp>/`. Numbers from synthetic data say nothing about how well real voices are handled. They only show that the pipeline runs from end to end and that the same seed gives the same result.

With real data:

```bash
# 1. Build the index and check the file count of each class
voice-pathology-detection prepare --data-root /path/to/corpus

# 2. Cross-validate the two-stage cascade
voice-pathology-detection train --route cascade --config configs/detection_dnn.yaml \
    --set paths.data_root=/path/to/corpus

# 3. Predict on the test set and write the prediction table
voice-pathology-detection predict results/cascade-20261008-101500 \
    --set paths.data_root=/path/to/corpus --output predictions.xlsx
```

By default `prepare` compares the file count of each class with the make-up of the reference corpus (Normal 50, Neoplasm 40, Phonotrauma 60, Vocal palsy 50) and only logs a warning when they differ; `--strict` turns that into an error. If your dataset was never meant to look like that, ignore the warning, or pass your own `expected_counts` through the Python API (see [docs/data-format.md](docs/data-format.md); the documents under `docs/` are in Traditional Chinese).

`predict` starts from the `config.yaml` in the results folder, so it usually needs nothing more than where the data is. Prediction has to use the same `mfcc` and `trim` settings the model was trained with. If `--config` or `--set` changes either of those sections, it stops with an error that lists the keys that differ.

If the results folder contains `flat.joblib`, that model is used. Otherwise `predict` reads `detection.joblib` and `disorder.joblib`, and the detector applies the threshold chosen during training. Point it at the output of `--route flat` or `--route cascade`: a folder from `detection` or `disorder` alone lacks the other model, and the error names the missing file. The prediction table has two columns, `File name` and `Prediction` (0–3). Without `--output` it is written to `predictions_<split>.xlsx` in the results folder.

---

## Usage

### CLI

```bash
# Show the fully resolved config and the available models
voice-pathology-detection info --config configs/default.yaml

# Stage 1 only (normal vs pathological), with 5 folds and a support vector machine
voice-pathology-detection train --route detection \
    --config configs/detection_dnn.yaml \
    --folds 5 --model svm \
    --set paths.data_root=/path/to/corpus

# Compare the single-stage four-class route with the two-stage cascade
voice-pathology-detection train --route flat    --config configs/flat_dnn.yaml
voice-pathology-detection train --route cascade --config configs/detection_dnn.yaml

# Extract and cache the features so later experiments do not recompute them
voice-pathology-detection features --config configs/default.yaml --output cache/train.npz

# Narrowband spectrograms (harmonic structure of a normal voice against the three disorders)
voice-pathology-detection spectrograms normal.wav neoplasm.wav phonotrauma.wav palsy.wav \
    --output results/figures/spectrograms.png
```

`--model` replaces the estimator itself. The `model.params` in the config file were written for the model being replaced (the number of network layers, for example) and would make a support vector machine fail, so `--model` clears them too and the new model runs with its own defaults. To set your own parameters, add `--set <section>.model.params='{...}'` to the same command (the section is `detection`, `disorder` or `flat`), and the whole set is kept. A single key, such as `--set detection.model.params.C=2`, is added to the now-empty params instead.

`--set` overrides any setting and can be repeated:

```bash
voice-pathology-detection train --route cascade \
    --set normalization.method=cvn_block \
    --set detection.model.params='{"hidden_layer_sizes": [300, 300], "max_iter": 50}' \
    --set detection.threshold_method=target_sensitivity \
    --set detection.target=0.95

# Change the model and set its parameters: --model does not clear params you set explicitly
voice-pathology-detection train --route detection --model dbn \
    --set detection.model.params='{"hidden_layer_sizes": [200, 200], "pretrain": true}'
```

### Python API

```python
from voice_pathology_detection import (
    Config, build_index, extract_corpus, run_cascade, save_run, set_global_seed,
)

# 1. Config: paths and hyperparameters all live in the config object
cfg = Config.from_yaml("configs/detection_dnn.yaml").override(
    {"paths.data_root": "/path/to/corpus", "split.n_splits": 5, "seed": 0}
)
set_global_seed(cfg.seed)

# 2. Index: scan the folders and check each class's file count against the expected one
index = build_index(cfg, strict_counts=True)
print(index[["subject_id", "label_name", "split"]].head())

# 3. Features: the FeatureSet keeps the frame matrix and which recording each frame came from
features = extract_corpus(index, cfg, cache_dir=cfg.paths.cache_dir)
print(features.X.shape, features.n_files, features.group_sizes.sum())

# 4. Cross-validate the two-stage cascade
result = run_cascade(features, cfg)
print(f"four-class UAR = {result.overall_summary['uar']['mean']:.4f} "
      f"± {result.overall_summary['uar']['std']:.4f}")
print(f"detection sensitivity = {result.detection_summary['sensitivity']['mean']:.4f}")
print(f"threshold chosen in each fold: {result.thresholds}")

# 5. Save: config, metrics, reports and per-subject predictions all go into one folder
run_dir = save_run(cfg.paths.results_dir, cfg, result.to_dict(),
                   tables={"predictions": result.predictions})
```

The front end and normalization on their own:

```python
import numpy as np
from speechdsp import read_wav
from voice_pathology_detection import Config, extract_features, make_normalizer, trim_audio

cfg = Config()
audio, sr = read_wav("subject_001.wav")

trimmed = trim_audio(audio, sr, cfg.trim)          # drop the non-phonation parts
X = extract_features(trimmed.audio, sr, cfg.mfcc)  # (frames, 39)

normalizer = make_normalizer(cfg.normalization).fit(X)
X_norm = normalizer.transform(X)
print(X.shape, trimmed.duration(sr))
```

A detection experiment put together by hand (continuing from `features` above). The threshold is chosen on the validation set split off the training folds and then used on the evaluation fold:

```python
from voice_pathology_detection.data.splits import (
    check_no_group_leakage, subject_folds, train_val_split,
)
from voice_pathology_detection.eval.aggregate import aggregate_frames
from voice_pathology_detection.eval.metrics import evaluate
from voice_pathology_detection.eval.threshold import select_threshold
from voice_pathology_detection.models.registry import build_model
from voice_pathology_detection.routes import binary_view

binary = binary_view(features)                  # labels become 0 = normal, 1 = pathological
folds = subject_folds(binary.files, n_splits=5, seed=0)
check_no_group_leakage(binary.files, folds)     # no subject ends up in two folds

fold = folds[0]
fit_rows, val_rows = train_val_split(binary.files, fold.train, val_fraction=0.2, seed=0)
fit_set, val_set, test_set = (binary.select(rows) for rows in (fit_rows, val_rows, fold.test))

model = build_model("gmm", {"n_components": 1}).fit(fit_set.X, fit_set.frame_labels)
val = aggregate_frames(model.predict_proba(val_set.X), val_set.group_sizes, mode="mean_proba")
choice = select_threshold(val_set.file_labels, val.score(1), method="youden")

test = aggregate_frames(model.predict_proba(test_set.X), test_set.group_sizes, mode="mean_proba")
pred = (test.score(1) >= choice.threshold).astype(int)
report = evaluate(test_set.file_labels, pred, test.score(1), level="subject")
```

### Example scripts

The two scripts in `examples/` run entirely on synthetic data and need no recordings:

```bash
python examples/end_to_end_synthetic.py --out-dir results/example
python examples/compare_normalizations.py --out-csv results/comparison.csv
```

The first runs the whole pipeline and compares the two routes. The second compares four normalization settings (`none`, `cvn_block`, `cmvn`, `cmvn_per_utt`) and three models on exactly the same folds and seed, so each row of the table changes one variable and nothing else.

---

## CLI reference

| Command | Purpose | Common options |
| --- | --- | --- |
| `voice-pathology-detection info` | Shows the resolved config, the available models and the class encoding | `--config`, `--set` |
| `voice-pathology-detection prepare` | Scans the folders into an index and checks the file count of each class | `--data-root`, `--split`, `--strict` / `--no-strict`, `--output` |
| `voice-pathology-detection features` | Extracts and caches the frame feature matrix | `--config`, `--split`, `--output` |
| `voice-pathology-detection train` | Cross-validates one route and writes metrics, figures and models | `--route`, `--folds`, `--model`, `--normalization`, `--run-id` |
| `voice-pathology-detection predict` | Runs a saved model on new recordings and writes a prediction table | `--data-root`, `--split`, `--output` |
| `voice-pathology-detection spectrograms` | Draws narrowband spectrograms | `--n-fft`, `--noverlap`, `--max-freq` |
| `voice-pathology-detection demo` | Runs the whole pipeline on synthetic audio, no data needed | `--per-class`, `--folds`, `--out-dir` |

The four values of `--route`:

| Value | Meaning |
| --- | --- |
| `detection` | Stage 1: normal vs pathological |
| `disorder` | Stage 2: the three disorder types (trained and evaluated on pathological recordings only) |
| `cascade` | Both stages in cascade, merged into a final four-class label 0/1/2/3 |
| `flat` | Single-stage four-class classifier (the comparison baseline) |

---

## Project layout

```
voice-pathology-detection/
├── README.md
├── README.zh-TW.md                   # Traditional Chinese README
├── LICENSE
├── pyproject.toml
├── configs/                          # config files; every experiment parameter lives here
│   ├── default.yaml                  # the built-in defaults written out in full
│   ├── mfcc_30ms.yaml                # front end with 30 ms / 15 ms / 23 mel
│   ├── detection_dnn.yaml            # stage 1: 5 layers × 200
│   ├── disorder_dnn.yaml             # stage 2: 4 layers × 100
│   ├── flat_dnn.yaml                 # single-stage four-class: 4 layers × 100
│   ├── dbn_pretrained.yaml           # DBN with RBM pre-training
│   └── baselines.yaml                # shallow baselines (SVM, GMM)
├── docs/
│   ├── images/workflow.png           # the README figure
│   ├── images/workflow.zh-TW.png     # the same figure in Traditional Chinese
│   ├── workflow.html                 # interactive diagram (open in a browser)
│   ├── workflow.zh-TW.html           # the same diagram in Traditional Chinese
│   ├── algorithms.md                 # equations and parameters of each algorithm
│   ├── data-format.md                # data formats and label encoding
│   └── reproduce.md                  # how to reproduce a run
├── examples/
│   ├── end_to_end_synthetic.py       # full walkthrough on synthetic data
│   └── compare_normalizations.py     # comparison table of normalizations and models
├── src/voice_pathology_detection/
│   ├── __init__.py                   # re-exports of the public API
│   ├── config.py                     # config object (dataclass + YAML + validation)
│   ├── seed.py                       # sets every random seed in one place and records it
│   ├── logging_utils.py              # logging setup
│   ├── cli.py                        # typer command-line interface
│   ├── prediction_table.py           # the table predict writes (File name / Prediction)
│   ├── data/
│   │   ├── labels.py                 # class codes, names, binary mapping, file-count check
│   │   ├── dataset.py                # builds the file index (folders or a label table)
│   │   ├── trim.py                   # trimming (fixed_edges / energy_vad)
│   │   ├── splits.py                 # grouped, stratified k-fold cross-validation
│   │   └── synthetic.py              # synthetic vowels and synthetic feature matrices
│   ├── features/
│   │   ├── mfcc.py                   # cepstral front end in the HTK MFCC_0 layout
│   │   ├── deltas.py                 # regression Δ / ΔΔ and block slicing
│   │   ├── normalize.py              # the six normalization settings (sklearn interface)
│   │   └── pipeline.py               # FeatureSet: frames bound to their source and index
│   ├── models/
│   │   ├── mlp.py                    # sigmoid multilayer perceptron
│   │   ├── rbm.py                    # restricted Boltzmann machine (CD-k / maxent)
│   │   ├── maxent.py                 # Newton solver for the maximum-entropy multipliers
│   │   ├── dbn.py                    # layer-wise pre-training + supervised fine-tuning
│   │   ├── torch_dbn.py              # the optional PyTorch version
│   │   ├── svm.py                    # support vector machine baselines
│   │   ├── gmm.py                    # Gaussian mixture model baseline
│   │   └── registry.py               # builds a model from its name
│   ├── routes/
│   │   ├── stage.py                  # shared training and evaluation loop
│   │   ├── detection.py              # stage 1
│   │   ├── disorder.py               # stage 2
│   │   ├── cascade.py                # two-stage cascade
│   │   └── flat.py                   # single-stage four-class
│   ├── eval/
│   │   ├── aggregate.py              # frames → subjects
│   │   ├── threshold.py              # threshold selection and sweep
│   │   ├── metrics.py                # UAR, sensitivity, specificity, PPV/NPV, AUC
│   │   ├── plots.py                  # confusion matrices, threshold curves, ROC, spectrograms
│   │   └── report.py                 # metrics.json / report.md / config.yaml
│   └── fusion/
│       └── score_fusion.py           # score fusion across systems (grid search and stacking)
└── tests/                            # all fixtures are generated in code
```

---

## Preparing the data

No recordings come with the package. You need your own data, and the configuration has to point at it.

### Supported folder layouts

#### Option 1: one folder per diagnosis (recommended)

```
/your/data/root/
├── train/
│   ├── Normal/            001.wav 002.wav ...
│   ├── Neoplasm/          001.wav 002.wav ...
│   ├── Phonotrauma/       001.wav 002.wav ...
│   └── Vocal palsy/       001.wav 002.wav ...
└── test/
    └── *.wav
```

Folder names are case-insensitive: `Vocal palsy`, `vocal_palsy`, `VocalPalsy` and `palsy` are all recognized.

#### Option 2: flat folders plus a label table

```
/your/data/root/
├── train/                 001.wav 002.wav ...
├── test/                  *.wav
└── Demographics.xlsx      columns: File name / Diagnosis(0-3) / Gender / Age
```

Set `paths.demographics: Demographics.xlsx`; `.csv` and `.tsv` work as well. Column names are matched against case-insensitive aliases: `File name`, `filename`, `Diagnosis(0-3)`, `label` and similar names are all recognized. Every cell is read as text, which keeps the leading zeros of a file name like `001`, and file names may be given with or without the extension. Recordings that are missing from the table, or whose diagnosis cell is empty, raise an error that lists their file names instead of being skipped silently.

When one person has several recordings, add a `Subject ID` (or `Speaker`) column to the table and set `split.group_by: speaker_id`. All recordings of one person then land in the same fold, and the inner validation split keeps them together as well. A recording with no speaker in the table forms a group of its own.

### Audio format

- WAV, mono or multichannel (multichannel audio is averaged down to mono)
- The sample rate should match `mfcc.sample_rate`; if it does not, a warning is logged and the actual sample rate is used
- A sustained `/a:/` vowel
- Any length, but recordings shorter than one frame (30 ms by default) or containing NaN or infinite samples are skipped, and the log records how many were skipped along with the first three file names

### Label encoding

| Code | Four-class task | Binary task |
| --- | --- | --- |
| 0 | Normal | Normal |
| 1 | Neoplasm | Pathological |
| 2 | Phonotrauma | Pathological |
| 3 | Vocal palsy | Pathological |

### Pointing to your data

Any one of these three works:

```bash
# a. Command-line option
voice-pathology-detection train --data-root /your/data/root

# b. Override with --set
voice-pathology-detection train --set paths.data_root=/your/data/root

# c. Your own config file
cp configs/default.yaml configs/local.yaml   # then edit paths.data_root in it
voice-pathology-detection train --config configs/local.yaml
```

`data/`, `results/`, `cache/`, `configs/local*.yaml` and every `*.wav`, `*.npy`, `*.npz`, `*.joblib`, `*.xlsx`, `*.csv` and `*.tsv` are listed in `.gitignore`: label tables, indexes and prediction tables carry file names and diagnoses, and your own config files carry local paths.

---

## Configuration

The configuration is a nested dataclass. Every field is validated on load, and a wrong key is named in the error:

```
ConfigError: unknown configuration key(s): ['mfcc.window_ms'];
valid keys in this section are ['accel_width', 'delta_width', 'frame_ms', ...]
```

Main sections:

| Section | Key fields |
| --- | --- |
| `paths` | `data_root`, `train_dir`, `test_dir`, `demographics`, `cache_dir`, `results_dir` |
| `mfcc` | `sample_rate`, `frame_ms`, `hop_ms`, `n_mels`, `n_ceps`, `lifter`, `preemph`, `low_freq`, `high_freq`, `use_deltas`, `delta_width`, `accel_width` |
| `trim` | `mode`, `edge_seconds`, `min_seconds`, `min_output_seconds` |
| `normalization` | `method`, `block_size`, `divisor`, `eps` |
| `split` | `n_splits`, `seed`, `group_by`, `val_fraction` |
| `detection` | `model.name`, `model.params`, `aggregation`, `threshold_method`, `threshold`, `target` |
| `disorder` | `model.name`, `model.params`, `aggregation` |
| `flat` | `model.name`, `model.params`, `aggregation` |
| Top level | `seed`, `log_level` |

Each route reads the section named after it; `cascade` reads both `detection` and `disorder`.

Without `--config` the built-in defaults apply (except for `predict`, which starts from the `config.yaml` in the results folder). `configs/default.yaml` writes the same values out in full (a test checks that the two agree), and a copy of it makes a good starting point for your own config file. The other config files hold only the sections their experiment uses, and any key they leave out keeps its default. Sections such as `seed` and `normalization` are written out even when they match the defaults, which makes the files easier to compare.

Every run writes the complete configuration it actually used to `config.yaml` in its results folder, so any experiment can be rerun from its own output:

```bash
voice-pathology-detection train --route cascade --config results/cascade-20261008-101500/config.yaml
```

---

## Results and metrics

Each run creates one folder. For `--route detection`:

```
results/<run_id>/
├── config.yaml              # the complete configuration in effect
├── metrics.json             # every metric (per fold + mean ± std + pooled)
├── report.md                # human-readable summary report
├── predictions.csv          # per-subject predictions (each subject exactly once)
├── folds_subject.csv        # subject-level metrics per fold
├── folds_frame.csv          # frame-level metrics per fold
├── threshold_sweep.csv      # threshold sweep
├── detection.joblib         # model and threshold for predict
└── figures/
    ├── confusion_subject.png
    ├── confusion_frame.png
    ├── folds_subject.png
    ├── threshold.png
    └── roc.png
```

Some of these depend on the route:

| File | Routes that write it |
| --- | --- |
| `config.yaml`, `metrics.json`, `report.md`, `predictions.csv` | All |
| `folds_subject.csv`, `folds_frame.csv` | `detection`, `disorder`, `flat`; `cascade` writes no per-fold data |
| `figures/confusion_subject.png`, `figures/confusion_frame.png`, `figures/folds_subject.png` | `detection`, `disorder`, `flat` |
| `threshold_sweep.csv`, `figures/threshold.png`, `figures/roc.png` | Only routes with a binary stage, i.e. `detection` and `cascade` |
| `figures/confusion_four_way.png`, `figures/confusion_detection.png` | `cascade` |
| `detection.joblib` | `detection`, `cascade` |
| `disorder.joblib` | `disorder`, `cascade` |
| `flat.joblib` | `flat` |

`--no-save-models` skips the `.joblib` files. `--no-figures` only skips the plots in `figures/`; `threshold_sweep.csv` is still written.

The `.joblib` files hold models trained separately after cross-validation, for `predict` to use. The detector's threshold needs files that took no part in training, so `detection.joblib` is built the same way as each cross-validation fold: an inner validation set is split off according to `split.val_fraction`, the model is trained on the remaining files only (usually four fifths at the default 0.2, somewhat less when a class has fewer than 5 groups), and the threshold is chosen on the validation set. `disorder.joblib` and `flat.joblib` have no threshold to choose, so they are trained on all training files, and so is `detection.joblib` when `detection.threshold_method` is `fixed`.

### Metrics

| Metric | Definition | Why it matters |
| --- | --- | --- |
| UAR | Arithmetic mean of the per-class recalls over the classes present in the true labels, the same as `sklearn.metrics.balanced_accuracy_score` | The main metric for this task. With 50 normal and 150 pathological cases, accuracy rewards a model that always guesses the majority class; UAR does not |
| Sensitivity | $TP / (TP + FN)$ | How many pathological voices are caught |
| Specificity | $TN / (TN + FP)$ | How many normal voices are correctly called normal |
| PPV / NPV | $TP/(TP+FP)$, $TN/(TN+FN)$ | Conditioned on the model's decision rather than on the true class. They shift with prevalence, so they can only be compared between experiments with the same class proportions |
| ROC-AUC | Area under the ROC curve | Independent of the threshold; measures the ranking itself |
| Confusion matrix | Counts of true against predicted classes | Which class is mistaken for which |

Sensitivity, specificity, PPV and NPV always take pathological as positive and normal as negative. The four-class final decisions of `flat` and `cascade` are scored the same way, with the three disorders together as the positive class, so their numbers mean the same thing as those of `detection`. `disorder` has no normal class, which leaves these four `null` in `metrics.json` (any undefined value is written as `null`). `per_class_recall` lists every class; a class that does not appear in the true labels is recorded as 0 but does not count towards the UAR.

`detection`, `disorder` and `flat` report their metrics at both the frame level and the subject level. A model can look strong on frames and weak on subjects, and only separate reports show that. The pooled frame-level figures (`pooled_frame`) are computed once over the frames of all evaluation folds together and therefore match the frame-level confusion matrix. `cascade` has the subject level only.

Results across folds are always reported as `mean ± std`, because a mean alone does not show how much the folds differ: a mean of 0.62 can come from folds that all sit near 0.62 or from folds scattered over 0.45–0.80.

### Reading the three cascade reports

The two-stage cascade reports three views:

1. `detection_summary`: how well stage 1 does on its own.
2. `disorder_summary`: three-class performance on the subjects correctly passed to stage 2.
3. `overall_summary`: the merged four-class final decision. This is the one to quote for the system as a whole.

A normal subject that stage 1 wrongly lets through is always assigned some disorder, which is bound to be wrong. A pathological subject that stage 1 calls normal never gets a chance to be classified. Both are errors of the cascade as a system, and both show up only in the third report.

`cascade` writes no per-fold reports. Its `metrics.json` holds the cross-fold `mean` / `std` / `n` of the three views, the metrics and confusion matrices over all subjects pooled, and for each fold the threshold and the number of subjects sent to stage 2.

---

## Design notes

An evaluation fold is used once, to score the model, and for nothing else. Files are split in groups by `split.group_by` (`subject_id` by default) and stratified by label, and `check_no_group_leakage()` confirms on every run that no group crosses folds. The model, the normalization statistics and the decision threshold all come from the training folds only, the threshold from a validation set split off inside them. [docs/reproduce.md](docs/reproduce.md) lists which data each step uses.

The complete configuration, seed, metrics and predictions of a run are all written to one results folder. `set_global_seed()` seeds `random`, NumPy and, if it is installed, PyTorch in one call, and the seed is recorded in `metrics.json`. Without `--run-id` the folder name carries a timestamp, and a second run within the same second gets `-2`, `-3` appended instead of writing into the same folder.

`detection`, `disorder` and `flat` share `run_stage()` in `routes/stage.py`, and `cascade` reuses its step functions (`inner_split`, `fit_normalizer`, `apply_normalizer`, `frame_posteriors`), so all four routes split and evaluate identically. DBN pre-training can be switched off with `pretrain: false`, and the SVM, linear SVM and GMM run through the same splits and evaluation. That makes it a fair test of whether the network's complexity pays off.

A single recording has tens to hundreds of frames, so the frame-to-subject aggregation is done in one `np.add.reduceat` call, the threshold sweep comes from one boolean matrix, and the maximum-entropy solver runs its Newton iterations on all parameters at once. None of them loops over files in Python. The library itself never calls `print`; messages go through `logging`. How much the command line prints is set by `log_level` in the config file, and `--log-level` overrides it for a single run (`spectrograms` and `demo` read no config file and default to INFO).

---

## Development and testing

```bash
# Run the tests (unit tests and docstring examples together)
pytest -q

# Static checks
ruff check .
ruff format --check .

# Tests for a single module
pytest tests/test_features.py -v
```

There are 435 tests, 354 unit tests plus 81 docstring examples, and all of them use synthetic fixtures: sine waves of known frequency, generated feature matrices and labels, and artificial vowels made with a source-filter model. No real recording is read.

The tests check mathematical properties and conditions the pipeline must always meet, rather than comparing output with stored results. For example, the first-order delta of a linear ramp must return the slope itself; UAR must equal `sklearn.metrics.balanced_accuracy_score`; the maximum-entropy solution must agree with an independent root found by `scipy.optimize.brentq`; no subject crosses folds; every subject appears exactly once in the out-of-fold predictions; and no normalization statistic leaks from the test data.

Tests that need PyTorch are skipped via `pytest.importorskip` when it is not installed. The whole package imports without PyTorch, and every other test still has to pass. CI runs `ruff check` and `ruff format --check` (`target-version = "py310"`), then checks the syntax of every file with `ast.parse(..., feature_version=(3, 10))`.

---

## References

### Feature extraction

- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980.
- S. Furui, "Speaker-independent isolated word recognition using dynamic features of speech spectrum," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 34, no. 1, pp. 52–59, 1986.
- S. Young, G. Evermann, M. Gales, et al., *The HTK Book (for HTK Version 3.4)*, Cambridge University Engineering Department, 2006.

### Cepstral normalization

- S. Furui, "Cepstral analysis technique for automatic speaker verification," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 29, no. 2, pp. 254–272, 1981.
- O. Viikki and K. Laurila, "Cepstral domain segmental feature vector normalization for noise robust speech recognition," *Speech Communication*, vol. 25, no. 1–3, pp. 133–147, 1998.

### Deep learning

- G. E. Hinton, S. Osindero, and Y.-W. Teh, "A fast learning algorithm for deep belief nets," *Neural Computation*, vol. 18, no. 7, pp. 1527–1554, 2006.
- G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," *Science*, vol. 313, no. 5786, pp. 504–507, 2006.
- G. E. Hinton, "Training products of experts by minimizing contrastive divergence," *Neural Computation*, vol. 14, no. 8, pp. 1771–1800, 2002.
- G. E. Hinton, "A practical guide to training restricted Boltzmann machines," UTML TR 2010-003, Department of Computer Science, University of Toronto, 2010.
- D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986.
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," *AISTATS*, pp. 249–256, 2010.
- E. T. Jaynes, "Information theory and statistical mechanics," *Physical Review*, vol. 106, no. 4, pp. 620–630, 1957.
- S. F. Chen and R. Rosenfeld, "A survey of smoothing techniques for ME models," *IEEE Transactions on Speech and Audio Processing*, vol. 8, no. 1, pp. 37–50, 2000.

### Baselines

- C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
- D. A. Reynolds and R. C. Rose, "Robust text-independent speaker identification using Gaussian mixture speaker models," *IEEE Transactions on Speech and Audio Processing*, vol. 3, no. 1, pp. 72–83, 1995.
- J. I. Godino-Llorente, P. Gómez-Vilda, and M. Blanco-Velasco, "Dimensionality reduction of a pathological voice quality assessment system based on Gaussian mixture models and short-term cepstral parameters," *IEEE Transactions on Biomedical Engineering*, vol. 53, no. 10, pp. 1943–1953, 2006.

### Task, dataset and evaluation

- A. Ramalingam, S. Kedari, and C. Vuppalapati, "IEEE FEMH Voice Data Challenge 2018," *2018 IEEE International Conference on Big Data (Big Data)*, 2018. doi:10.1109/BigData.2018.8622164
- S.-H. Fang, Y. Tsao, M.-J. Hsiao, J.-Y. Chen, Y.-H. Lai, F.-C. Lin, and C.-T. Wang, "Detection of pathological voice using cepstrum vectors: A deep learning approach," *Journal of Voice*, vol. 33, no. 5, pp. 634–641, 2019. doi:10.1016/j.jvoice.2018.02.003
- B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion Challenge," *INTERSPEECH*, pp. 312–315, 2009.
- W. J. Youden, "Index for rating diagnostic tests," *Cancer*, vol. 3, no. 1, pp. 32–35, 1950.
- L. R. Rabiner and M. R. Sambur, "An algorithm for determining the endpoints of isolated utterances," *Bell System Technical Journal*, vol. 54, no. 2, pp. 297–315, 1975.

> The four classes, and the file counts `prepare` checks against by default (Normal 50, Neoplasm 40, Phonotrauma 60, Vocal palsy 50), come from the 200 training recordings of the FEMH Voice Data Challenge 2018 dataset. The dataset has to be requested from the organizers and is not distributed with this package.

---

## License

MIT License, see [LICENSE](LICENSE).

Author: RL
