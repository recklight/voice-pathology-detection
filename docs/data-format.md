# 資料格式

程式讀進來和寫出去的每一種檔案都在這裡：格式、欄位意義與標籤編碼。

---

## 1. 輸入音檔

| 項目 | 說明 |
| --- | --- |
| 格式 | WAV（PCM，未壓縮） |
| 聲道 | 單聲道；多聲道會自動平均成單聲道 |
| 位元深度 | 8 / 16 / 24 / 32 位元整數，或 32 / 64 位元浮點數 |
| 取樣率 | 與 `mfcc.sample_rate` 一致（預設 44100 Hz）；不一致時會記錄警告並以實際取樣率分析 |
| 內容 | 持續母音 `/a:/` |
| 長度 | 不限。短於一個音框（預設 30 ms）的錄音會被跳過，日誌記下跳過的份數與前三個檔名 |
| 取樣值 | 必須是有限值；含 NaN 或無限大的錄音同樣會被跳過 |

讀檔由 `speechdsp.read_wav` 處理，回傳 `float64` 單聲道訊號（標稱範圍 `[-1, 1]`）與取樣率。

---

## 2. 資料夾結構

### 方式一：一個診斷一個資料夾

```
<paths.data_root>/
├── <paths.train_dir>/          預設 train
│   ├── Normal/                 001.wav 002.wav ...
│   ├── Neoplasm/
│   ├── Phonotrauma/
│   └── Vocal palsy/
└── <paths.test_dir>/           預設 test
    └── *.wav
```

診斷資料夾可以有任意深度的上層資料夾（例如 `Pathological/Neoplasm/`），索引時會取**最深的**一個可辨識名稱。可辨識的名稱（不分大小寫，底線與空格等價）：

| 別名 | 對應編碼 |
| --- | --- |
| `normal`、`healthy` | 0 |
| `neoplasm` | 1 |
| `phonotrauma` | 2 |
| `vocal palsy`、`vocal_palsy`、`vocalpalsy`、`palsy` | 3 |

訓練集中若有音檔不在任何可辨識的診斷資料夾底下，索引會直接失敗並列出前幾個檔名，不會靜默跳過。

### 方式二：扁平資料夾 + 標註表

```
<paths.data_root>/
├── train/                      001.wav 002.wav ...
├── test/                       *.wav
└── Demographics.xlsx
```

設定 `paths.demographics: Demographics.xlsx`。

---

## 3. 標註表（xlsx / csv / tsv）

欄位名稱以不分大小寫的別名比對：

| 檔案中的欄位名 | 解析為 | 必要 |
| --- | --- | --- |
| `File name`、`filename`、`file` | `file_name` | 是 |
| `Diagnosis`、`Diagnosis(0-3)`、`label`、`class` | `label` | 是 |
| `Gender`、`Sex` | `gender` | 否 |
| `Age` | `age` | 否 |
| `Subject`、`Subject ID`、`Speaker`、`Speaker ID` | `speaker_id` | 否 |

「解析為」的每個欄位只能對到表格裡的一欄。表格裡同時有 `Speaker` 和 `Subject ID` 時，讀取會直接報錯並列出這兩欄，留其中一欄即可。

表格以文字讀入，`001` 這類檔名會保留前導零；檔名可含或不含副檔名。`label` 欄位可以是 `0`–`3` 的數字（寫成 `2.0` 也可以），也可以是類別名稱（`Vocal palsy` 等）。

若有任何錄音在標註表中找不到對應的診斷，或診斷欄是空白的，`build_index()` 會拋出 `DatasetError` 並列出那些檔名。

說話者欄位用在一個人有多份錄音的資料集。它會以 `speaker_id` 帶進索引，再設定 `split.group_by: speaker_id`，同一個人的錄音就會落在同一折，內層驗證集也照這個分組切。表格沒給說話者的錄音，`speaker_id` 就是它自己的 `subject_id`，自成一組。

---

## 4. 檔案索引（`pandas.DataFrame`）

`build_index()` 與 `scan_class_directories()` 回傳這張索引，後面每個階段要找檔案、標籤與分組，都只看它。

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `path` | str | 音檔的完整路徑（POSIX 格式） |
| `file_name` | str | 檔名，含副檔名 |
| `subject_id` | str | 錄音的識別碼，取自相對於該 split 資料夾（`paths.data_root` 下的 `train_dir`／`test_dir`）的路徑（去副檔名），也是預設的分組鍵。**必須唯一**，否則索引會被拒絕 |
| `label` | int | 四類編碼 `0..3`；未標註為 `-1` |
| `label_name` | str | 類別顯示名稱；未標註為 `Unknown` |
| `label_binary` | int | 二元編碼 `0`/`1`；未標註為 `-1` |
| `split` | str | `train` 或 `test` |
| `n_frames` | int | 該錄音貢獻的音框數（抽完特徵後才填入） |
| `speaker_id` | str | 說話者識別碼，只在標註表有說話者欄位時出現 |

`subject_id` 之所以取相對路徑而非單純的檔名主幹，是因為 `001.wav` 這種檔名會在每個診斷資料夾中重複出現。它代表一份錄音：預測表與 `predictions.csv` 都是一份錄音一列。一個人有多份錄音時，分組就用 `speaker_id`（見上一節）。

索引可以直接寫成 CSV 供人工檢視：

```bash
voice-pathology-detection prepare --output results/index_train.csv
```

---

## 5. 特徵矩陣

### 5.1 記憶體中：`FeatureSet`

`voice_pathology_detection.features.pipeline.FeatureSet` 綁定三件必須保持一致的東西：

| 屬性 | 形狀 | 說明 |
| --- | --- | --- |
| `X` | `(n_frames, n_dim)` | 音框特徵矩陣 |
| `frame_file` | `(n_frames,)` | 每個音框來自 `files` 的第幾列 |
| `files` | `(n_files, ...)` | 檔案索引 |

建構時會驗證：列數相符、`frame_file` 不超出索引範圍、且**單調不遞減**（同一份錄音的音框必須連續且依檔案順序排列）。`group_sizes` 能當成合法的分段，靠的就是這個單調性。`files` 若已有 `n_frames` 欄，必須與 `frame_file` 中每份錄音的音框數相符，否則直接報錯；沒有這一欄時由 `frame_file` 算出來補上。

常用衍生屬性：

| 屬性 | 說明 |
| --- | --- |
| `group_sizes` | 每份錄音的音框數，依檔案順序 |
| `file_labels` / `frame_labels` | 每份錄音 / 每個音框的標籤 |
| `subjects` / `frame_subjects` | 每份錄音 / 每個音框的受試者識別碼 |
| `select(rows)` | 取子集，並重新編號 `frame_file`；位置會先排序，子集裡的檔案維持索引中的先後順序 |

### 5.2 落檔：`.npz` + `.index.csv`

`FeatureSet.save(path)` 寫出兩個檔案：

| 檔案 | 內容 |
| --- | --- |
| `<name>.npz` | 壓縮的 `X` 與 `frame_file` |
| `<name>.index.csv` | 檔案索引 |

`FeatureSet.load(path)` 需要兩個檔案都在，少了哪一個，錯誤訊息會指名。讀回索引時，`path`、`file_name`、`subject_id`、`speaker_id` 等識別碼欄位一律當文字讀，`001` 不會變成 `1`。

### 5.3 特徵快取：`.npy`

`extract_file(..., cache_dir=...)` 會把每份錄音的特徵矩陣存成一個 `.npy`，檔名是以下內容的 SHA-256 前 32 個十六進位字元：

- 快取格式版本
- 音檔的絕對路徑、位元組大小、修改時間
- 完整的 `MFCCConfig` 與 `TrimConfig`

所以改了任何前端參數或音檔本身，都會對到新的快取項目，不必手動清除。

### 5.4 欄位語意

39 維特徵的欄位排列：

```
索引  0..11   c1..c12          靜態倒頻譜係數
索引  12      c0               靜態第零階（注意在區塊的最後一欄）
索引  13..24  Δc1..Δc12        一階差分
索引  25      Δc0
索引  26..37  ΔΔc1..ΔΔc12      二階差分
索引  38      ΔΔc0
```

`use_deltas: false` 時只有前 13 欄。

---

## 6. 標籤編碼

| 編碼 | 四類任務 | 二元任務 | 常數 |
| --- | --- | --- | --- |
| 0 | Normal | Normal | `VoiceClass.NORMAL` / `BinaryClass.NORMAL` |
| 1 | Neoplasm | Pathological | `VoiceClass.NEOPLASM` |
| 2 | Phonotrauma | Pathological | `VoiceClass.PHONOTRAUMA` |
| 3 | Vocal palsy | Pathological | `VoiceClass.VOCAL_PALSY` |

二元映射：`0 → 0`，`{1, 2, 3} → 1`（`voice_pathology_detection.data.labels.to_binary`）。

各階段使用的標籤集合：

| 階段 | 標籤 |
| --- | --- |
| `detection` | `(0, 1)` |
| `disorder` | `(1, 2, 3)` |
| `flat` / `cascade` 的最終輸出 | `(0, 1, 2, 3)` |

參考語料的各類別檔案數為 Normal 50、Neoplasm 40、Phonotrauma 60、Vocal palsy 50，共 200 份。`build_index(strict_counts=True)` 會據此檢查。你的資料集組成不同時，在 Python API 傳入自己的 `expected_counts`；命令列預設的 `--no-strict` 只記錄警告。

---

## 7. 輸出格式

### 7.1 執行結果資料夾

```
results/<run_id>/
├── config.yaml              實際生效的完整設定
├── metrics.json             所有指標
├── report.md                可讀摘要
├── predictions.csv          逐受試者預測
├── folds_subject.csv        逐折的受試者層級指標   （單階段路線）
├── folds_frame.csv          逐折的音框層級指標     （單階段路線）
├── threshold_sweep.csv      門檻掃描               （二元階段）
├── detection.joblib         供推論的模型與門檻     （detection / cascade）
├── disorder.joblib          供推論的模型           （disorder / cascade）
├── flat.joblib              供推論的模型           （flat）
└── figures/*.png
```

前四項每條路線都有；其餘隨路線而異。`--no-save-models` 不寫 `.joblib`；`--no-figures` 不畫 `figures/` 裡的圖，`threshold_sweep.csv` 仍會寫出。

`detection.joblib` 的門檻要在沒參與訓練的檔案上選，所以模型只在內層切分的訓練部分上訓練（`split.val_fraction` 為 0.2 時通常是五分之四的檔案，某一類的組數少於 5 時會再少一些），門檻在其餘的檔案上選；門檻設成 `fixed` 時用全部檔案。`disorder.joblib` 與 `flat.joblib` 用全部訓練檔案。

### 7.2 `predictions.csv`

單階段路線（`detection` / `disorder` / `flat`）：

| 欄位 | 說明 |
| --- | --- |
| `fold` | 該受試者被評估時所在的折 |
| `subject_id` | 受試者識別碼 |
| `y_true` / `y_pred` | 真實 / 預測標籤 |
| `score` | 正類別的連續分數 |
| `proba_<label>` | 每個類別一欄的平均後驗機率 |

兩階段串接（`cascade`）另有：

| 欄位 | 說明 |
| --- | --- |
| `y_true_binary` / `y_pred_binary` | 第一階段的真實 / 預測 |
| `forwarded` | 該受試者是否被送進第二階段 |

每位受試者在整份 `predictions.csv` 中恰好出現一次（out-of-fold 預測）。

### 7.3 `metrics.json`

以 JSON 儲存，NumPy 數值與陣列會被轉成原生型別。未定義的數值（`nan`、`inf`）寫成 `null`，所以 `jq` 或 JavaScript 這類嚴格的 JSON 解析器可以直接讀。

單階段路線（`detection` / `disorder` / `flat`）的鍵：

| 鍵 | 內容 |
| --- | --- |
| `task`、`labels`、`n_folds` | 執行的基本資訊 |
| `subject_summary`、`frame_summary` | 各指標的 `mean` / `std` / `n`（參與平均的折數） |
| `pooled_subject`、`pooled_frame` | 全部 out-of-fold 預測合併後算一次的指標與混淆矩陣（每位受試者、每個評估音框各算一次） |
| `folds` | 逐折的完整報表與該折選出的門檻 |
| `thresholds` | 每折的門檻值 |
| `config` | 該次執行的模型與前處理設定 |
| `seed` | 種子與被設定的產生器 |

`cascade` 的結構不同，它有三個視角，而不是一個：

| 鍵 | 內容 |
| --- | --- |
| `task`、`labels`、`label_names`、`n_files`、`n_frames` | 執行的基本資訊 |
| `detection_summary`、`disorder_summary`、`overall_summary` | 三個視角的跨折 `mean` / `std` / `n` |
| `pooled_detection`、`pooled_disorder`、`pooled_overall` | 對應的合併指標與混淆矩陣 |
| `forwarded_per_fold` | 每折被送進第二階段的受試者數 |
| `thresholds`、`config`、`seed` | 同上 |

也就是說 `cascade` 沒有 `subject_summary` / `frame_summary` / `pooled_subject` / `pooled_frame` / `folds` / `n_folds`，只有受試者層級的指標；要引用整個系統的表現，看 `overall_summary`。

所有路線的 `sensitivity`、`specificity`、`ppv`、`npv` 都以病理為陽性：四類的 `flat` 與 `cascade` 的 `overall_*` 把三種病變合在一起當陽性，`disorder` 與 `cascade` 的 `disorder_*` 沒有正常類別，這四項為 `null`。

### 7.4 預測表

`predict` 以 `voice_pathology_detection.prediction_table` 寫出兩欄：

| 欄位 | 內容 |
| --- | --- |
| `File name` | 檔案識別碼，**以文字型別寫入** |
| `Prediction` | `0`–`3` 的整數 |

識別碼刻意以文字寫入：`001` 這類編號一旦被試算表當成數字，前導零就會消失，表格就對不回原本的錄音。支援 `.xlsx`（需要 `openpyxl`）與 `.csv`，依副檔名決定；未指定 `--output` 時寫成結果資料夾裡的 `predictions_<split>.xlsx`。
