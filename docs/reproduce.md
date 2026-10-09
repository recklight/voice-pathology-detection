# 重現一次實驗

這份文件說明一次執行要怎麼重現，以及改設定時要注意什麼。

---

## 1. 一次執行由什麼決定

| 因素 | 在哪裡 |
| --- | --- |
| 資料 | `paths.data_root` 底下的檔案 |
| 前處理 | `trim`、`mfcc`、`normalization` 區段 |
| 切分 | `split.n_splits`、`split.seed`、`split.group_by`、`split.val_fraction` |
| 模型 | `detection.model`、`disorder.model`、`flat.model` |
| 決策 | `detection.threshold_method`、`detection.target`、`aggregation` |
| 亂數 | 頂層 `seed` |

這些全部寫在同一份 YAML 裡，每次執行都會把實際生效的那一份存進結果資料夾。要重跑過去的實驗，把那個檔案餵回去就好：

```bash
voice-pathology-detection train --route cascade --config results/cascade-20261008-101500/config.yaml
```

---

## 2. 亂數的來源

`set_global_seed(seed)` 一次設定三處：

| 產生器 | 作用範圍 |
| --- | --- |
| `random` | 標準函式庫 |
| `numpy.random`（全域） | 使用全域狀態的呼叫端，包含 scikit-learn 的部分路徑 |
| `torch`（若已安裝） | 選配的 PyTorch 路徑 |

另外，每個會用到亂數的元件都直接收一個種子：

- `subject_folds(..., seed=cfg.split.seed)`：折的切分
- `train_val_split(..., seed=cfg.split.seed + fold_index)`：內層驗證集；加上折編號，讓不同折的內層切分彼此不同但仍可重現
- `build_model(..., seed=cfg.seed)`：模型的初始化與洗牌
- `BernoulliRBM(seed=...)`：預訓練的初始化與 Gibbs 取樣，第 L 層使用 `seed + L`

種子與「哪些產生器被設定了」會一起寫進 `metrics.json` 的 `seed` 欄位。

要確認結果能重現，同一個設定跑兩次再比對：

```bash
voice-pathology-detection train --route detection --run-id repro-a --config configs/detection_dnn.yaml
voice-pathology-detection train --route detection --run-id repro-b --config configs/detection_dnn.yaml
diff <(jq -S .pooled_subject results/repro-a/metrics.json) \
     <(jq -S .pooled_subject results/repro-b/metrics.json)
```

---

## 3. 切分為什麼是這樣切的

切分有三個條件：

1. 切在檔案層級，不是音框層級。一份錄音會產生數十到數百個高度相關的音框，若在音框層級切分，同一個人的音框會同時出現在訓練與評估側，回報的數字會被嚴重高估。
2. 依 `split.group_by` 分組（預設 `subject_id`）。`StratifiedGroupKFold` 讓同一個分組鍵的所有列落在同一折，`check_no_group_leakage()` 每次執行都會再檢查一次。
3. 依標籤分層。各折的類別比例接近整體比例，避免某一折剛好沒有某個類別。

`subject_id` 代表一份錄音，在索引中必須唯一。若你的資料集中一個人有多份錄音，在標註表加一欄 `Subject ID`（或 `Speaker`），它會以 `speaker_id` 帶進索引，再設定 `split.group_by: speaker_id`，這個人的所有錄音就會綁在同一折，內層驗證集也照同樣的分組切（見 [data-format.md](data-format.md) 第 3 節）。

---

## 4. 評估折被碰了幾次

一次。

| 步驟 | 使用的資料 |
| --- | --- |
| 切出內層驗證集 | 訓練折 |
| 配適正規化統計量 | 訓練折的內層訓練部分 |
| 訓練模型 | 訓練折的內層訓練部分 |
| 選擇決策門檻 | 訓練折的內層驗證部分 |
| 選擇訓練輪數（early stopping） | 交給模型的訓練音框中隨機抽出的一小塊 |
| **計算回報的指標** | **評估折** |

內層驗證集是把訓練折依標籤分層、依同樣的分組再切成 k = round(1 / `split.val_fraction`) 份，取其中一份，所以實際保留 1/k 的檔案：0.2 是五分之一，0.3 是三分之一，0.4 與 0.5 都是一半。訓練折裡某一類只有不到 k 組時，k 會降到那個組數，保留的比例就跟著變大。`val_fraction` 大於 0.5 會被拒絕。

門檻選好之後模型不會重新配適。重新配適會讓門檻屬於一個已經不存在的模型，保持一致比多用那一點驗證資料更重要。存給 `predict` 用的 `detection.joblib` 也照這個做法，只在內層訓練部分上訓練。

若資料量小到無法切出合法的內層驗證集（分層 + 分組後某個類別只剩一組，或切出來的驗證集缺了一類），流程會記錄警告並退回使用固定門檻 `0.5`，不會在訓練資料上調門檻。資料集很小時，記得看一下日誌裡有沒有這則警告。

early stopping 由 scikit-learn 的 `MLPClassifier` 自己處理：它從交給它的音框矩陣中隨機抽一小塊當驗證，這些音框和訓練音框來自同一批錄音，所以停止的 epoch 是在熟悉的說話者上決定的，可能稍晚一些。它只影響訓練何時停止，不碰評估折。設 `early_stopping: false` 時不抽驗證音框，停止條件看的是訓練損失：連續 `n_iter_no_change`（預設 5）個 epoch 沒有改善就停，最多跑 `max_iter` 個 epoch。

---

## 5. 快取的行為

特徵快取的鍵包含音檔路徑、大小、修改時間，以及完整的 `MFCCConfig` 與 `TrimConfig`。因此：

- 改動任何前端參數 → 自動產生新的快取項目，不需要手動清除
- 更換或重新產生音檔 → 修改時間改變 → 自動失效
- 只改模型或切分設定 → 快取被重用，實驗跑得更快

需要完全重算時，刪除 `cache/` 即可。

---

## 6. 改變設定時的注意事項

### 換模型

```bash
voice-pathology-detection train --route detection --model svm
voice-pathology-detection train --route detection --model dbn \
    --set detection.model.params='{"hidden_layer_sizes": [200, 200], "pretrain": true}'
```

第一道指令沒有指定參數，`--model` 會把設定檔中屬於前一個模型的 `model.params` 一併清掉，讓 `svm` 使用自己的預設值，留著網路的層數只會讓它報錯。第二道指令明確給了參數，那些值會被保留。

`mean_proba` 聚合需要模型能輸出後驗機率。`linear_svm` 沒有 `predict_proba`，此時會退回以 `decision_function` 的 softmax 近似，那不是校準過的機率，所以 `linear_svm` 建議搭配 `majority_vote`。

### 換正規化

六種設定洩漏資料的風險不一樣：

| 方法 | 有狀態？ | 備註 |
| --- | --- | --- |
| `none`、`cvn_block`、`cmvn_per_utt` | 無 | 不可能洩漏 |
| `cmn`、`cvn`、`cmvn` | 有 | 統計量只在訓練折上估計；`fit` 與 `transform` 分開，評估折只會經過 `transform` |

### 換折數

折數變少，每折拿來訓練的資料也跟著變少：五折時是五分之四，三折只剩三分之二。折數太多則可能讓某個類別在某折中只剩極少樣本；`subject_folds()` 會在這種情況下直接報錯並說明原因，而不是產生一個無效的切分。

### 比較實驗

一張比較表上的每一列應該只差一個變因，也就是種子、折數與特徵都相同。`examples/compare_normalizations.py` 就是這樣做的：特徵只抽一次，之後所有列共用同一批音框與同一組折。

---

## 7. 從頭到尾跑一次實驗

```bash
# 0. 檢查設定是否如預期
voice-pathology-detection info --config configs/detection_dnn.yaml \
    --set paths.data_root=/data/corpus

# 1. 建索引，嚴格檢查各類別檔案數
voice-pathology-detection prepare --config configs/detection_dnn.yaml \
    --set paths.data_root=/data/corpus --strict

# 2. 抽特徵並快取（之後的實驗不必重算）
voice-pathology-detection features --config configs/detection_dnn.yaml \
    --set paths.data_root=/data/corpus

# 3. 第一階段
voice-pathology-detection train --route detection --config configs/detection_dnn.yaml \
    --set paths.data_root=/data/corpus --run-id det-mlp

# 4. 淺層對照組，同樣的折與種子
voice-pathology-detection train --route detection --config configs/baselines.yaml \
    --set paths.data_root=/data/corpus --run-id det-svm

# 5. 有預訓練的 DBN，與 det-mlp 同拓撲
voice-pathology-detection train --route detection --config configs/dbn_pretrained.yaml \
    --set paths.data_root=/data/corpus --run-id det-dbn

# 6. 沒有預訓練的同一個網路（對照）
voice-pathology-detection train --route detection --config configs/dbn_pretrained.yaml \
    --set paths.data_root=/data/corpus \
    --set detection.model.params='{"hidden_layer_sizes": [200,200,200,200,200], "pretrain": false}' \
    --run-id det-dbn-random

# 7. 兩階段串接與單階段四類的對照
voice-pathology-detection train --route cascade --config configs/detection_dnn.yaml \
    --set paths.data_root=/data/corpus --run-id cascade
voice-pathology-detection train --route flat --config configs/flat_dnn.yaml \
    --set paths.data_root=/data/corpus --run-id flat

# 8. 對測試集推論，預測表寫成 results/cascade/predictions_test.xlsx
voice-pathology-detection predict results/cascade --set paths.data_root=/data/corpus
```

比較結果：

```bash
for run in det-mlp det-svm det-dbn det-dbn-random; do
  printf '%-16s ' "$run"
  jq -r '.subject_summary.uar | "UAR \(.mean) +/- \(.std)"' "results/$run/metrics.json"
done
```

---

## 8. 需要回報什麼

要讓別人能檢驗一次結果，至少附上：

1. 設定檔 `config.yaml`，而不只是「用了 5 層 200 個神經元」這種文字描述。
2. 跨折的平均與標準差，不是單一數字。
3. 折數與分組方式。以受試者分組的五折和以音框切分的五折，得到的數字不能互相比較。
4. 主要指標 UAR，並附上靈敏度與特異度。類別不平衡時，單獨的準確率沒有意義。
5. 決策門檻怎麼選出來，以及每折選到什麼值。
6. 受試者層級與音框層級分開報告（`cascade` 只有受試者層級）。

這些都在結果資料夾的 `config.yaml`、`metrics.json` 與 `report.md` 裡，附上這三個檔案就夠了。
