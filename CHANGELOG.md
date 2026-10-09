# 變更紀錄

本檔案記錄 `voice-pathology-detection` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### 新增

- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖也有英文版：`docs/workflow.html` 與 `docs/images/workflow.png` 是英文，中文版在 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。

---

## [0.1.0] - 2026-10-09

首次釋出。

### 新增

- 從持續母音 `/a:/` 的錄音偵測病理嗓音（正常 vs 病理），並判別 Neoplasm、Phonotrauma、Vocal palsy 三種病變類別；提供兩階段串接（`cascade`）與單階段四類（`flat`）兩條可對照的路線。
- 前端：音檔裁切（`fixed_edges`／`energy_vad`）、HTK `MFCC_0` 排列的 13 維倒頻譜係數、回歸式一階與二階差分（共 39 維），以及 `none`、`cvn_block`、`cmn`、`cvn`、`cmvn`、`cmvn_per_utt` 六種正規化設定。
- 模型：sigmoid 多層感知器、以受限波茲曼機逐層預訓練的深度信念網路（另有選配的 PyTorch 版本）、支援向量機與高斯混合模型對照組；受限波茲曼機可選用對比散度或最大熵更新規則。
- 評估：以受試者分組的分層 k 折交叉驗證（標註表有說話者欄位時可改以說話者分組）、只在訓練折內決定的決策門檻、音框與受試者層級雙軌報告，以及多系統分數融合（權重網格搜尋與 stacking）。
- 命令列介面（`info`、`prepare`、`features`、`train`、`predict`、`spectrograms`、`demo`）與 Python API；所有設定集中在 dataclass，可由 YAML 載入並以 `--set` 覆寫。
- 共用的訊號處理與評估函式來自 PyPI 上的 `speechdsp`。
- 435 項測試（354 個單元測試與 81 個 docstring 範例），全部使用程式合成的資料。
- GitHub Actions CI：Ubuntu／Windows × Python 3.10–3.13，含 3.10 語法閘門、ruff 與 pytest，另有一個以最低相依版本執行測試的 job。
- 繁體中文說明文件（README、`docs/`、CONTRIBUTING）與 CITATION.cff。
