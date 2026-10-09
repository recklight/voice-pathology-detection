# 演算法說明

流程裡用到的每個演算法各占一節：開頭是實作所在的模組，接著是數學式、參數與文獻出處。

---

## 1. 音檔裁切

`voice_pathology_detection.data.trim`

### 1.1 固定邊界裁切（`fixed_edges`）

對取樣率 $f_s$、長度 $N$ 的訊號 $x$：

$$
x_{\text{trim}} = \begin{cases}
x[\,e : N - e\,], & N > \lfloor T_{\min} f_s \rceil \\
x, & \text{否則}
\end{cases}
\qquad e = \lfloor T_{\text{edge}} f_s \rceil
$$

預設 $T_{\text{edge}} = 2$ 秒、$T_{\min} = 4.5$ 秒。加上長度條件，是因為只有夠長的錄音才可能在前後夾雜對話；短提示音本身就只有發聲，不該被切。

裁切後若短於 `min_output_seconds`，這次裁切就不算，保留未裁切的訊號並記錄警告。

### 1.2 能量端點偵測（`energy_vad`）

由 `speechdsp.endpoint_detect` 計算，用的是 Rabiner–Sambur 的雙門檻演算法：音框能量要超過上門檻 ITU 才算進入語音，再從那一點往回找到最後一個高於下門檻 ITL 的音框。兩個門檻都由開頭幾個只有背景雜訊的音框能量與峰值能量算出，所以和錄音的絕對音量無關。

> L. R. Rabiner and M. R. Sambur, "An algorithm for determining the endpoints of isolated utterances," *Bell System Technical Journal*, 54(2):297–315, 1975.

---

## 2. 梅爾頻率倒頻譜係數

`voice_pathology_detection.features.mfcc`（數值運算委派給 `speechdsp.mfcc`）

### 2.1 前處理

預強調，補償語音訊號的 $-6$ dB/octave 頻譜傾斜：

$$
y[n] = x[n] - \alpha\, x[n-1], \qquad \alpha = 0.97
$$

分框：音框長 $T_{\text{frame}} = 30$ ms、框移 $T_{\text{hop}} = 15$ ms（50% 重疊）。每框加漢明窗：

$$
w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{L}\right), \qquad n = 0, 1, \ldots, L-1
$$

分母是 $L$ 而不是 $L-1$，也就是週期型（periodic）漢明窗。做頻譜分析習慣用週期型，它的旁瓣結構和 FFT 的離散頻率格點對得上；分母 $L-1$ 的對稱型多用在 FIR 濾波器設計。$L$ 很大時兩者差異極小，但數值上並不相等。

FFT 長度取不小於音框長度的最小 2 的冪次（44.1 kHz、30 ms → 1323 點 → 2048），音框裡的取樣點一個都不會被丟掉。

### 2.2 梅爾濾波器組

$M = 23$ 個三角濾波器，中心頻率在梅爾刻度上等距：

$$
m(f) = 1127 \ln\left(1 + \frac{f}{700}\right), \qquad
f(m) = 700\left(e^{m/1127} - 1\right)
$$

第 $k$ 個濾波器自 $f_{k}$ 線性上升至中心 $f_{k+1}$，再下降至 $f_{k+2}$，峰值為 1（HTK 慣例），因此濾波器組輸出保持輸入功率頻譜的單位。

### 2.3 離散餘弦轉換與倒頻譜提升

$$
c_k = \sqrt{\frac{2}{M}} \sum_{m=1}^{M} \log(E_m)
      \cos\left(\frac{k\pi}{M}\left(m - \tfrac{1}{2}\right)\right)
$$

正弦倒頻譜提升，平衡各階係數的動態範圍（僅作用於 $k \ge 1$）：

$$
\hat{c}_k = \left(1 + \frac{L}{2}\sin\frac{\pi k}{L}\right) c_k, \qquad L = 22
$$

### 2.4 欄位排列與 $c_0$ 的正規化

輸出遵循 HTK 的 `MFCC_0` 慣例：

```
[c1, c2, ..., c12, c0]
```

第零階係數在最後一欄。HTK 對所有係數（含 $c_0$）都乘上同一個 $\sqrt{2/M}$，正交 DCT-II 對第零階用的卻是 $\sqrt{1/M}$，兩者差 $\sqrt{2}$。`mfcc.py` 用常數 `C0_SCALE` 把這個因子補回來，測試裡也鎖住了這個關係。

### 2.5 頻帶限制

`low_freq` / `high_freq` 以 4 階 Butterworth 濾波器經 `scipy.signal.sosfiltfilt` 做零相位濾波，在進入前端之前施加。預設 `low_freq = 64` Hz、`high_freq` 不設，也就是只有 64 Hz 的高通；在設定檔裡把某一側設成 `null` 就關掉那一側。零相位是必要的：音框是在濾波後的訊號上切出來的，相位失真會改變音框內容。訊號太短、沒辦法做零相位濾波時，會原樣通過並記錄警告。

濾波前會先檢查波形：只要有一個 NaN 或無限大的取樣值，`sosfiltfilt` 就會把它擴散到整段訊號，所以這種錄音直接被拒絕，由 `extract_corpus` 跳過，日誌記下跳過的份數與前三個檔名。

> S. B. Davis and P. Mermelstein, *IEEE Trans. ASSP*, 28(4):357–366, 1980.
> S. Young et al., *The HTK Book (v3.4)*, 2006.

---

## 3. 動態特徵

`voice_pathology_detection.features.deltas`（數值運算委派給 `speechdsp.delta`）

回歸式差分：

$$
d_t = \frac{\sum_{n=1}^{N} n \,(c_{t+n} - c_{t-n})}{2\sum_{n=1}^{N} n^2},
\qquad N = \frac{\text{width} - 1}{2}
$$

- 一階（速度）：`delta_width = 7`，即 $N = 3$，分母 $2 \times 14 = 28$
- 二階（加速度）：`accel_width = 5`，即 $N = 2$，分母 $2 \times 5 = 10$，且是**對一階差分再做一次**回歸，而非對靜態係數取二階差分

邊界以首尾音框複製補足，因此輸出的音框數與輸入相同。

最終 39 維排列：

```
[c1..c12, c0 | Δc1..Δc12, Δc0 | ΔΔc1..ΔΔc12, ΔΔc0]
  靜態 13 維     一階 13 維        二階 13 維
```

> S. Furui, *IEEE Trans. ASSP*, 34(1):52–59, 1986.

---

## 4. 倒頻譜正規化

`voice_pathology_detection.features.normalize`

### 4.1 區塊變異數正規化（`cvn_block`）

這是逐音框的運算。把第 $t$ 個音框的 39 維向量看成 3 個 13 維區塊 $B_{t,b}$（$b \in \{0,1,2\}$ 分別對應靜態、一階、二階）：

$$
\hat{B}_{t,b} = \frac{B_{t,b}}{\sigma(B_{t,b}) + \epsilon},
\qquad \sigma(B) = \sqrt{\frac{1}{13}\sum_{i=1}^{13}\left(B_i - \bar{B}\right)^2}
$$

三個區塊為 `0:13 / 13:26 / 26:39`。除數預設為標準差（保持無因次）；設定 `divisor: var` 可改為變異數。

它完全不跨音框，所以沒有狀態，也不可能把資訊從一折帶到另一折；一份錄音正規化的結果，和它跟哪些錄音放在同一個資料集無關。

### 4.2 沿時間軸的正規化（`cmn` / `cvn` / `cmvn`）

文獻上標準的倒頻譜均值／變異數正規化，對第 $j$ 個係數：

$$
\hat{c}_{t,j} = \frac{c_{t,j} - \mu_j}{\sigma_j}
$$

其中 $\mu_j, \sigma_j$ 只在訓練折的音框上估計，再原樣套用到驗證與測試折。實作上包了一層 `sklearn.preprocessing.StandardScaler`，`fit` 與 `transform` 是分開的兩步。

### 4.3 逐錄音正規化（`cmvn_per_utt`）

公式相同，但統計量取自每份錄音自己，用來去掉那一次錄音的通道效應（麥克風、房間）。它同樣沒有狀態。

> S. Furui, *IEEE Trans. ASSP*, 29(2):254–272, 1981.
> O. Viikki and K. Laurila, *Speech Communication*, 25(1–3):133–147, 1998.

---

## 5. 受限波茲曼機與對比散度

`voice_pathology_detection.models.rbm`

可見層 $v$（Bernoulli，輸入須在 $[0,1]$）、隱藏層 $h$，能量定義為：

$$
E(v, h) = -v^\top W h - b^\top v - c^\top h
$$

條件機率：

$$
P(h_j = 1 \mid v) = \sigma\!\left(\sum_i v_i W_{ij} + c_j\right), \qquad
P(v_i = 1 \mid h) = \sigma\!\left(\sum_j W_{ij} h_j + b_i\right)
$$

CD-$k$ 更新（$\langle\cdot\rangle_{\text{data}}$ 為資料期望、$\langle\cdot\rangle_{\text{model}}$ 為 $k$ 次 Gibbs 取樣後的模型期望）：

$$
\Delta W_{ij} \propto \langle v_i h_j\rangle_{\text{data}} - \langle v_i h_j\rangle_{\text{model}}
$$

重建時，由資料驅動的隱藏狀態先取樣成 0/1，每個隱藏單元最多只傳一個位元，有正則化的效果；重建出來的可見層與所有統計量（包括正相位）則用機率，梯度的變異數比較小。權重初始化為標準差 0.01 的零均值高斯分布（即 $\mathcal{N}(0, 0.01^2)$），偏置初始化為零（Hinton 的指南建議可見層偏置取 $\log[p_i/(1-p_i)]$，這裡兩層都從零開始）。

> G. E. Hinton, *Neural Computation*, 14(8):1771–1800, 2002.
> G. E. Hinton, "A practical guide to training restricted Boltzmann machines," UTML TR 2010-003, 2010.

---

## 6. 最大熵乘子的牛頓法求解

`voice_pathology_detection.models.maxent`

對比散度之外的另一種更新規則。把參數視為帶高斯先驗（變異數 $\sigma^2$）的最大熵問題的拉格朗日乘子，求解穩定條件（Chen 與 Rosenfeld 在高斯先驗下的迭代縮放更新）：

$$
f(\gamma) = F\sigma^2 e^{\gamma S} + \gamma + \lambda - \mu\sigma^2 = 0
$$

$$
f'(\gamma) = F S \sigma^2 e^{\gamma S} + 1
$$

其中 $F$ 為該特徵的模型期望、$\mu$ 為資料期望、$\lambda$ 為現有參數值、$S = \sum_i F_i$ 為純量耦合項。牛頓迭代：

$$
\gamma \leftarrow \gamma - \eta \frac{f(\gamma)}{f'(\gamma)}
$$

收斂條件為 $\sum_i |f(\gamma_i)| < \text{tol}$（預設 $10^{-3}$）。導數為 1 加上一個隨 $F$ 消失的項，因此對任何實際的統計量都遠離零。指數的引數經 clip 以避免溢位。未收斂時只記錄 debug 訊息，不拋出例外：最後一次迭代仍是一個下降方向，為了單一 mini-batch 收斂較慢就中止訓練並不划算。

先驗會把大幅的更新往零拉，可以壓住對比散度在前幾個 epoch 的劇烈震盪。

> E. T. Jaynes, *Physical Review*, 106(4):620–630, 1957.（最大熵原理）
> S. F. Chen and R. Rosenfeld, "A survey of smoothing techniques for ME models," *IEEE Trans. Speech and Audio Processing*, 8(1):37–50, 2000.（高斯先驗下的更新式）

---

## 7. 深度信念網路

`voice_pathology_detection.models.dbn`

1. 特徵以 `MinMaxScaler` 映射到 $[0,1]$（Bernoulli 可見單元的要求）。
2. 逐層預訓練：第 $\ell$ 層的受限波茲曼機以第 $\ell-1$ 層的隱藏層**機率**為輸入，訓練完後保留權重 $W^{(\ell)}$ 與隱藏偏置 $c^{(\ell)}$。層與層之間傳的是機率而不是取樣值，下一層的輸入比較平滑，這也是逐層預訓練常見的做法。
3. 展開成前饋網路，加上 softmax 輸出層。預訓練權重寫入隱藏層；輸出層沒有非監督式的對應物，保留隨機初始化。
4. 以反向傳播微調整個網路（`sklearn.neural_network.MLPClassifier`，sigmoid 活化、SGD、交叉熵）。

`pretrain: false` 保留隨機初始化，提供同拓撲、同種子的對照組。

> G. E. Hinton, S. Osindero, and Y.-W. Teh, *Neural Computation*, 18(7):1527–1554, 2006.
> G. E. Hinton and R. R. Salakhutdinov, *Science*, 313(5786):504–507, 2006.

---

## 8. 高斯混合模型分類器

`voice_pathology_detection.models.gmm`

每個類別 $k$ 各自配適一個高斯混合模型，以貝氏定理合成後驗：

$$
\log P(k \mid x) \propto \log p(x \mid k) + \log \pi_k
$$

其中 $p(x \mid k) = \sum_{m=1}^{M} w_{km}\, \mathcal{N}(x; \mu_{km}, \Sigma_{km})$。

共變異數預設為對角矩陣：參數量遠少於完整矩陣，而離散餘弦轉換已經移除係數之間大部分的相關性。類別先驗 $\pi_k$ 可取訓練頻率（`empirical`）或均勻（`uniform`）；類別不平衡而以 UAR 評分時，均勻先驗較合適。

> J. I. Godino-Llorente et al., *IEEE Trans. Biomedical Engineering*, 53(10):1943–1953, 2006.
> D. A. Reynolds and R. C. Rose, *IEEE Trans. Speech and Audio Processing*, 3(1):72–83, 1995.

---

## 9. 音框到受試者的聚合

`voice_pathology_detection.eval.aggregate`

設第 $i$ 份錄音貢獻音框集合 $\mathcal{F}_i$，音框 $t$ 的後驗為 $p_t \in \mathbb{R}^K$。

**平均後驗（`mean_proba`，預設）**

$$
\bar{p}_i = \frac{1}{|\mathcal{F}_i|}\sum_{t \in \mathcal{F}_i} p_t,
\qquad \hat{y}_i = \arg\max_k \bar{p}_{i,k}
$$

**多數決（`majority_vote`）**

$$
r_{i,k} = \frac{1}{|\mathcal{F}_i|}\left|\{t \in \mathcal{F}_i : \arg\max_j p_{t,j} = k\}\right|,
\qquad \hat{y}_i = \arg\max_k r_{i,k}
$$

**比例門檻（`ratio_threshold`，二元任務專用）**

$$
\hat{y}_i = \begin{cases}
1, & r_{i,1} \ge \tau \\
0, & \text{否則}
\end{cases}
$$

$\tau$ 不可為負，但可以大於 1，代表沒有人被判為病理。門檻搜尋選在「全部判為陰性」那一點時，給的是略高於最大分數的值；選門檻的資料裡只要有一位受試者的音框全部判為病理，最大比例就是 1，這個門檻也就略大於 1（見第 10 節）。

調門檻時，比較的分數跟著聚合規則走：`mean_proba` 用 $\bar{p}_{i,1}$，`majority_vote` 與 `ratio_threshold` 用 $r_{i,1}$。選門檻與套用門檻用的是同一種分數。

實作上以 `np.add.reduceat` 對所有群組一次完成分段求和，不使用逐檔迴圈。音框數為零的群組會直接被拒絕，因為 `reduceat` 遇到空群組會悄悄拿相鄰那一列的值，算錯了也看不出來。

---

## 10. 決策門檻選擇

`voice_pathology_detection.eval.threshold`

**Youden 指標**

$$
J(\tau) = \text{sensitivity}(\tau) + \text{specificity}(\tau) - 1,
\qquad \tau^\star = \arg\max_\tau J(\tau)
$$

**指定操作點**：`target_sensitivity` 取滿足 $\text{sensitivity}(\tau) \ge s^\star$ 的門檻中，特異度最高者；`target_specificity` 對稱。目標無法達成時，退回該軸能達到的最佳值並記錄警告。

候選門檻來自 `sklearn.metrics.roc_curve`。ROC 曲線的第一列是「全部判為陰性」的退化點，scikit-learn 1.3 以後給的門檻是無限大，更早的版本（CI 最舊依賴組合用的 1.2）給最大分數加 1。這個操作點是合法的，所以保留下來，但門檻一律換成最大分數之上的下一個浮點數，讓結果能被序列化與繪圖。分數是音框比例時，要有某位受試者的音框全部判為病理、最大分數等於 1，這個值才會略大於 1。

在選門檻的資料上，最低的判陽分數 $s_+$ 與最高的判陰分數 $s_-$ 之間，任何一個 $\tau \in (s_-, s_+]$ 都給出相同的靈敏度與特異度。`roc_curve` 給的是區間的上端 $s_+$，新受試者只要分數稍低就會被判為陰性，所以實際回傳的是中點 $(s_- + s_+)/2$。沒有任何分數被判為陽性時，門檻維持在最大分數之上。

選門檻的資料是訓練折再切出的內層驗證集：把訓練折依標籤分層、依 `split.group_by` 分組再切成 $k = \operatorname{round}(1/\text{val\_fraction})$ 份，取其中一份。因此實際保留的比例是 $1/k$（0.2 → 1/5，0.3 → 1/3，0.4 與 0.5 → 1/2），`val_fraction` 不可大於 0.5。訓練折裡某一類只有不到 $k$ 組時，$k$ 會降到那個組數，保留的比例就跟著變大。切不出兩類都有的驗證集時，該折改以固定門檻 0.5 判斷並記錄警告。

> W. J. Youden, *Cancer*, 3(1):32–35, 1950.

---

## 11. 兩階段串接

`voice_pathology_detection.routes.cascade`

在單一折內：

1. 以訓練檔訓練偵測器，門檻由訓練側再切出的內層驗證集決定。
2. 以訓練檔中真正為病理的錄音訓練三類分類器。訓練標籤是已知的，沒有理由餵給它偵測器的錯誤。
3. 對評估檔先跑偵測器，只有被放行的受試者才進入三類分類器。
4. 合併成 $0..3$ 的單一標籤。

被偵測器誤放行的正常受試者一定會被指派某個病變類別（必然錯誤）；被誤判為正常的病理受試者永遠不會有機會被分類。兩種錯誤都是串接系統誤差的一部分，也都會出現在合併後的四類混淆矩陣中，所以代表整個系統的是四類指標。四類結果的靈敏度與特異度把三種病變合在一起當陽性，和偵測階段的定義相同。

---

## 12. 分數層融合

`voice_pathology_detection.fusion.score_fusion`

**凸組合網格搜尋**：權重非負且和為 1，以星與條（stars and bars）列舉單體上的規則網格：

$$
\hat{s} = \sum_{m} w_m s_m, \qquad w_m \ge 0, \quad \sum_m w_m = 1
$$

網格點數為 $\binom{d + m - 1}{m - 1}$，其中 $m$ 為系統數、$d = 1/\text{step}$。

**Stacking**：以邏輯迴歸學習組合。權重不受約束（可為負），因此一個系統可以修正另一個，而不只是稀釋它；代價是組合器本身也是一個配適出來的模型，訓練它的分數必須來自兩個系統訓練時都沒用過的資料。

---

## 13. 評估指標

`voice_pathology_detection.eval.metrics`

| 指標 | 定義 |
| --- | --- |
| UAR | $\frac{1}{K}\sum_{k} \text{recall}_k$，$K$ 為 `y_true` 中出現的類別數 |
| Sensitivity | $TP / (TP + FN)$ |
| Specificity | $TN / (TN + FP)$ |
| PPV | $TP / (TP + FP)$ |
| NPV | $TN / (TN + FN)$ |
| Macro F1 | 各類別 F1 的算術平均 |
| ROC-AUC | 二元直接計算；多類別以 one-vs-rest 巨觀平均 |

UAR 是本任務的主要指標：正常 50 例、病理 150 例的不平衡下，準確率會獎勵「一律猜多數類」的模型，UAR 不會。只在預測中出現、真實標籤裡沒有的類別不計入 $K$，所以 UAR 與 `sklearn.metrics.balanced_accuracy_score` 相同；`per_class_recall` 仍會列出這種類別，召回率記為 0。

靈敏度、特異度、PPV 與 NPV 以病理為陽性。四類任務（`flat` 與 `cascade` 的合併結果）把三種病變合在一起當陽性；`disorder` 沒有正常類別，這四項不定義，以 `nan` 表示（寫進 `metrics.json` 時為 `null`）。

靈敏度與特異度以「真實類別」為條件，PPV 與 NPV 以「模型的預測」為條件。後者依賴盛行率，只能在類別比例相同的實驗之間比較。

跨折彙總時，未定義（`nan`）的項目會被跳過而不是污染平均值，並回報實際參與平均的折數。

> B. Schuller, S. Steidl, and A. Batliner, *INTERSPEECH*, pp. 312–315, 2009.
