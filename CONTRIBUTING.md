# 參與開發

歡迎修正錯誤或補上功能，送 pull request 前請先跑過下面的檢查。

---

## 開發環境

需要 Python 3.10 以上，建議開一個虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 安裝本專案

用可編輯模式安裝，順便裝上開發工具。共用的訊號處理函式來自 PyPI 上的
[`speechdsp`](https://pypi.org/project/speechdsp/)，pip 會一併裝好：

```bash
pip install -e ".[dev]"
```

確認命令列工具可用：

```bash
voice-pathology-detection --help
```

---

## 提交前的檢查

下面三項都要通過，CI 跑的是同一組檢查：

```bash
ruff check .           # 靜態檢查
ruff format --check .  # 格式（確認沒有未套用的變更；要看差異就加 --diff）
pytest -q              # 測試
```

### Python 3.10 相容性

開發機上的 Python 可能比較新，但程式碼要能在 3.10 上跑，
所以請避開 3.11 以後才有的語法與標準函式庫：

| 不要用 | 可用 |
| --- | --- |
| `typing.Self` | 類別名稱字串或 `TypeVar` |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| PEP 695 泛型（`class Foo[T]`、`type X = ...`） | `TypeVar` 與 `TypeAlias` |
| `itertools.batched` | 自行實作切片 |
| `tomllib` | `tomli`，或乾脆不讀 TOML |
| `asyncio.TaskGroup` | `asyncio.gather` |
| `except*`／`ExceptionGroup` | 一般的 `except` |

CI 的語法閘門會用 `ast.parse(..., feature_version=(3, 10))` 逐檔檢查，用到新語法就會失敗。
本機可以先自己跑一次（只掃 `src/`、`tests/` 與 `examples/`，`.venv` 裡的第三方套件不用掃）：

```bash
python -c "import ast,pathlib; [ast.parse(f.read_text(encoding='utf-8'), filename=str(f), feature_version=(3,10)) for d in ('src','tests','examples') for f in pathlib.Path(d).rglob('*.py')]"
```

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式要有 type hints 與英文的 numpy 風格 docstring；實作的是某篇文獻的方法，就在 `References` 段寫出處。
- README 有兩份：`README.md` 是英文，`README.zh-TW.md` 是繁體中文，改了其中一份就要同步另一份。其他給使用者看的文件（`docs/`、這份文件）用繁體中文寫。
- 診斷訊息用 `logging` 輸出，不要用 `print`。
- 陣列運算盡量向量化，不寫逐元素的 Python 迴圈。

---

## 資料

這個 repository 裡沒有任何實驗資料，測試一律用程式合成的訊號當 fixture。
請不要提交音檔、`.mat`、`.npy`、模型權重或任何含個人資訊的檔案。
`.gitignore` 擋掉了常見的副檔名，提交前還是請自己再看一次。

---

## 提交訊息

建議用 [Conventional Commits](https://www.conventionalcommits.org/) 的格式：

```
feat: 新增 xxx 功能
fix: 修正 xxx 在 yyy 情況下的錯誤
docs: 更新 xxx 說明
test: 補上 xxx 的測試
refactor: 重構 xxx
```

有功能變動時，請一併更新 `CHANGELOG.md` 的 `[Unreleased]` 區塊。
