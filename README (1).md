# FD-VQA

FD-VQA 提供一條龍的 **Video Quality Assessment (VQA)** 流程：  
1) 先在 `data/` 內依各資料集執行 **info** 產生索引與統計  
2) 使用 **FD-VQA extract** 提取特徵  
3) 使用 **FD-VQA test** 訓練／評估分數  
4) 若為大型資料集（如 LSVQ、YouTube‑UGC 全量），改用 **DATALAZY**（分批／快取／串流）

> ⚠️ 專案中兩個主程式檔名含空白：`FD-VQA extract.py`、`FD-VQA test.py`。  
> 執行指令時請**加引號**（例如 `python "FD-VQA extract.py"`）。

---

## 📦 目前倉庫重點檔案

- `DATALAZY.py`
- `FD-VQA extract.py`
- `FD-VQA test.py`
- `data/`
  - `data.py`、`data_info_maker.m`
  - 多個資料集 **MATLAB 產生器**（例如：`Kon.m`, `LIVE_Qualcomm.m`, `CVD2014.m`, `LSVQ.m`）
  - 多個已生成的 **info 檔**（例如：`KoNViD-1kinfo.mat`, `LIVE-Qualcomm_info.mat`, `CVD2014info.mat`, `LSVQ1080p_info.mat`, `LIVE-VQCinfo.mat` …）

---

## 🔧 環境需求

- Python ≥ 3.9
- 建議套件：`torch`, `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `tqdm`, `matplotlib`
- （如需抽幀）安裝 **ffmpeg** 並加入 PATH
- 安裝依賴：
  ```bash
  pip install -r requirements.txt
  ```
  或手動安裝所需套件。

---

## 📂 建議資料結構

```
.
├─ data/
│  ├─ datasets/
│  │  ├─ KoNViD-1k/
│  │  │  ├─ videos/
│  │  │  └─ labels.csv
│  │  ├─ LIVE-Qualcomm/
│  │  ├─ CVD2014/
│  │  └─ LSVQ/
│  ├─ KoNViD-1kinfo.mat
│  ├─ LIVE-Qualcomm_info.mat
│  ├─ CVD2014info.mat
│  ├─ LSVQ1080p_info.mat
│  └─ ...（其他 .m/.mat）
├─ "FD-VQA extract.py"
├─ "FD-VQA test.py"
├─ DATALAZY.py
└─ README.md
```

---

## 🚀 使用流程（TL;DR）

1. **先產生各資料集的 info**（`.mat`）  
   - 直接使用 `data/` 內已提供的 `*_info.mat`；或  
   - 在 MATLAB 執行對應的 `*.m`（如 `Kon.m`、`LIVE_Qualcomm.m`、`CVD2014.m`、`LSVQ.m`）+ `data_info_maker.m` 重新產生。
2. **執行特徵提取**：`"FD-VQA extract.py"`  
3. **訓練／測試**：`"FD-VQA test.py"`  
4. **大型資料集**：改用 `DATALAZY.py`（分批／快取／串流）

---

## 1) 產生資料集資訊（info）

### A. 直接使用倉庫中既有的 `.mat`
若你的路徑與命名一致，可直接引用下列之一：
- `data/KoNViD-1kinfo.mat`
- `data/LIVE-Qualcomm_info.mat`
- `data/CVD2014info.mat`
- `data/LSVQ1080p_info.mat`
- `data/LIVE-VQCinfo.mat`

> 若不一致，請改用 B 法重建或調整你的載入程式。

### B. 以 MATLAB 重新產生
1. 依你的實際影片與標註路徑，編輯 `data/<DATASET>.m`（例如 `Kon.m`）。  
2. 於 MATLAB 執行該 `.m` 腳本與 `data_info_maker.m` 以輸出 `*_info.mat`。  
3. 請將輸出 `*.mat` 放到 `data/` 供 Python 腳本使用。

---

## 2) 特徵提取 — **FD-VQA extract**

> **注意檔名含空白，請加引號。**

### Windows（PowerShell）
```powershell
python ".\FD-VQA extract.py" `
  --dataset KoNViD-1k `
  --info_mat ".\data\KoNViD-1kinfo.mat" `
  --videos ".\data\datasets\KoNViD-1kideos" `
  --out_features ".eatures\KoNViD-1k" `
  --backbone convnext_base `
  --batch_size 8 --num_workers 8 --device cuda:0
```

### Linux / macOS（bash）
```bash
python "FD-VQA extract.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --videos   ./data/datasets/KoNViD-1k/videos   --out_features ./features/KoNViD-1k   --backbone convnext_base   --batch_size 8 --num_workers 8 --device cuda:0
```

**常見參數（依你的實作為準）**
- `--dataset`：`KoNViD-1k | LIVE-Qualcomm | CVD2014 | LSVQ | LIVE-VQC | ...`
- `--info_mat`：對應資料集的 `*_info.mat`
- `--videos`：影片根目錄
- `--out_features`：特徵輸出資料夾
- `--split`：`train|val|test|all`（若支援）
- `--backbone`：`convnext_base|vit_b|swin_b|r3d_18|...`
- `--fps` / `--frame_stride`：抽幀策略（選用）
- `--device`：`cuda:0` 或 `cpu`

**輸出**
- `features/<DATASET>/*.npy|*.pt`（每支影片一或多個特徵檔）
- `features/<DATASET>/meta/`（抽取設定）

---

## 3) 訓練／測試 — **FD-VQA test**

### Windows（PowerShell）
```powershell
python ".\FD-VQA test.py" `
  --dataset KoNViD-1k `
  --info_mat ".\data\KoNViD-1kinfo.mat" `
  --features_dir ".eatures\KoNViD-1k" `
  --epochs 50 --batch_size 32 --lr 1e-4 --seed 42 `
  --out_dir ".esults\KoNViD-1k"
```

### Linux / macOS（bash）
```bash
python "FD-VQA test.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --features_dir ./features/KoNViD-1k   --epochs 50 --batch_size 32 --lr 1e-4 --seed 42   --out_dir ./results/KoNViD-1k
```

**輸出**
- `results/<DATASET>/metrics.(csv|json)`：SROCC／PLCC／KROCC／RMSE（依你的實作）
- 訓練日誌與（可選）`checkpoints/`、`tensorboard/`

---

## 4) 大型資料集 — **DATALAZY**

針對 LSVQ、YouTube-UGC 全量等大型資料集，建議改用 DATALAZY 分批載入與快取，避免 OOM。

### 串接「提取 + 測試」
```bash
python DATALAZY.py   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --videos   ./data/datasets/LSVQ/videos   --out_root ./results/LSVQ_DATALAZY   --extract --test   --chunk_size 64 --num_workers 12 --prefetch 4   --device cuda:0
```

### 僅提取（之後用 test）
```bash
python DATALAZY.py   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --videos   ./data/datasets/LSVQ/videos   --out_root ./features/LSVQ_DATALAZY   --extract   --chunk_size 64 --num_workers 12 --device cuda:0

python "FD-VQA test.py"   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --features_dir ./features/LSVQ_DATALAZY   --epochs 20 --batch_size 64 --seed 42   --out_dir ./results/LSVQ
```

**常見參數**
- `--chunk_size`：批次切塊大小
- `--out_root`：輸出／快取根目錄
- `--prefetch`：預先取用批數
- `--resume`：斷點續跑（若支援）
- 其他 dataloader 效能參數：`--num_workers`, `--pin_memory`, `--persistent_workers`

---

## 🧪 典型完整流程範例

### KoNViD-1k（小型）
```bash
# 使用既有 info：./data/KoNViD-1kinfo.mat

python "FD-VQA extract.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --videos   ./data/datasets/KoNViD-1k/videos   --out_features ./features/KoNViD-1k   --backbone convnext_base   --batch_size 8 --num_workers 8 --device cuda:0

python "FD-VQA test.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --features_dir ./features/KoNViD-1k   --epochs 50 --batch_size 32 --seed 42   --out_dir ./results/KoNViD-1k
```

### LSVQ（大型，DATALAZY）
```bash
# 使用既有 info：./data/LSVQ1080p_info.mat

python DATALAZY.py   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --videos   ./data/datasets/LSVQ/videos   --out_root ./results/LSVQ_DATALAZY   --extract --test   --chunk_size 64 --num_workers 16 --prefetch 4   --device cuda:0
```

---

## 📤 預期輸出位置

- `features/<DATASET>/`：特徵檔（`.npy` / `.pt`）與 `meta/`
- `results/<DATASET>/`：指標（metrics）、訓練日誌、（選用）`checkpoints/`、`tensorboard/`
- `data/*.mat`：各資料集 info

---

## 🔁 重現性建議

- 固定亂數種子：`--seed 42`
- （若支援）開啟 deterministic
- 將 `pip freeze` 與 `git commit hash` 輸出到 `results/<DATASET>/run_meta.json`

---

## ❗ Troubleshooting

- **找不到 split/索引**：`*_info.mat` 與實際資料路徑不一致 → 以 MATLAB 重新產生或調整 `data.py` 載入邏輯。  
- **GPU OOM**：降低 `--batch_size`；大型資料集改用 `DATALAZY` 並調整 `--chunk_size`／`--prefetch`。  
- **NaN/Inf**：檢查標註空值、特徵是否損毀；必要時啟用梯度裁切或標準化。  
- **Windows 路徑含空白**：以**引號**包住腳本與路徑（例如 `"FD-VQA test.py"`）。  
- **解碼錯誤**：安裝並設定 `ffmpeg`，或改用 OpenCV 內建解碼（依你的實作）。  

---

## 📜 授權 / 貢獻

- 授權條款請見 `LICENSE`（若未提供，請依需求補上）。  
- 歡迎透過 Issue／PR 回報問題或貢獻功能。
