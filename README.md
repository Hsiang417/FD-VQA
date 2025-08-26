# FD-VQA

FD-VQA provides an end‑to‑end **Video Quality Assessment (VQA)** workflow:
1) Generate per‑dataset **info** (index & statistics)  
2) Run **FD-VQA extract** to compute features  
3) Run **FD-VQA test** to train/evaluate scores  
4) For large datasets, switch to **DATALAZY** (chunked loading/cache/streaming)

> ⚠️ Two main scripts in this repo have spaces in their filenames: `FD-VQA extract.py`, `FD-VQA test.py`.  
> Always wrap them with quotes when executing (e.g., `python "FD-VQA extract.py"`).

---

## 📦 What’s in this repo

- `DATALAZY.py`  
- `FD-VQA extract.py`  
- `FD-VQA test.py`  
- `data/`
  - `data.py`, `data_info_maker.m`
  - MATLAB generators for multiple datasets (e.g., `Kon.m`, `LIVE_Qualcomm.m`, `CVD2014.m`, `LSVQ.m`)
  - Prebuilt **info** files (e.g., `KoNViD-1kinfo.mat`, `LIVE-Qualcomm_info.mat`, `CVD2014info.mat`, `LSVQ1080p_info.mat`, `LIVE-VQCinfo.mat` …)

---

## 🔧 Requirements

- Python ≥ 3.9
- Suggested packages: `torch`, `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `tqdm`, `matplotlib`
- For frame decoding: install **ffmpeg** and ensure it is in your PATH
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```
  or install packages manually.

---

## 📂 Suggested layout

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
│  └─ ... (other .m / .mat)
├─ "FD-VQA extract.py"
├─ "FD-VQA test.py"
├─ DATALAZY.py
└─ README.md
```

---

## 🚀 TL;DR

1. **Prepare dataset info** (`*.mat`)  
   - Use the provided `data/*_info.mat`, or  
   - Regenerate via MATLAB with `<DATASET>.m` (e.g., `Kon.m`, `LIVE_Qualcomm.m`, `CVD2014.m`, `LSVQ.m`) + `data_info_maker.m`.
2. **Extract features** with `"FD-VQA extract.py"`  
3. **Train/Test** with `"FD-VQA test.py"`  
4. **Large datasets** → use `DATALAZY.py` (chunk/cache/stream)

---

## 1) Build dataset info

### A. Use prebuilt `.mat` files
If your paths/names are compatible, you can use:
- `data/KoNViD-1kinfo.mat`
- `data/LIVE-Qualcomm_info.mat`
- `data/CVD2014info.mat`
- `data/LSVQ1080p_info.mat`
- `data/LIVE-VQCinfo.mat`

> If not compatible, regenerate via MATLAB or adjust your loader.

### B. Regenerate via MATLAB
1. Edit `data/<DATASET>.m` to reflect your actual video/label paths (e.g., `Kon.m`).  
2. Run that `.m` along with `data_info_maker.m` to export `*_info.mat`.  
3. Place the resulting `.mat` under `data/` for Python scripts to consume.

---

## 2) Feature extraction — **FD-VQA extract**

> Filenames contain spaces — *use quotes*.

### Windows (PowerShell)
```powershell
python ".\FD-VQA extract.py" `
  --dataset KoNViD-1k `
  --info_mat ".\data\KoNViD-1kinfo.mat" `
  --videos ".\data\datasets\KoNViD-1kideos" `
  --out_features ".eatures\KoNViD-1k" `
  --backbone convnext_base `
  --batch_size 8 --num_workers 8 --device cuda:0
```

### Linux / macOS (bash)
```bash
python "FD-VQA extract.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --videos   ./data/datasets/KoNViD-1k/videos   --out_features ./features/KoNViD-1k   --backbone convnext_base   --batch_size 8 --num_workers 8 --device cuda:0
```

**Common args (subject to your implementation)**
- `--dataset`: `KoNViD-1k | LIVE-Qualcomm | CVD2014 | LSVQ | LIVE-VQC | ...`
- `--info_mat`: path to the dataset’s `*_info.mat`
- `--videos`: root directory for videos
- `--out_features`: output folder for features
- `--split`: `train|val|test|all` (if supported)
- `--backbone`: `convnext_base|vit_b|swin_b|r3d_18|...`
- `--fps` / `--frame_stride`: frame sampling strategy (optional)
- `--device`: `cuda:0` or `cpu`

**Outputs**
- `features/<DATASET>/*.npy|*.pt` (per‑video feature files)
- `features/<DATASET>/meta/` (extraction settings)

---

## 3) Train / Test — **FD-VQA test**

### Windows (PowerShell)
```powershell
python ".\FD-VQA test.py" `
  --dataset KoNViD-1k `
  --info_mat ".\data\KoNViD-1kinfo.mat" `
  --features_dir ".eatures\KoNViD-1k" `
  --epochs 50 --batch_size 32 --lr 1e-4 --seed 42 `
  --out_dir ".
esults\KoNViD-1k"
```

### Linux / macOS (bash)
```bash
python "FD-VQA test.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --features_dir ./features/KoNViD-1k   --epochs 50 --batch_size 32 --lr 1e-4 --seed 42   --out_dir ./results/KoNViD-1k
```

**Outputs**
- `results/<DATASET>/metrics.(csv|json)` with SROCC/PLCC/KROCC/RMSE (depending on your code)
- training logs and (optionally) `checkpoints/`, `tensorboard/`

---

## 4) Large datasets — **DATALAZY**

For LSVQ or full YouTube‑UGC, prefer DATALAZY to avoid OOM by chunking and caching.


### Extract only → then run test
```bash
python FD-VQA extract.py   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --videos   ./data/datasets/LSVQ/videos   --out_root ./features/LSVQ_DATALAZY   --extract   --chunk_size 64 --num_workers 12 --device cuda:0

python "DATALAZY.py"   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --features_dir ./features/LSVQ_DATALAZY   --epochs 20 --batch_size 64 --seed 42   --out_dir ./results/LSVQ
```

**Common args**
- `--chunk_size`: chunk size for streaming
- `--out_root`: output/cache root
- `--prefetch`: number of prefetched chunks
- `--resume`: resume from checkpoints (if supported)
- data‑loader tuning: `--num_workers`, `--pin_memory`, `--persistent_workers`

---

## 🧪 Typical end‑to‑end examples

### KoNViD‑1k (small)
```bash
# Using the existing info: ./data/KoNViD-1kinfo.mat

python "FD-VQA extract.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --videos   ./data/datasets/KoNViD-1k/videos   --out_features ./features/KoNViD-1k   --backbone convnext_base   --batch_size 8 --num_workers 8 --device cuda:0

python "FD-VQA test.py"   --dataset KoNViD-1k   --info_mat ./data/KoNViD-1kinfo.mat   --features_dir ./features/KoNViD-1k   --epochs 50 --batch_size 32 --seed 42   --out_dir ./results/KoNViD-1k
```

### LSVQ (large, DATALAZY)
```bash
# Using the existing info: ./data/LSVQ1080p_info.mat
python "FD-VQA extract.py"   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat    --videos   ./data/datasets/LSVQ/videos    --out_features ./features/LSVQ   --backbone convnext_base   --batch_size 8 --num_workers 8 --device cuda:0
python "DATALAZY.py"   --dataset LSVQ   --info_mat ./data/LSVQ1080p_info.mat   --features_dir ./features/LSVQ_DATALAZY   --epochs 20 --batch_size 64 --seed 42   --out_dir ./results/LSVQ
```

---

## 📤 Expected outputs

- `features/<DATASET>/` — feature files (`.npy` / `.pt`) and `meta/`
- `results/<DATASET>/` — metrics, logs, optional `checkpoints/`, `tensorboard/`
- `data/*.mat` — dataset info

---

## 🔁 Reproducibility tips

- Fix random seeds: `--seed 42`
- (If available) enable deterministic behavior
- Save `pip freeze` and the `git commit` into `results/<DATASET>/run_meta.json`

---

## ❗ Troubleshooting

- **Missing splits/index**: your `*_info.mat` may not match the actual paths → regenerate via MATLAB or adjust the loader in `data.py`.  
- **GPU OOM**: lower `--batch_size`; for large datasets, use DATALAZY and tune `--chunk_size`/`--prefetch`.  
- **NaN/Inf**: check for empty labels, corrupted features; consider gradient clipping/normalization.  
- **Spaces in paths on Windows**: wrap script names and paths in quotes (e.g., `"FD-VQA test.py"`).  
- **Decoding failures**: install and expose `ffmpeg` in PATH, or switch to OpenCV decoding if supported.

---

## 📜 License / Contributing

- See `LICENSE` (or add one if missing).  
- Contributions and issues are welcome.
