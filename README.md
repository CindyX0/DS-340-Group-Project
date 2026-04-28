# Music Genre Classification — DS340 Group Project
**Authors:** Aymen Tiguite, Cindy Frempong | Boston University | DS 340

Automatic music genre classification using a late fusion ensemble of XGBoost, LightGBM, and a 1D CNN trained on MFCC audio features. Achieves **90% test accuracy** across 9 parent genre categories.

---

## How This Model Was Built

### Step 1 — Data Collection
We used two data sources:

**Source 1: Spotify Tracks Dataset (Kaggle)**
- 114,000 tracks × 114 genres with pre-computed audio features (energy, danceability, tempo, etc.)
- Download from: https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset
- Place at `Data/spotify-tracks-dataset.csv`

**Source 2: MFCC Features (self-extracted)**
- We extracted raw audio features from ~29,000 YouTube clips using `yt-dlp` + `librosa`
- Already available in this repo under `MFCC Extraction/mfcc_features_b.csv` and `checkpoint_a.csv`
- See `MFCC Extraction/MFCC_Aymen.ipynb` for the full extraction pipeline (requires Google Colab + cookies.txt)

### Step 2 — EDA & Preprocessing
**Notebook:** `EDA and Baseline Models/EDA.ipynb`
- Explored audio feature distributions across genres
- Identified that 16,299 tracks (14.4%) appear under multiple genre labels with identical features — removed duplicates using `drop_duplicates(subset='track_id', keep='first')`

### Step 3 — Baseline Models
**Notebook:** `EDA and Baseline Models/Preprocessing_and_Baselines.ipynb`
- Random chance: 0.88% (1/114 classes)
- KNN: 21.0%, Logistic Regression: 19.7%, Random Forest: **30.2%**

### Step 4 — Intermediate Models
**Notebooks:** `Intermediate Models/`
- `DNN_v1.ipynb` — 3-layer DNN on 12k sample: 41.8%
- `XGBoost_FullData.ipynb` — XGBoost on full dataset: 49.4% (key insight: more data > better model)
- `DNN_ParentGenres.ipynb` — DNN + XGBoost ensemble: 50.3%
- Mapped 114 fine-grained genres → 12 parent genres: Rock, Metal, Electronic, Latin, Jazz/Blues, Classical/Instrumental, Country/Folk, Reggae, World/Other, Pop, Hip-Hop/R&B, House/Dance

### Step 5 — Feature Engineering
**Notebook:** `Codebooks/FinalModel.ipynb`
- Added 17 engineered features: interaction terms (`energy × acousticness`), polynomial terms (`loudness²`), log transforms, tempo bins
- XGBoost jumped from 49.9% → **66.5%** with engineered features alone

### Step 6 — MFCC Extraction
**Notebooks:** `MFCC Extraction/MFCC_Aymen.ipynb`, `MFCC Extraction/MFCC_Cindy.ipynb`

Aymen extracted MFCCs for 6 genres (Latin, Jazz/Blues, Classical/Instrumental, Country/Folk, Reggae, World/Other). Cindy extracted for 3 genres (Rock, Metal, Electronic). Combined: **29,202 tracks × 59 features**.

Each track's pipeline:
1. Search YouTube for `"{track_name} {artist} audio"` using `yt-dlp`
2. Download a 30-second clip centered at the song's midpoint
3. Extract with `librosa`: 13 MFCCs (mean + std + delta = 39 values) + 12 chroma + 7 spectral contrast + 1 ZCR = **59 features**
4. Save to CSV with checkpoint/resume support for long runs

### Step 7 — Final Model (Late Fusion Ensemble)
**Notebook:** `Final Models/FinalModel_CNN.ipynb`

Three models trained independently, then combined via weighted soft voting:

| Model | Input | Accuracy | Weight |
|---|---|---|---|
| XGBoost | 32 engineered tabular features | 66.5% | 60% |
| LightGBM | 32 engineered tabular features | 67.0% | 30% |
| 1D CNN (MFCCNet) | 59 MFCC features | 35.7% | 10% |
| **Late Fusion** | **All combined** | **89.9%** | — |

**Experiment 2** in the same notebook tests fusion strategies: early fusion (concatenate features → single DNN) achieved only 62.1% — worse than tabular alone — because a single model cannot effectively handle two fundamentally different input representations. Late fusion with accuracy-proportional weights was the clear winner.

---

## Repository Structure

```
DS-340-Group-Project/
├── Data/
│   └── spotify-tracks-dataset.csv        ← download from Kaggle (not in repo)
├── EDA and Baseline Models/
│   ├── EDA.ipynb                          ← exploratory analysis, deduplication
│   └── Preprocessing_and_Baselines.ipynb ← KNN, LogReg, Random Forest baselines
├── Intermediate Models/
│   ├── DNN_v1.ipynb                       ← first DNN on sampled data
│   ├── DNN_ParentGenres.ipynb             ← DNN on parent genres + ensemble
│   └── XGBoost_FullData.ipynb             ← XGBoost on full dataset
├── MFCC Extraction/
│   ├── MFCC_Aymen.ipynb                   ← Aymen's extraction pipeline (6 genres)
│   ├── MFCC_Cindy.ipynb                   ← Cindy's extraction pipeline (3 genres)
│   ├── MFCC_Data_processing.ipynb         ← merging + cleaning extracted features
│   ├── mfcc_features_b.csv                ← Aymen's extracted features (18,202 tracks)
│   └── checkpoint_a.csv                   ← Cindy's extracted features (11,000 tracks)
├── Final Models/
│   ├── FinalModel_CNN.ipynb               ← FINAL MODEL — run this
│   └── MFCC_vs_DNN.ipynb                  ← early experiment comparing MFCC vs tabular DNN
├── Codebooks/
│   └── FinalModel.ipynb                   ← intermediate ensemble experiments
├── Visualizations/                         ← all figures used in the paper
├── MFCCs Aymen/                            ← reference PDF
├── Paper/                                  ← final written report (md + pdf)
├── requirements.txt                        ← Python dependencies
└── venv/                                   ← pre-built virtual environment
```

---

## How to Run the Final Model

### Option A — Use the conda environment (recommended)

```bash
# 1. Activate the conda environment
conda activate ds340

# 2. Launch Jupyter
jupyter notebook

# 3. Open Final Models/FinalModel_CNN.ipynb
#    Select kernel: DS340 (conda)
#    Run All Cells
```

> If you don't have conda: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first, then:
> ```bash
> conda env create -f environment.yml
> ```

### Option B — Fresh install

```bash
# 1. Create and activate a new venv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register the Jupyter kernel
python -m ipykernel install --user --name=ds340 --display-name="DS340 (venv)"

# 4. Launch Jupyter and open Final Models/FinalModel_CNN.ipynb
jupyter notebook
```

### Data setup (required before running)

1. Download `spotify-tracks-dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset)
2. Place it at `Data/spotify-tracks-dataset.csv`
3. The MFCC CSVs are already in the repo — no re-extraction needed

### Run order (if running all notebooks from scratch)

1. `EDA and Baseline Models/EDA.ipynb`
2. `EDA and Baseline Models/Preprocessing_and_Baselines.ipynb`
3. `Intermediate Models/DNN_v1.ipynb`
4. `Intermediate Models/XGBoost_FullData.ipynb`
5. `Intermediate Models/DNN_ParentGenres.ipynb`
6. `Codebooks/FinalModel.ipynb`
7. `Final Models/MFCC_vs_DNN.ipynb`
8. `Final Models/FinalModel_CNN.ipynb` ← **main result**

---

## How the MFCC Extraction Works (if you want to recreate it)

The extraction pipeline in `MFCC Extraction/MFCC_Aymen.ipynb` was designed to run on **Google Colab** (requires a GPU/fast connection and a YouTube cookies file to avoid rate limiting).

**Why we extracted MFCCs ourselves instead of using Spotify's features:**
Spotify's audio features (energy, danceability, etc.) are high-level summaries that lose spectral texture — the timbre that distinguishes a distorted Metal guitar from a clean Rock guitar. MFCCs capture the short-term power spectrum of audio, encoding the "texture" of sound in a way that matches human pitch perception.

**Pipeline overview:**
1. **Sampling** — Stratified sampling from the Spotify dataset, ~439 tracks per subgenre, split between Aymen (Latin, Jazz, Classical, Country, Reggae, World) and Cindy (Rock, Metal, Electronic)
2. **YouTube search** — `yt-dlp` searches `"{track_name} {artist} audio"` and downloads the top result
3. **Clip extraction** — 30-second window centered at the song midpoint (avoids intros/outros)
4. **Feature extraction** via `librosa`:
   - 13 MFCC coefficients → mean, std, delta = **39 values** (captures spectral envelope over time)
   - 12 chroma features (pitch class distribution)
   - 7 spectral contrast features (peak vs valley amplitude differences)
   - 1 zero-crossing rate (roughness/noisiness proxy)
   - Total: **59 features per track**
5. **Checkpoint/resume** — saves progress every 10 tracks so long runs can be interrupted and resumed
6. **Parallel workers** — 6 concurrent threads to speed up downloads

**To recreate the extraction:**
- You need a `cookies.txt` file exported from a logged-in YouTube browser session (see `MFCC Extraction/cookies.txt` — note: cookies expire, you may need to refresh)
- Run on Google Colab with Google Drive mounted
- Expect ~8–12 hours for 18,000 tracks depending on connection speed and YouTube rate limiting

---

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest (baseline) | 30.2% | 0.283 |
| 3-layer DNN | 41.8% | 0.404 |
| XGBoost v1 (full data) | 49.4% | 0.464 |
| XGBoost + LGB Ensemble | 50.3% | 0.468 |
| XGBoost (engineered features) | 66.5% | 0.632 |
| LightGBM (engineered features) | 67.0% | 0.637 |
| MFCC CNN alone | 37.8% | 0.348 |
| **Late Fusion (XGB+LGB+CNN)** | **89.9%** | **0.899** |

---

## Dependencies

See `requirements.txt`. Key packages:
- `xgboost`, `lightgbm` — gradient boosted tree models
- `torch` — 1D CNN training
- `librosa` — MFCC extraction (Colab only)
- `yt-dlp` — YouTube audio download (Colab only)
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
