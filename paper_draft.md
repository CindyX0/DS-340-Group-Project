# Automatic Music Genre Classification Using Spotify Audio Features and MFCC-Based Late Fusion
### Ensemble Methods for Multi-Class Genre Prediction
**Authors:** Aymen Tiguite, Cindy Frempong
**Boston University | DS 340**
**April 27, 2026**

---

## Introduction

Music genre classification is a classic problem in audio and music information retrieval (MIR). Automatically identifying a song's genre has real-world applications across the music industry — from powering recommendation engines on streaming platforms like Spotify and Apple Music, to organizing large digital music libraries, to enabling content-based search and music discovery. Despite its practical importance, genre classification is a genuinely hard problem: genre boundaries are fuzzy, listener perceptions vary, and many songs blend elements of multiple genres simultaneously.

In this paper, we explore multiple model types and ensemble methods to train a system that correctly identifies the genre of a given song. We draw on a dataset of tracks described by a set of numerical audio descriptors from Spotify's audio analysis API. These features summarize the feel and energy of a song, its sound source, and its musical structure. For example, danceability estimates how suitable a track is for dancing, energy reflects intensity and activity, valence measures how positive or happy a track sounds, and acousticness estimates whether a song is more acoustic versus electronically produced.

In addition to these descriptors, we make use of Mel-Frequency Cepstral Coefficients (MFCCs). MFCCs are a compact numerical representation of the short-term power spectrum of a sound, computed by mapping the audio signal onto the mel scale — a perceptual scale of pitch that closely mirrors how human hearing works. MFCCs are widely used in speech and music processing because they capture the texture of a sound in a way that correlates well with human perception of musical style. In our work, MFCCs serve as a complementary input representation, allowing a convolutional neural network (CNN) to learn patterns in the spectral content of songs that are characteristic of different genres.

Our final model combines XGBoost, LightGBM, and an MFCC-based CNN in a late fusion ensemble, achieving **90.0% test accuracy** on 9 parent genre categories — up from a random chance baseline of 11.1% (1/9) and a best single-model baseline of 30.2%.

---

## Data Description

**Source 1: Spotify Tracks Dataset**
https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset

The dataset contains 114,000 tracks spanning 114 music genres, with each row representing an individual song and each column capturing a specific attribute of that track. After dropping 3 null rows the working dataset contains 113,999 tracks with 22 columns: track metadata (track_id, artists, album_name, track_name), 15 audio features (popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature), and the track_genre label.

**Source 2: MFCC Dataset (self-extracted)**
Available in our GitHub repository under `MFCC Extraction/mfcc_features_b.csv` and `checkpoint_a.csv`.

We manually extracted audio features for approximately 29,202 tracks from the Spotify dataset. For each track, we computed five types of features using librosa: (1) 13 MFCC coefficients summarized by mean, standard deviation, and delta (39 values total); (2) 12 chroma features capturing the distribution of pitch classes; (3) 7 spectral contrast features capturing the difference in amplitude between peaks and valleys in the spectrum; and (4) zero-crossing rate (1 value). This produces **59 features per track** alongside the track_id and genre label.

---

## EDA and Dataset Overview

**Data Preprocessing and Cleaning**

The Spotify dataset was relatively clean. We identified and dropped 3 null rows. We then investigated whether individual tracks appeared under multiple genre labels. We found that **16,299 tracks (14.4%) are assigned to more than one genre** with identical audio features — the same song listed under different labels. For example, "Better" by Pink Sweat$ appears as both "chill" and "soul" with identical danceability, energy, and tempo values. This is noise: a model trained on this data sees the same input features paired with two conflicting labels. We removed these duplicates using `drop_duplicates(subset='track_id', keep='first')`, reducing the dataset to **89,740 unique tracks**.

[INSERT: duplicate_analysis.png — Figure showing before/after deduplication]

**Genre Mapping**

114 fine-grained genres (e.g., psych-rock, deep-house, detroit-techno) were mapped to 9 parent genres matching our MFCC extraction coverage: Rock, Metal, Electronic, Latin, Jazz/Blues, Classical/Instrumental, Country/Folk, Reggae, and World/Other. Three parent genres — Pop, Hip-Hop/R&B, and House/Dance — were excluded from the final model because we did not extract MFCC clips for those genres. The final working dataset contains **55,191 deduplicated tracks** across the 9 genres.

**Exploratory Data Analysis**

The raw dataset contains 114 unique genres, each with exactly 1,000 tracks, resulting in a perfectly balanced class distribution. This balance is ideal for training models, as it prevents bias toward any particular genre.

[INSERT: fig1_feature_distributions.png — Figure 1: Audio feature distributions across 4 parent genres]

As shown in Figure 1, audio features show clear separability for some genre pairs. Metal has high energy and low acousticness, Classical/Instrumental has high acousticness and low energy, and Hip-Hop/R&B shows elevated speechiness and danceability. However, genres like Rock and Electronic share similar energy ranges, which explains why they are among the most commonly confused pairs in our models.

[INSERT: fig3_mfcc_heatmap.png — Figure 3: MFCC heatmap by genre]

Figure 3 shows the average MFCC coefficients per genre. Metal and Rock show distinctly higher values on MFCC 1 (capturing overall spectral energy/brightness) compared to Classical/Instrumental and Jazz/Blues, which display lower, more concentrated coefficient values. This confirms that MFCC features capture complementary information not present in the Spotify tabular features.

---

## Methodology

**Baseline Models**

We trained three baseline models on the 15 raw Spotify audio features to establish performance benchmarks.

*Random chance baseline:* With 9 parent genres and a balanced dataset, a random classifier achieves 1/9 = **11.1%** accuracy. With all 114 genres, random chance is 1/114 = **0.88%**.

*K-Nearest Neighbors:* A KNN classifier trained on the 15 features, standardized with StandardScaler and dimensionality-reduced with PCA (retaining 95% variance). We tuned k from 1–30 using 5-fold cross-validation. KNN achieved **21.0% accuracy** on 114 genres, rising to 44% when genres were binned into broader categories. KNN struggles in high-dimensional spaces where distances lose discriminative power.

*Logistic Regression:* Trained using the saga solver on a stratified sample of 300 tracks per genre (34,200 total), with a 70/15/15 train/val/test split. Achieved **19.7% accuracy** and macro F1 of 0.17 on 114 genres.

*Random Forest:* 100 trees with maximum depth 15, same 300-per-genre sample. Achieved **30.2% accuracy** and macro F1 of 0.28 — the strongest baseline.

**Intermediate Models**

After establishing baselines, we trained increasingly powerful models, progressively improving classification accuracy on 12 parent genres.

*DNN (3-layer):* A deep neural network with three hidden layers (256 → 128 → 64), batch normalization, ReLU activations, and dropout (0.4), trained on a 12,000-track stratified sample (1,000 per parent genre). Achieved **41.8% accuracy** and macro F1 of 0.41 — a significant improvement over baselines, but still limited by the small sample size.

*XGBoost v1 (full data):* A key insight was that we had been training on only 12,000 of the available tracks. Switching to the full cleaned dataset with XGBoost (1,000 trees, max depth 6, learning rate 0.05, subsample 0.8) immediately improved accuracy to **49.4%** with macro F1 of 0.46 — a 7-point gain from simply using more data. This confirmed that dataset size was a more important bottleneck than model architecture at this stage.

*XGBoost v2 (tuned):* Hyperparameter tuning (depth 7, 1,000 trees, early stopping on validation log-loss) achieved **49.9%** accuracy — a modest gain, indicating the model was approaching the ceiling of what the raw Spotify features could support.

*LightGBM:* Trained with leaf-wise tree growth (num_leaves=63, learning rate 0.05, balanced class weights). Achieved **49.7%** accuracy — nearly identical to XGBoost but capturing different feature interactions due to its leaf-wise splitting strategy.

*XGBoost + LightGBM Ensemble:* Soft voting (averaging predicted class probabilities) produced **50.3%** accuracy and macro F1 of 0.47, confirming ensemble methods are more robust than either model alone.

**Feature Engineering**

For the final model, we engineered 17 additional features from the base 15 Spotify features to better capture non-linear relationships:
- Interaction terms: energy × acousticness, energy × danceability, loudness × energy
- Polynomial terms: energy², acousticness², loudness²
- Log transforms: log(duration_ms), log(popularity + 1)
- Tempo bins: slow (<90 BPM), mid (90–120), fast (120–140), very fast (>140)

These engineered features boosted XGBoost from 49.9% to **66.5%** and LightGBM to **67.0%**, confirming that the model was not saturated — it was simply missing better-crafted input representations.

**MFCC Dataset Extraction**

After evaluating intermediate models, we found that Spotify's tabular audio features alone were not sufficient to reliably distinguish acoustically similar genres. We decided to incorporate MFCCs, extracted directly from audio clips.

We built a custom extraction pipeline that: (1) searched YouTube for each track using track name and artist, (2) downloaded a 30-second audio clip centered at the song's midpoint using yt-dlp, and (3) extracted audio features using librosa: 13 MFCC coefficients (mean, std, delta = 39 values), 12 chroma features, 7 spectral contrast features, and zero-crossing rate — producing a **59-feature vector** per track. Tracks that could not be retrieved or whose extraction failed were skipped. We included "official audio" in the search query to reduce the chance of retrieving covers or live versions. The final audio feature dataset contains **29,202 tracks** across 9 parent genres.

[INSERT: fig4_mfcc_distributions.png — Figure 4: MFCC coefficient distributions showing genre separability]

**Final Model: Late Fusion Ensemble**

[INSERT: fig5_architecture.png — Figure 5: Architecture diagram]

The final model combines three components via late fusion:

- *Stage 1A — XGBoost* on 32 engineered tabular features: **66.5%** alone
- *Stage 1B — LightGBM* on 32 engineered tabular features: **67.0%** alone  
- *Stage 1C — 1D CNN* on 59 audio features (MFCC + chroma + spectral contrast + ZCR): **37.2%** alone

The CNN architecture consists of three 1D convolutional blocks (64 → 128 → 256 filters, kernel sizes 5 → 3 → 3), followed by global average pooling and a fully connected head (256 → 128 → 64 → 9 classes), with batch normalization and dropout throughout. It was trained with cosine annealing learning rate scheduling and class-weighted cross-entropy loss to handle class imbalance.

Late fusion combines the predicted probability vectors from all three models using a weighted average: **XGB (60%) + LGB (30%) + CNN (10%)**. The weights were selected based on each model's individual validation accuracy. This approach allows each model to specialize — XGBoost and LightGBM on structured tabular features, the CNN on raw spectral texture — and their outputs are combined only at the prediction level. The tabular models also provide predictions for tracks without MFCC coverage, making the system robust to missing data.

---

## Evaluation Metrics and Results

We evaluated all models on a held-out test set using two metrics:
- **Accuracy**: fraction of correctly predicted genres
- **Macro F1**: average F1 score across all classes, weighted equally regardless of class size

The random chance baseline for 9 balanced classes is **11.1%**. All models substantially exceed this.

[INSERT: table1_results.png — Table 1: Full model comparison]

[INSERT: fig2_model_progression.png — Figure 2: Model progression bar chart]

[INSERT: confusion_matrix_final.png — Figure 6: Confusion matrix for final model]

The late fusion ensemble achieves **90.0% accuracy** and a macro F1 of **0.899** on the 9-genre classification task. The confusion matrix shows that the strongest-performing genres are Country/Folk (93.1% recall), Jazz/Blues (98.9% precision), and Classical/Instrumental (90.8%). The most common confusions occur between Electronic and Rock (both high-energy genres) and between Latin and World/Other (broad overlapping categories).

---

## Discussion

**Why XGBoost outperforms DNNs on tabular data:**
Spotify's audio features are pre-computed summary statistics — there is no spatial or sequential structure between them. XGBoost excels on this type of structured tabular data because it can learn non-linear decision boundaries directly on the feature values through gradient-boosted trees. DNNs require significantly more data to learn useful representations from scratch, and with 15 raw features the network had little structure to exploit. This is consistent with the broader machine learning literature, where tree-based methods consistently outperform neural networks on tabular tasks.

**Why late fusion works:**
The tabular and MFCC features are fundamentally different representations. Tabular features are high-level computed summaries; MFCCs capture raw spectral texture from the audio waveform itself. These two modalities capture complementary information — a song's danceability score and its MFCC pattern together are far more informative than either alone. Late fusion preserves each model's specialized strength while combining their knowledge at the probability level, rather than forcing a single model to handle two very different input spaces.

**Limitations:**
- MFCC coverage is limited to 29,202 of the 89,740 available tracks, and only covers 9 of 12 parent genres. Extending MFCC extraction to Pop, Hip-Hop/R&B, and House/Dance could further improve performance.
- YouTube extraction introduces noise: some retrieved clips may be covers, live versions, or incorrect matches despite "official audio" filtering.
- The 90% result is measured on 9 parent genres. Performance on the full 114-genre task remains around 66–67% (tabular only), which reflects the genuine acoustic similarity between many fine-grained subgenres.

---

## Conclusion

We built a music genre classification system that achieves **90.0% accuracy** on 9 parent genre categories by combining gradient-boosted tree models (XGBoost + LightGBM) trained on engineered Spotify audio features with a 1D CNN trained on self-extracted MFCC features, fused via a weighted late fusion ensemble.

The key findings from our work are:

1. **Feature engineering matters more than architecture** — engineering 17 additional tabular features boosted accuracy by 16 points (49.9% → 66.5%) without changing the model type.
2. **MFCCs add substantial complementary signal** — the CNN alone achieved only 37.2%, but combining it with the tabular models via late fusion jumped accuracy to 90.0%.
3. **Data quality is critical** — removing 16,299 duplicate tracks that appeared under multiple genre labels with identical features measurably improved model stability and final accuracy.
4. **More data beats better models** — switching from a 12k sample to the full 89k dataset improved accuracy by 7 points with the same model architecture.

Future work includes extending MFCC extraction to the remaining 3 parent genres, exploring transformer-based audio models (e.g., wav2vec), and building a fully hierarchical two-stage classifier that first predicts parent genre and then routes to a specialist model for subgenre classification.

---

## AI Usage Disclaimer

We used Claude (Anthropic) to assist with code structure, debugging, and notebook organization throughout this project. Specifically: designing the DNN and CNN architectures, identifying the multi-genre duplicate issue in the dataset, structuring the late fusion ensemble pipeline, and drafting portions of this report. All experimental decisions, hyperparameter choices, and result interpretations are our own.

---

## References

- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293–302.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.
- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *SciPy 2015*.
- Spotify Tracks Dataset: https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset
