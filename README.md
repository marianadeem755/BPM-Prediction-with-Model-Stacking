# Beats Per Minute (BPM) Prediction with Model Stacking

## Overview
This project aims to predict the Beats Per Minute (BPM) of songs using a machine learning approach with model stacking. The notebook implements an ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost, and HistGradientBoosting) and uses a RidgeCV meta-learner to get the best competition results. The dataset includes audio features such as rhythm, loudness, vocal content, and energy, with the target variable being BPM.

## Dataset
- **Source**: Combined dataset from training, testing, and original music data (538,797 rows, 11 columns).
- **Features**:
  - RhythmScore
  - AudioLoudness
  - VocalContent
  - AcousticQuality
  - InstrumentalScore
  - LivePerformanceLikelihood
  - MoodScore
  - TrackDurationMs
  - Energy
- **Target**: BeatsPerMinute (BPM)
- **Preprocessing**:
  - Imputed missing numerical values with median.
  - No duplicate rows found.
  - Applied `RobustScaler` for numerical scaling to handle outliers.
  - Feature engineering:
    - Interaction terms (e.g., Rhythm_Loudness_Prod, Vocal_Acoustic_Quot).
    - Polynomial features on selected columns.
    - Log-transformations for skewed features (e.g., TrackDurationMs, AudioLoudness).
    - Binned continuous variables into categories (e.g., Duration_Cat, Energy_Cat).

## Methodology
### Model Configurations
The notebook uses four gradient boosting models with tuned hyperparameters:
- **LightGBM**
- **XGBoost**
- **CatBoost**
- **HistGradientBoosting**

### Cross-Validation
- **Setup**: 15-fold cross-validation (`KFold`, `random_state=42`) for robust evaluation.
- **Metrics**: Root Mean Squared Error (RMSE) calculated for each fold.
- **Results**:
  - LightGBM Avg RMSE: 26.4577 (Std: 0.0920)
  - XGBoost Avg RMSE: 26.4605 (Std: 0.0917)
  - CatBoost Avg RMSE: 26.4591 (Std: 0.0918)
  - HistGradientBoosting Avg RMSE: 26.4598 (Std: 0.0916)

### Stacking
- **Meta Features**: Out-of-fold (OOF) predictions from the four models.
- **Meta Learner**: `RidgeCV` with cross-validated regularization.
- **Output**: Final predictions clipped to BPM range.

### Submission
- Predictions saved in `submission.csv` with columns `id` and `BeatsPerMinute`.
- Sample submission preview:
