# Disease Prediction from Medical Data

A machine-learning classification project using structured clinical features from the UCI Heart Disease dataset to predict whether a record indicates heart disease.

## Objective

Compare four classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

**Educational use only:** This is not a medical diagnostic tool and must not be used to make clinical decisions.

## Dataset

- Source: UCI Machine Learning Repository, Heart Disease dataset (ID 45)
- Original records: 303
- Original columns: 14, including `num`
- Predictor features: 13
- Target: `target = 1` when `num > 0`; otherwise `target = 0`
- Missing values in `ca` and `thal` were filled with their respective column medians.

| Target | Meaning in this project | Records | Share |
|---|---|---:|---:|
| 0 | No disease indicated by dataset target | 164 | 54.13% |
| 1 | Disease indicated by dataset target | 139 | 45.87% |

## Recorded Model Results

Models were evaluated on a stratified 80/20 train-test split (`random_state=42`); the test set contained 61 records.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | **91.80%** | 87.10% | 96.43% | **91.53%** | 95.78% |
| XGBoost | 90.16% | 84.38% | 96.43% | 90.00% | **95.89%** |
| Logistic Regression | 86.89% | 81.25% | 92.86% | 86.67% | 95.13% |
| SVM | 85.25% | 80.65% | 89.29% | 84.75% | 94.37% |

The training script selected `models/best_model.joblib` by highest test ROC-AUC, so it contains XGBoost for this run. Random Forest achieved the highest test accuracy and F1-score. These are results from one small test split and are not evidence of clinical performance or generalization to other populations.

## Project Structure

```text
CodeAlpha_DiseasePrediction/
├── download_data.py
├── prepare_data.py
├── train_models.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── heart_disease.csv
│   └── heart_disease_clean.csv
├── models/
│   ├── best_model.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   └── xgboost.joblib
└── reports/
    ├── best_model.txt
    ├── confusion_matrix_best_model.png
    ├── correlation_heatmap.png
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── model_metrics.json
    ├── roc_curves.png
    └── target_distribution.png
```

This is the intended structure based on the project steps. Keep only files that actually exist locally and adjust filenames if needed.

## Setup and Run

### 1. Clone the repository

Replace the placeholder URL with your actual repository URL.

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python download_data.py
```

Uses `ucimlrepo` to fetch UCI Heart Disease dataset ID 45 and saves `data/heart_disease.csv`.

### 5. Prepare the data

```bash
python prepare_data.py
```

Creates `data/heart_disease_clean.csv` and exploratory plots in `reports/`.

### 6. Train and evaluate models

```bash
python train_models.py
```

Saves trained model pipelines in `models/` and metrics/plots in `reports/`.

## Evaluation Artifacts

- `reports/model_comparison.csv` — model metrics
- `reports/model_comparison.png` — visual metric comparison
- `reports/confusion_matrix_best_model.png` — confusion matrix for the model selected by test ROC-AUC
- `reports/roc_curves.png` — ROC curves
- `reports/correlation_heatmap.png` — feature correlations
- `reports/target_distribution.png` — target distribution

## Limitations

- The dataset is small and may not represent all patient populations.
- Results depend on the split and preprocessing.
- The binary target is derived from `num > 0`.
- Test metrics are for a portfolio experiment, not a validated clinical evaluation.
- Do not use this project for diagnosis, treatment, or medical decisions.

## Author

Machine-learning portfolio project for the CodeAlpha Machine Learning Internship task.

## License

Add a license only if you intend to grant others permission to reuse, modify, or distribute this project.
