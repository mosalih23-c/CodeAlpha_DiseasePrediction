
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/heart_disease_clean.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X.shape[1])

# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )
}

results = []
fitted_models = {}
probabilities = {}

for name, classifier in models.items():
    # Scaling is useful for Logistic Regression and SVM.
    # Using a pipeline keeps preprocessing fitted only on training data.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probs)
    }

    results.append(metrics)
    fitted_models[name] = pipeline
    probabilities[name] = probs

    print(pd.Series(metrics).to_string())
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Disease", "Disease"],
        zero_division=0
    ))

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(
        pipeline,
        MODEL_DIR / f"{safe_name}.joblib"
    )

# 4. Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="ROC-AUC", ascending=False
).reset_index(drop=True)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False, float_format="%.4f"))

# 5. Save metrics as JSON
with open(REPORT_DIR / "model_metrics.json", "w") as f:
    json.dump(
        results_df.to_dict(orient="records"),
        f,
        indent=4
    )

# 6. Plot model comparison
metrics_to_plot = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)
plt.ylim(0, 1)
plt.title("Heart Disease Model Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "model_comparison.png",
    dpi=150
)
plt.close()

# 7. Confusion matrix and ROC curves for the best ROC-AUC model
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
best_predictions = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    display_labels=["No Disease", "Disease"],
    cmap="Blues"
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix_best_model.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(8, 6))
ax = plt.gca()

for name, probs in probabilities.items():
    RocCurveDisplay.from_predictions(
        y_test,
        probs,
        name=name,
        ax=ax
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves - Heart Disease Models")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "roc_curves.png",
    dpi=150
)
plt.close()

# 8. Save best model based on ROC-AUC
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

with open(REPORT_DIR / "best_model.txt", "w") as f:
    f.write(f"Best model by test ROC-AUC: {best_name}\n")
    f.write(
        results_df.iloc[0].to_string()
    )

print(f"\nBest model by ROC-AUC: {best_name}")
print("Saved models to:", MODEL_DIR)
print("Saved reports to:", REPORT_DIR)
