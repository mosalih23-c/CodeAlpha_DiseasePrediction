
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/heart_disease.csv")
OUTPUT_DIR = Path("data")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Convert target to binary classification
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

# 3. Impute missing values using column medians
for column in ["ca", "thal"]:
    df[column] = df[column].fillna(df[column].median())

# 4. Save cleaned dataset
clean_path = OUTPUT_DIR / "heart_disease_clean.csv"
df.to_csv(clean_path, index=False)

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)
print("Shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())
print("\nTarget percentages:")
print(
    (df["target"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

# 5. Target distribution chart
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(REPORT_DIR / "target_distribution.png", dpi=150)
plt.close()

# 6. Feature correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Heart Disease Feature Correlations")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print("\nSaved cleaned data to:", clean_path)
print("Saved charts in:", REPORT_DIR)
