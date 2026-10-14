
from ucimlrepo import fetch_ucirepo
from pathlib import Path
import pandas as pd

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Downloading Heart Disease dataset...")

# UCI Heart Disease dataset (ID: 45)
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save raw dataset
output_path = DATA_DIR / "heart_disease.csv"
df.to_csv(output_path, index=False)

print("\nDataset downloaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
