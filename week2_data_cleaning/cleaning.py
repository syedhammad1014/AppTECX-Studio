"""Week 2 - Data collection, cleaning, preprocessing and transformation."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from src.common import DATA, out_dir

df = pd.read_csv(DATA / "churn_raw.csv")
log = []
log.append(f"Raw shape: {df.shape}")
log.append("Missing values before (internet_service NaN = customers with no internet):\n" + df.isna().sum().to_string())

n0 = len(df); df = df.drop_duplicates(); log.append(f"Duplicates removed: {n0-len(df)}")

# standardise text
df["gender"] = df["gender"].str.strip().str.lower().map(
    {"male": "male", "m": "male", "female": "female", "f": "female"})
df["contract"] = df["contract"].str.strip().str.lower()
# pandas reads the string "None" as NaN, so give "no internet" an explicit label
df["internet_service"] = df["internet_service"].fillna("no_internet").str.strip().str.lower()

# outliers (IQR cap)
q1, q3 = df["monthly_charges"].quantile([.25, .75]); iqr = q3 - q1
hi = q3 + 1.5*iqr; lo = q1 - 1.5*iqr
n_out = int((df["monthly_charges"] > hi).sum())
df["monthly_charges"] = df["monthly_charges"].clip(lo, hi)
log.append(f"Outliers capped in monthly_charges: {n_out} (upper bound {hi:.2f})")

# missing values
df["age"] = df["age"].fillna(df["age"].median())
df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
df["total_charges"] = df["total_charges"].fillna(df["tenure_months"] * df["monthly_charges"])
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
log.append("Missing values after:\n" + df.isna().sum().to_string())

df.to_csv(DATA / "churn_clean.csv", index=False)
log.append(f"Clean shape: {df.shape}")
(out_dir("week2_data_cleaning") / "cleaning_log.txt").write_text("\n\n".join(log))
print("\n\n".join(log))
