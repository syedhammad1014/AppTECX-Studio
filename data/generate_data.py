"""Generates a synthetic, intentionally messy customer-churn dataset (seeded, reproducible)."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 1500
age = rng.integers(18, 75, n).astype(float)
tenure = rng.integers(1, 72, n)
monthly = np.round(rng.normal(65, 25, n).clip(15, 140), 2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
internet = rng.choice(["DSL", "Fiber", "None"], n, p=[.35, .45, .20])
calls = rng.poisson(2, n)
gender = rng.choice(["Male", "Female"], n)
total = np.round(tenure * monthly + rng.normal(0, 30, n), 2)

z = (-1.2 + 0.03*(monthly-65) - 0.04*tenure + 0.45*calls
     + 1.2*(contract == "Month-to-month") - 0.8*(contract == "Two year")
     + 0.5*(internet == "Fiber"))
churn = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)

df = pd.DataFrame(dict(customer_id=range(1000, 1000+n), age=age, gender=gender,
    tenure_months=tenure, monthly_charges=monthly, total_charges=total,
    contract=contract, support_calls=calls, internet_service=internet, churn=churn))

# --- inject mess ---
for col, frac in [("age", .06), ("total_charges", .04), ("monthly_charges", .03)]:
    df.loc[rng.choice(n, int(n*frac), replace=False), col] = np.nan
df.loc[rng.choice(n, 15, replace=False), "monthly_charges"] = 900.0   # outliers
idx = rng.choice(n, 120, replace=False)
df.loc[idx, "gender"] = rng.choice(["male ", "FEMALE", "M", "F"], 120)
df.loc[rng.choice(n, 100, replace=False), "contract"] = "month-to-month "
df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)  # duplicates
df.to_csv(Path(__file__).parent / "churn_raw.csv", index=False)
print("saved churn_raw.csv", df.shape)
