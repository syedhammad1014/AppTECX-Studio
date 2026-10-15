"""Predict churn for one customer using the trained model. Run train.py first."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import joblib, pandas as pd
from src.common import out_dir

model = joblib.load(out_dir("week8_final_project") / "churn_model.joblib")
customer = pd.DataFrame([dict(age=34, gender="male", tenure_months=3, monthly_charges=95.0,
                              total_charges=285.0, support_calls=5, contract="month-to-month",
                              internet_service="fiber")])
prob = model.predict_proba(customer)[0, 1]
print(f"Churn probability: {prob:.1%} -> {'LIKELY TO CHURN' if prob > .5 else 'likely to stay'}")
