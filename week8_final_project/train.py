"""Week 8 - Final project: Customer Churn Prediction (end-to-end pipeline)."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from src.common import DATA, out_dir

o = out_dir("week8_final_project")
df = pd.read_csv(DATA / "churn_clean.csv")
num = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
cat = ["gender", "contract", "internet_service"]
X, y = df[num + cat], df.churn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

pipe = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
    ("clf", RandomForestClassifier(300, max_depth=10, min_samples_leaf=5, random_state=42)),
]).fit(Xtr, ytr)

p = pipe.predict(Xte)
metrics = dict(accuracy=accuracy_score(yte, p), precision=precision_score(yte, p),
               recall=recall_score(yte, p), f1=f1_score(yte, p))
json.dump({k: round(v, 4) for k, v in metrics.items()}, open(o / "metrics.json", "w"), indent=2)
(o / "classification_report.txt").write_text(classification_report(yte, p))
joblib.dump(pipe, o / "churn_model.joblib"); print(metrics)
