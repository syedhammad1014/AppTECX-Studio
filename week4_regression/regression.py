"""Week 4 - Supervised learning: Linear Regression (diabetes) and Logistic Regression (churn)."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from src.common import DATA, out_dir

o = out_dir("week4_regression"); res = []

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
lr = LinearRegression().fit(Xtr, ytr); p = lr.predict(Xte)
res.append(f"Linear Regression (diabetes): R2={r2_score(yte,p):.3f}  RMSE={np.sqrt(mean_squared_error(yte,p)):.2f}")
plt.scatter(yte, p, alpha=.6); plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Linear Regression: actual vs predicted")
plt.savefig(o / "linear_regression.png", dpi=120); plt.close()

df = pd.read_csv(DATA / "churn_clean.csv")
X = pd.get_dummies(df.drop(columns=["customer_id", "churn"]), drop_first=True)
Xtr, Xte, ytr, yte = train_test_split(X, df.churn, test_size=.2, random_state=42, stratify=df.churn)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
p = m.predict(Xte)
res.append(f"Logistic Regression (churn): accuracy={accuracy_score(yte,p):.3f}\n" + classification_report(yte, p))
coef = pd.Series(m[-1].coef_[0], index=X.columns).sort_values()
coef.plot.barh(figsize=(7, 5), title="Logistic regression coefficients"); plt.tight_layout()
plt.savefig(o / "logistic_coefficients.png", dpi=120)
(o / "results.txt").write_text("\n\n".join(res)); print("\n\n".join(res))
