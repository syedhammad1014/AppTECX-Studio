"""Week 7 - Training/validation/testing, metrics, confusion matrix, ROC, hyper-parameter tuning."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from src.common import DATA, out_dir

o = out_dir("week7_model_evaluation")
df = pd.read_csv(DATA / "churn_clean.csv")
X = pd.get_dummies(df.drop(columns=["customer_id", "churn"]), drop_first=True); y = df.churn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(300, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "Naive Bayes": GaussianNB(),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
fig, ax = plt.subplots(figsize=(6, 5))
for n, m in models.items():
    cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="f1")
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append(dict(model=n, cv_f1_mean=cvs.mean(), accuracy=accuracy_score(yte, p),
                     precision=precision_score(yte, p), recall=recall_score(yte, p), f1=f1_score(yte, p)))
    RocCurveDisplay.from_estimator(m, Xte, yte, ax=ax, name=n)
ax.set_title("ROC curves"); plt.tight_layout(); plt.savefig(o / "roc_curves.png", dpi=120); plt.close()

tbl = pd.DataFrame(rows).round(3).sort_values("f1", ascending=False)
tbl.to_csv(o / "model_comparison.csv", index=False); print(tbl.to_string(index=False))

gs = GridSearchCV(RandomForestClassifier(random_state=42),
                  {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 5]},
                  cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
best = f"Best RF params: {gs.best_params_}  CV f1={gs.best_score_:.3f}  Test f1={f1_score(yte, gs.predict(Xte)):.3f}"
(o / "grid_search.txt").write_text(best); print(best)
ConfusionMatrixDisplay.from_estimator(gs.best_estimator_, Xte, yte, display_labels=["stay", "churn"])
plt.title("Confusion matrix (tuned RF)"); plt.savefig(o / "confusion_matrix.png", dpi=120)
