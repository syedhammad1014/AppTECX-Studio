"""Week 5 - Decision Trees, Random Forest, feature engineering and feature selection."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from src.common import DATA, out_dir

o = out_dir("week5_trees_features"); res = []
df = pd.read_csv(DATA / "churn_clean.csv")

df["charges_per_month"] = df.total_charges / df.tenure_months
df["is_new_customer"] = (df.tenure_months <= 12).astype(int)
df["high_support"] = (df.support_calls >= 4).astype(int)

X = pd.get_dummies(df.drop(columns=["customer_id", "churn"]), drop_first=True)
y = df.churn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

def ev(name, m, a, b):
    m.fit(a, ytr); p = m.predict(b)
    res.append(f"{name}: accuracy={accuracy_score(yte,p):.3f}  f1={f1_score(yte,p):.3f}")
    return m

for d in [2, 3, 5, 8, None]:
    ev(f"DecisionTree depth={d}", DecisionTreeClassifier(max_depth=d, random_state=42), Xtr, Xte)
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
plt.figure(figsize=(14, 6)); plot_tree(dt, feature_names=list(X.columns), class_names=["stay", "churn"], filled=True, fontsize=7)
plt.savefig(o / "decision_tree.png", dpi=120); plt.close()

rf = ev("RandomForest (all features)", RandomForestClassifier(300, random_state=42), Xtr, Xte)
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot.barh(figsize=(7, 6), title="Random Forest importances")
plt.tight_layout(); plt.savefig(o / "rf_importances.png", dpi=120); plt.close()

sel = SelectKBest(f_classif, k=6).fit(Xtr, ytr)
cols = X.columns[sel.get_support()]
res.append("Top 6 features (ANOVA F): " + ", ".join(cols))
ev("RandomForest (top 6 features)", RandomForestClassifier(300, random_state=42), Xtr[cols], Xte[cols])
(o / "results.txt").write_text("\n".join(res)); print("\n".join(res))
