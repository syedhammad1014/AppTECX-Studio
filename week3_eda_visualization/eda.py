"""Week 3 - Exploratory Data Analysis, statistics and visualization."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.common import DATA, out_dir

df = pd.read_csv(DATA / "churn_clean.csv")
o = out_dir("week3_eda_visualization")
df.describe().round(2).to_csv(o / "statistics.csv")

print("Churn rate: %.1f%%" % (df.churn.mean()*100))
print(df.groupby("contract").churn.mean().round(3))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
sns.countplot(data=df, x="churn", ax=ax[0, 0]).set_title("Churn distribution")
sns.histplot(df, x="monthly_charges", hue="churn", kde=True, ax=ax[0, 1]).set_title("Monthly charges by churn")
sns.barplot(data=df, x="contract", y="churn", ax=ax[1, 0]).set_title("Churn rate by contract")
sns.boxplot(data=df, x="churn", y="tenure_months", ax=ax[1, 1]).set_title("Tenure by churn")
plt.tight_layout(); plt.savefig(o / "eda_overview.png", dpi=120); plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").drop(columns="customer_id").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(o / "correlation_heatmap.png", dpi=120)
