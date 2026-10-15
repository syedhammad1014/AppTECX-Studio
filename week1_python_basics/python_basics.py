"""Week 1 - Python for AI/ML: NumPy, pandas, functions, file I/O on the Iris dataset."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from src.common import out_dir

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")

def summarize(frame):
    """Return mean/std/min/max for every numeric column."""
    return frame.describe().T[["mean", "std", "min", "max"]].round(3)

arr = df.iloc[:, :4].to_numpy()
print("NumPy shape:", arr.shape)
print("Column means:", arr.mean(axis=0).round(3))
print("Correlation (sepal length vs petal length): %.3f" % np.corrcoef(arr[:, 0], arr[:, 2])[0, 1])
print(df.groupby("species").mean().round(3))

o = out_dir("week1_python_basics")
df.to_csv(o / "iris.csv", index=False)
summarize(df).to_csv(o / "iris_summary.csv")
df.groupby("species").mean().round(3).to_csv(o / "iris_group_means.csv")
