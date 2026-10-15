"""Week 6 - SVM, KNN, Naive Bayes (supervised) + KMeans and PCA (unsupervised)."""
import sys; from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from src.common import out_dir

o = out_dir("week6_svm_knn_nb_clustering"); res = []
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)

models = {"SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
          "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
          "Naive Bayes": GaussianNB()}
for n, m in models.items():
    res.append(f"{n}: accuracy={accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte)):.3f}")

ks = range(1, 26)
accs = [accuracy_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(Xtr, ytr).predict(Xte)) for k in ks]
plt.plot(list(ks), accs, marker="o"); plt.xlabel("k"); plt.ylabel("accuracy"); plt.title("KNN: choosing k")
plt.savefig(o / "knn_k_sweep.png", dpi=120); plt.close()

Xs = StandardScaler().fit_transform(X)
sil = {k: silhouette_score(Xs, KMeans(k, n_init=10, random_state=42).fit_predict(Xs)) for k in range(2, 8)}
res.append("Silhouette by k: " + ", ".join(f"{k}:{v:.3f}" for k, v in sil.items()))
labels = KMeans(3, n_init=10, random_state=42).fit_predict(Xs)
Z = PCA(2).fit_transform(Xs)
plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="viridis"); plt.title("KMeans (k=3) on PCA projection")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.savefig(o / "kmeans_pca.png", dpi=120)
(o / "results.txt").write_text("\n".join(res)); print("\n".join(res))
