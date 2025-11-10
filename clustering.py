# cluster_ucimlrepo_heart.py
# Uses ucimlrepo to fetch UCI Heart Disease, builds mixed-feature clustering,
# selects K by silhouette, makes figures, and exports tidy outputs.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1) Load UCI Heart Disease via ucimlrepo
heart = fetch_ucirepo(name="Heart Disease")

# Combine features + targets into one DataFrame
X_raw = heart.data.features.copy()
y_raw = heart.data.targets.copy()
df = pd.concat([X_raw, y_raw], axis=1)

# Harmonize column names (lowercase, strip spaces)
df.columns = [str(c).strip().lower() for c in df.columns]

# Identify target column: common variants are 'num' or 'target'
target_col = None
for cand in ["num", "target", "disease", "diagnosis"]:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    raise ValueError("Could not find the heart-disease target column (e.g., 'num' or 'target').")

# Binary outcome: 0 vs >0
df["target_bin"] = (df[target_col] > 0).astype(int)

# 2) Feature selection (use all core variables if present)
# Canonical Cleveland columns; only keep those that exist
cont_pref = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
bin_pref  = ["sex", "fbs", "exang"]
cat_pref  = ["cp", "restecg", "slope", "thal"]

cont_cols = [c for c in cont_pref if c in df.columns]
bin_cols  = [c for c in bin_pref if c in df.columns]
cat_cols  = [c for c in cat_pref if c in df.columns]

if "oldpeak" not in cont_cols:
    raise ValueError("OLDPEAK not found—needed for ischemia burden analysis.")

# 3) Preprocess: impute, scale (continuous), one-hot (categorical)
cont_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

bin_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("cont", cont_tf, cont_cols),
        ("bin",  bin_tf,  bin_cols),
        ("cat",  cat_tf,  cat_cols),
    ],
    remainder="drop"
)

X = df[cont_cols + bin_cols + cat_cols].copy()
X_proc = preprocess.fit_transform(X)

# Build feature names after OHE (for profiles)
feature_names = []
feature_names += [f"z_{c}" for c in cont_cols]
feature_names += bin_cols
if cat_cols:
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    feature_names += list(ohe.get_feature_names_out(cat_cols))

# 4) Choose K via silhouette (k=2..6)
k_grid = range(2, 7)
sils, models = [], []

for k in k_grid:
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_proc)
    sil = silhouette_score(X_proc, labels)
    sils.append(sil)
    models.append((k, km))

best_idx = int(np.argmax(sils))
best_k, best_model = models[best_idx]
labels = best_model.predict(X_proc)
df["cluster"] = labels

print(f"[INFO] Best k by silhouette: {best_k} (score={sils[best_idx]:.3f})")

# 5) PCA to 2D for visualization only (on processed matrix)
pca = PCA(n_components=2, random_state=42)
XY = pca.fit_transform(X_proc)
df["PC1"] = XY[:, 0]
df["PC2"] = XY[:, 1]

# 6) Visualizations (matplotlib)
# (a) Silhouette vs k
plt.figure(figsize=(6,4))
plt.plot(list(k_grid), sils, marker="o")
plt.xlabel("k (clusters)")
plt.ylabel("Mean silhouette")
plt.title("Silhouette score by k")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("silhouette_by_k.png", dpi=160)

# (b) PCA scatter colored by cluster
plt.figure(figsize=(6,5))
for c in sorted(df["cluster"].unique()):
    mask = df["cluster"] == c
    plt.scatter(df.loc[mask,"PC1"], df.loc[mask,"PC2"], s=26, alpha=0.85, label=f"Cluster {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA(2D) of processed features (k={best_k})")
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_scatter_clusters.png", dpi=160)

# (c) OLDPEAK by cluster (boxplot)
plt.figure(figsize=(6,4))
data = [df.loc[df["cluster"]==c, "oldpeak"].dropna().values for c in sorted(df["cluster"].unique())]
plt.boxplot(data, labels=[f"C{c}" for c in sorted(df["cluster"].unique())])
plt.xlabel("Cluster")
plt.ylabel("OLDPEAK (mm)")
plt.title("OLDPEAK by cluster")
plt.tight_layout()
plt.savefig("oldpeak_by_cluster.png", dpi=160)

# (d) Outcome prevalence by cluster (bar)
prev = df.groupby("cluster")["target_bin"].mean().reindex(sorted(df["cluster"].unique()))
plt.figure(figsize=(6,4))
plt.bar([f"C{c}" for c in prev.index], prev.values)
plt.xlabel("Cluster")
plt.ylabel("Outcome prevalence (target>0)")
plt.title("CVD outcome rate by cluster")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("outcome_rate_by_cluster.png", dpi=160)

# (e) Cluster profile heatmap: z-means for continuous, proportions for binary/OHE
X_df = pd.DataFrame(X_proc, columns=feature_names, index=df.index)
X_df["cluster"] = df["cluster"]
profile = X_df.groupby("cluster").mean().reindex(sorted(df["cluster"].unique()))

# Clip for readability
prof_clip = profile.clip(lower=-1.5, upper=1.5)
plt.figure(figsize=(min(12, 2 + 0.25*len(profile.columns)), 3 + 0.6*best_k))
im = plt.imshow(prof_clip.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
plt.colorbar(im, fraction=0.02, pad=0.04, label="Std. mean / proportion")
plt.yticks(range(len(profile.index)), [f"C{c}" for c in profile.index])
plt.xticks(range(len(profile.columns)), profile.columns, rotation=90, fontsize=8)
plt.title("Cluster profile heatmap\n(continuous=z-mean, binary/one-hot=proportion)")
plt.tight_layout()
plt.savefig("cluster_profile_heatmap.png", dpi=180)

# 7) Summary table
outdir = Path(".")
df.to_csv(outdir / "heart_with_clusters.csv", index=False)

summary = df.groupby("cluster").agg(
    n=("cluster","size"),
    oldpeak_mean=("oldpeak","mean"),
    oldpeak_sd=("oldpeak","std"),
    outcome_rate=("target_bin","mean"),
    age_mean=("age","mean") if "age" in df.columns else ("cluster", "size"),
    trestbps_mean=("trestbps","mean") if "trestbps" in df.columns else ("cluster","size"),
    chol_mean=("chol","mean") if "chol" in df.columns else ("cluster","size"),
    thalach_mean=("thalach","mean") if "thalach" in df.columns else ("cluster","size")
).reset_index()
summary.to_csv(outdir / "cluster_summary.csv", index=False)

print("\n[DONE] Wrote files:")
for fn in [
    "silhouette_by_k.png",
    "pca_scatter_clusters.png",
    "oldpeak_by_cluster.png",
    "outcome_rate_by_cluster.png",
    "cluster_profile_heatmap.png",
    "heart_with_clusters.csv",
    "cluster_summary.csv",
]:
    print(" -", fn)

