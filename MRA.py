# ============================================
# Multiple Regression on OLDPEAK + Residual Plot (with custom legend colors)
# ============================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

sns.set(style="whitegrid")

# -----------------------
# 1. Load & clean data
# -----------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, header=None, names=cols)
df = df.replace("?", np.nan)

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().reset_index(drop=True)
df["target_binary"] = (df["target"] > 0).astype(int)  # 0 = no disease, 1 = yes

print("Rows after cleaning:", df.shape[0])

# -----------------------
# 2. Define Y and X
# -----------------------
y = df["oldpeak"]
predictors = [
    "age", "sex", "trestbps", "chol", "thalach",
    "fbs", "cp", "restecg", "exang", "slope", "thal", "ca"
]

X = df[predictors].copy()
cat_vars = ["cp", "restecg", "slope", "thal"]
X = pd.get_dummies(X, columns=cat_vars, drop_first=True)
X = sm.add_constant(X)

# -----------------------
# 3. Fit model
# -----------------------
model = sm.OLS(y, X).fit()
print(model.summary())

# -----------------------
# 4. Residuals vs Fitted Plot (custom colors)
# -----------------------
fitted = model.fittedvalues
resid = model.resid

# Define consistent blue/orange palette
palette = {0: "#1f77b4", 1: "#ff7f0e"}  # blue for No, orange for Yes

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=fitted,
    y=resid,
    hue=df["target_binary"],
    palette=palette,
    alpha=0.8,
    s=60
)
plt.axhline(0, color="red", linestyle="--", lw=1)
plt.xlabel("Fitted values (Predicted oldpeak)")
plt.ylabel("Residuals (Observed - Fitted)")
plt.title("Residuals vs Fitted Values for Multiple Regression on OLDPEAK")

# Manually adjust legend to enforce correct labels and order
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles=handles,
    labels=["No disease", "Heart disease"],
    title="Heart disease",
    loc="upper right"
)

plt.tight_layout()
# plt.savefig("residuals_vs_fitted_oldpeak.png", dpi=300)
plt.show()

# -----------------------
# 5.  Q–Q plot
# -----------------------
sm.qqplot(resid, line="45", fit=True)
plt.title("Q–Q Plot of Regression Residuals")
plt.tight_layout()
plt.show()
