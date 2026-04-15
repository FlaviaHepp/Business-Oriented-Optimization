# =========================
# BUSINESS PROBLEM
# =========================
"""
Objetivo:
Predecir qué clientes tienen mayor probabilidad de abandonar (churn)
para optimizar estrategias de retención.

En banca, retener un cliente es significativamente más barato que adquirir uno nuevo.
Por lo tanto, el objetivo no es solo predecir churn, sino priorizar clientes
en función de su impacto económico.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", None)

df = pd.read_csv("Dataset.csv")
df.columns = df.columns.str.strip().str.lower()

df = df.drop(columns=["rownumber", "customerid", "surname"], errors="ignore")

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["exited"].value_counts(normalize=True))

plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "white"

sns.boxplot(x="exited", y="age", data=df)
plt.show()

sns.boxplot(x="exited", y="balance", data=df)
plt.show()

sns.countplot(x="numofproducts", hue="exited", data=df)
plt.show()

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["exited"])
y = df["exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

def calculate_profit_proba(y_true, y_proba, threshold=0.5, gain=100, contact_cost=20):
    y_pred = (y_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (tp * gain) - ((tp + fp) * contact_cost)


def calculate_profit_binary(y_true, y_pred, gain=100, contact_cost=20):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (tp * gain) - ((tp + fp) * contact_cost)

thresholds = np.arange(0.10, 0.91, 0.05)
results = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    profit = calculate_profit_proba(y_test, y_proba, threshold=t)

    results.append({
        "threshold": t,
        "profit": profit
    })

df_business = pd.DataFrame(results)

best_row = df_business.sort_values("profit", ascending=False).iloc[0]
best_threshold = best_row["threshold"]
best_profit = best_row["profit"]

y_pred_model = (y_proba >= best_threshold).astype(int)

print("Threshold óptimo:", best_threshold)
print("Profit máximo:", best_profit)
print("Clientes a contactar:", y_pred_model.sum())

plt.plot(df_business["threshold"], df_business["profit"], marker="o")
plt.axvline(best_threshold, linestyle="--")
plt.show()

np.random.seed(42)
num_targets = y_pred_model.sum()

y_pred_random = np.zeros_like(y_test)
random_indices = np.random.choice(len(y_test), size=num_targets, replace=False)
y_pred_random[random_indices] = 1

profit_model = calculate_profit_binary(y_test, y_pred_model)
profit_random = calculate_profit_binary(y_test, y_pred_random)

print("Profit modelo:", profit_model)
print("Profit random:", profit_random)

import shap

# Sample
X_sample = X_test.sample(200, random_state=42)

# Explainer
explainer = shap.TreeExplainer(model)

# Calcular shap values
shap_values = explainer(X_sample)

# =========================
# GLOBAL INTERPRETATION
# =========================

# Summary plot
shap.summary_plot(shap_values.values[:, :, 1], X_sample)

# Bar plot
shap.summary_plot(shap_values.values[:, :, 1], X_sample, plot_type="bar")

# =========================
# LOCAL INTERPRETATION
# =========================

i = 0

shap.force_plot(
    explainer.expected_value[1],
    shap_values.values[i, :, 1],
    X_sample.iloc[i],
    matplotlib=True
)