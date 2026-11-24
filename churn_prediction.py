import pandas as pd
import numpy as np
from pathlib import Path
import joblib
# Load Excel dataset
file_path = Path(__file__).resolve().parent / "European_Bank .xlsx"

if not file_path.exists():
    raise FileNotFoundError("Excel dataset not found!")

# Read Excel workbook
excel = pd.ExcelFile(file_path)

print("Dataset loaded successfully!")
print("File:", file_path.name)

# Display available sheets
print("\nAvailable Sheets:")
print(excel.sheet_names)
# Load Raw Data
df = pd.read_excel(file_path, sheet_name=excel.sheet_names[0])
# Display dataset information
print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())
# Customer Churn Analysis

print("\nCUSTOMER CHURN DISTRIBUTION")

churn_counts = df["Exited"].value_counts()

print("Customers Stayed (0):", churn_counts.get(0, 0))
print("Customers Exited (1):", churn_counts.get(1, 0))

# Calculate churn percentage
churn_rate = df["Exited"].mean() * 100

print("\nOverall Churn Rate:", round(churn_rate, 2), "%")

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())
# STEP 11: MACHINE LEARNING MODEL

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
# Separate features and target
# FEATURE ENGINEERING

# 1. Balance to Salary Ratio
df["BalanceToSalaryRatio"] = (
    df["Balance"] / (df["EstimatedSalary"] + 1)
)

# 2. Product Density
df["ProductDensity"] = (
    df["NumOfProducts"] / 4
)

# 3. Engagement-Product Interaction
df["EngagementProductInteraction"] = (
    df["IsActiveMember"] * df["NumOfProducts"]
)

# 4. Age-Tenure Interaction
df["AgeTenureInteraction"] = (
    df["Age"] * df["Tenure"]
)
X = df.drop(columns=["Exited", "CustomerId", "Surname", "Year"],
            errors="ignore")
y = df["Exited"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Identify numerical columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nADDITIONAL MODEL EVALUATION")

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("Precision:", round(precision_score(y_test, y_pred), 3))

print("Recall:", round(recall_score(y_test, y_pred), 3))

print("F1 Score:", round(f1_score(y_test, y_pred), 3))

y_prob = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
# COMPARE MACHINE LEARNING MODELS

models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest": model,

    "Gradient Boosting": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
}

results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print(results_df.round(3))

results_df.to_excel("model_comparison.xlsx", index=False)

print("Model comparison saved successfully!")
print("\nRANDOM FOREST MODEL RESULTS")

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
joblib.dump(model, "churn_model.pkl")
print("Model saved successfully!")
# FEATURE IMPORTANCE ANALYSIS

print("\nFEATURE IMPORTANCE")

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance = model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

print(importance_df.head(15))

importance_df.to_excel("feature_importance.xlsx", index=False)

print("Feature importance saved successfully!")
# SHAP EXPLAINABILITY ANALYSIS

import shap
import matplotlib.pyplot as plt

print("\nSHAP EXPLAINABILITY ANALYSIS")

# Transform test data
X_test_transformed = model.named_steps["preprocessor"].transform(X_test)

# Convert sparse data to dense format if needed
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# Select 200 test records
X_shap_sample = X_test_transformed[:200]

# Create SHAP explainer
explainer = shap.TreeExplainer(
    model.named_steps["classifier"]
)

# Calculate SHAP values
shap_values = explainer.shap_values(X_shap_sample)

# Select churn class (class 1)
if isinstance(shap_values, list):
    shap_values_churn = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_churn = shap_values[:, :, 1]
else:
    shap_values_churn = shap_values

# Get feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Create and save SHAP summary chart
shap.summary_plot(
    shap_values_churn,
    X_shap_sample,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("SHAP summary chart saved successfully!")
# MODEL PERFORMANCE COMPARISON CHART

import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

comparison_plot = results_df.set_index("Model")[metrics]

comparison_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(title="Evaluation Metrics")
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("Model comparison chart saved successfully!")
print("SHAP values calculated successfully!")
