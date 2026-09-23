from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the included dataset. If it is missing (for example, after a fresh
# GitHub clone), download the public IBM sample and save a local copy.
DATA_PATH = Path("data/customer_churn.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)
else:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_URL)
    data.to_csv(DATA_PATH, index=False)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Fill missing values
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

# Remove customer ID
data = data.drop(columns=["customerID"])

# Convert Churn into 0 and 1
data["Churn"] = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Separate input features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Find numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create preprocessing system
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numbers",
            StandardScaler(),
            numeric_features
        ),
        (
            "words",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Display information
print("Dataset loaded successfully!")
print("Dataset size:", data.shape)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nChurn count:")
print(data["Churn"].value_counts())

print("\nPreprocessor created successfully!")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
   ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel trained successfully!")

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions))

import joblib

joblib.dump(model, "customer_churn_model.pkl")

print("\nModel saved successfully as customer_churn_model.pkl")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["No Churn", "Churn"],
    cmap="Blues"
)

plt.title("Customer Churn Confusion Matrix")
plt.show()

probabilities = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, probabilities)

print("\nROC-AUC Score:")
print(roc_auc)

RocCurveDisplay.from_predictions(y_test, probabilities)
plt.title("Customer Churn ROC Curve")
plt.show()

