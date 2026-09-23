# Project Report: Customer Churn Prediction

## Objective

Use customer and account details to estimate whether a telecom customer is likely to leave (churn) or stay. This is the Logistic Regression classification project described in Project.pdf.

## Data collection and preprocessing

- Dataset: IBM Telco Customer Churn sample dataset, 7,043 rows and 21 columns.
- Target: `Churn` (`No = 0`, `Yes = 1`).
- Removed the unique `customerID` identifier from model inputs.
- Converted `TotalCharges` to numeric and filled blank values with its median.
- Used an 80/20 train/test split with stratification and random state 42.
- Scaled numeric features with `StandardScaler`.
- One-hot encoded categorical features with `OneHotEncoder(handle_unknown="ignore")`.
- Kept preprocessing and classification together in a scikit-learn Pipeline to reuse the same transformations for new customers.

## Model

Logistic Regression with `max_iter=1000` and `class_weight="balanced"`. The full fitted pipeline is saved to `customer_churn_model.pkl` using joblib.

## Evaluation

The held-out test set contains 1,409 customers.

| Metric | Result |
|---|---:|
| Accuracy | 73.8% |
| Churn precision | 50.4% |
| Churn recall | 78.3% |
| Churn F1-score | 61.4% |
| ROC-AUC | 0.84 |

Confusion matrix:

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actual stay | 747 | 288 |
| Actual churn | 81 | 293 |

The ROC-AUC of 0.84, shown in the training output, indicates that the model separates churners from non-churners better than random ranking on this held-out test set.

### Interpretation

- **Accuracy** is the share of all test customers classified correctly.
- **Precision** indicates how often a customer flagged as churn actually churned.
- **Recall** indicates how many of the customers who churned were found by the model.
- **F1-score** balances precision and recall.
- The model found 293 of 374 churners (78.3%) and flagged 288 customers who stayed. A retention team would need to consider the cost of contacting customers who were not going to leave.

## Streamlit application

`app.py` lets the user enter a customer profile. After choosing **Predict Churn**, it shows a stay/churn prediction, predicted churn probability, and a summary table of the submitted values. The application loads the saved model pipeline.

## Limitations

This is an educational model built from a historical sample about a fictional telecom company. Its predictions are estimates, not guarantees, and its performance may change on newer data or for another provider. The classification threshold and business cost of false alarms should be considered before using predictions for customer outreach.

## Run the project

Install dependencies with `python -m pip install -r requirements.txt`, then run `python -m streamlit run app.py`. See [README.md](README.md) for the full setup steps.

