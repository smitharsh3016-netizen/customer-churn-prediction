# Customer Churn Prediction

An end-to-end machine-learning project that estimates whether a telecom customer is likely to leave (churn) or stay. It includes data preparation, Logistic Regression, model evaluation and saving, a Streamlit interface, and a command-line prediction example.

## Project workflow

1. Load the IBM Telco Customer Churn sample dataset.
2. Convert `TotalCharges` to numeric and fill blank values with the median.
3. Remove `customerID`, encode the target (`No = 0`, `Yes = 1`), and split the data into stratified 80% training and 20% test sets.
4. Scale numeric features and one-hot encode categorical features in a scikit-learn `ColumnTransformer`.
5. Train a Logistic Regression classifier with balanced class weights.
6. Evaluate on the held-out test set and save the complete preprocessing/model pipeline.
7. Run the Streamlit form to enter a customer profile, get a churn prediction and probability, and review the input summary.

## Dataset

The IBM sample contains 7,043 customer rows and 21 columns, including demographics, account tenure, services, billing information, and the `Churn` target. The customer identifier is excluded from model inputs. The CSV is included in the working project folder. If it is missing after cloning the GitHub repository, `train.py` downloads it from the public IBM sample-data repository and saves it under `data/`.

The data describes a **fictional** telecommunications company. It is an educational sample, not a real customer export. Source: [IBM Telco Customer Churn sample data](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). IBM's description is available in its [Telco customer churn sample overview](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

## Model and evaluation

The model is a scikit-learn Pipeline containing the preprocessing steps and a `LogisticRegression(max_iter=1000, class_weight="balanced")` classifier. The same pipeline is saved to `customer_churn_model.pkl`, so predictions use the same transformations as training.

Results on the stratified 20% test split (1,409 customers), as recorded in [REPORT.md](REPORT.md):

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

The model's ROC-AUC is 0.84. Recall is relatively high: the model identifies many customers who churned, while also flagging some customers who stayed. This makes it a starting point for retention prioritization, not a final decision system. Predictions are estimates; results depend on this historical sample and may not generalize to another company or time period.

## Project files

- `train.py` — data preparation, model training, evaluation, and model saving.
- `app.py` — interactive Streamlit prediction interface.
- `predict.py` — example prediction for one customer from the terminal.
- `data/customer_churn.csv` — dataset used by the project.
- `customer_churn_model.pkl` — saved fitted preprocessing and model pipeline.
- `REPORT.md` — project and evaluation summary.
- `screenshots/` — screenshots of the Streamlit app and prediction output.

## Run locally

### 1. Install Python

Use Python 3.12 or newer. Open a terminal in this project folder.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the commands below with `.venv\Scripts\python.exe` instead of activating it.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the application

Train once to download the data if needed, evaluate the model, and save `customer_churn_model.pkl`:

```powershell
python train.py
```

Then launch the app:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`). Enter customer details and choose **Predict Churn**.

### Retrain the model (optional)

```powershell
python train.py
```

This reads `data/customer_churn.csv` (downloading it if it is not present), prints evaluation metrics, saves `customer_churn_model.pkl`, and displays the evaluation plots.

### Run the example prediction (optional)

```powershell
python predict.py
```

## Screenshots

See the `screenshots/` folder for the app and prediction screenshots.

## Limitations

- The dataset is an educational sample and contains a class imbalance.
- A predicted probability is not a guarantee that a customer will leave.
- The model may perform differently on newer data, different regions, or another telecom provider.
- False positives and false negatives have different business costs; the decision threshold should be chosen with the intended use in mind.

