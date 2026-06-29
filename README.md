
# Telco Customer Churn Prediction

End-to-end ML system to predict customer churn, deployed as a REST API.

## Results
- **Model:** XGBoost (tuned)
- **Recall:** 93.9% — catches 351 out of 374 churners
- **AUC:** 0.838
- **Business impact:** 133,632 EUR annual revenue saved vs baseline

## Tech Stack
- ML: XGBoost, Scikit-learn, SHAP
- Experiment tracking: MLflow
- API: FastAPI
- Containerisation: Docker
- Language: Python 3.11

## How to Run

With Docker:
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api

Without Docker:
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

API: http://localhost:8000
Docs: http://localhost:8000/docs

## Model Comparison

| Model | Recall | AUC |
|-------|--------|-----|
| Baseline LR | 0.527 | 0.836 |
| Weighted LR | 0.807 | 0.835 |
| SMOTE LR | 0.746 | 0.821 |
| XGBoost Default | 0.652 | 0.803 |
| XGBoost Tuned | 0.939 | 0.838 |

## Key Findings
- Month-to-month customers churn 15x more than two-year contract customers
- New customers (0-12 months) churn at 47.7% — critical retention window
- Top drivers: tenure, contract type, high value at risk, online security

## Author
Pankaj Dawani — Senior Data Scientist
