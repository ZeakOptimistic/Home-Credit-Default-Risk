# Home Credit Default Risk

## 1. Problem Statement
Home Credit Default Risk is a binary classification problem: predict the probability that a loan applicant will have repayment difficulties (`TARGET=1`). The competition metric is ROC-AUC, so the project focuses on ranking risky applicants correctly rather than maximizing plain accuracy.

## 2. Dataset Structure
The dataset is relational, not a single flat table. `application_train` and `application_test` are the core modeling tables, while `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `POS_CASH_balance`, and `credit_card_balance` provide historical credit behavior. Because these auxiliary tables contain one-to-many and monthly-history records, they must be aggregated to the customer level (`SK_ID_CURR`) before modeling.

## 3. Project Structure
```text
Home-Credit-Default-Risk/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_main_table.ipynb
│   ├── 03_baseline_main_table.ipynb
│   ├── 04_bureau_features.ipynb
│   ├── 05_model_with_bureau.ipynb
│   ├── 06_previous_application_features.ipynb
│   ├── 07_model_with_previous.ipynb
│   ├── 08_payment_history_features.ipynb
│   ├── 09_model_with_payments.ipynb
│   ├── 10_error_analysis.ipynb
│   └── 11_final_submission_and_report.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   ├── features_bureau.py
│   ├── features_previous.py
│   └── features_payments.py
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── submissions/
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. Methodology
1. Audit the dataset structure and join keys.
2. Perform EDA on the main application table.
3. Build a baseline model using only the main table.
4. Engineer relational features from bureau history.
5. Add previous application features.
6. Add payment-history features from installments, POS cash, and credit card balance.
7. Train a full 5-fold CatBoost model and evaluate with OOF ROC-AUC.
8. Perform error analysis on false positives, false negatives, and feature-family importance.

## 5. Feature Engineering Overview
Feature engineering is the core of this project. I created:
- **Main application features**: cleaned anomalies and useful ratios
- **Bureau features**: external credit history and active/closed debt behavior
- **Previous application features**: approval/refusal behavior and prior loan application patterns
- **Payment-history features**: installment lateness, POS delinquency, and revolving credit card behavior

The key idea is to aggregate all history tables to the same customer level before merging.

## 6. Modeling Results
Model comparison based on cross-validated OOF ROC-AUC:

| model_name | mean_fold_auc | oof_auc | lift_vs_previous | lift_vs_baseline |
| --- | --- | --- | --- | --- |
| Baseline (Main Table) | 0.759922 | 0.759895 | N/A | 0.000000 |
| Main + Bureau | 0.765376 | 0.765354 | 0.005459 | 0.005459 |
| Main + Bureau + Previous | 0.772989 | 0.772974 | 0.007620 | 0.013079 |
| Main + Bureau + Previous + Payments | 0.785389 | 0.785274 | 0.012300 | 0.025379 |

Best model: **Main + Bureau + Previous + Payments**
Best OOF AUC: **0.785274**

Generated figures:
- `outputs/figures/final_report/01_model_comparison.png`
- `outputs/figures/final_report/02_project_pipeline_overview.png`

## 7. How to Reproduce
1. Clone the repository.
2. Place the full Kaggle competition data into `data/raw/`.
3. Install dependencies from `requirements.txt`.
4. Run notebooks in order:
   - `01_data_audit.ipynb`
   - `02_eda_main_table.ipynb`
   - `03_baseline_main_table.ipynb`
   - `04_bureau_features.ipynb`
   - `05_model_with_bureau.ipynb`
   - `06_previous_application_features.ipynb`
   - `07_model_with_previous.ipynb`
   - `08_payment_history_features.ipynb`
   - `09_model_with_payments.ipynb`
   - `10_error_analysis.ipynb`
   - `11_final_submission_and_report.ipynb`
5. Use the generated file in `outputs/submissions/final_submission.csv` for Kaggle submission.

## Notes
- Do not commit raw data to GitHub.
- The dataset is relational and requires aggregation from one-to-many history tables to the customer level (`SK_ID_CURR`).
- The strongest gains in this project come from relational feature engineering, not from trying many different model families.