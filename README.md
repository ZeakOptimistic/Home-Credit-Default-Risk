# Home Credit Default Risk

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

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
│   ├── 01_eda_and_audit.ipynb
│   ├── 02_feature_engineering_and_modeling.ipynb
│   └── 03_error_analysis_and_submission.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── inference.py
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   └── submissions/
├── Makefile
├── requirements.txt
└── README.md
```

## 4. Methodology
1. Audit the dataset structure and join keys.
2. Perform EDA on the main application table.
3. Build a baseline model using only the main table.
4. Engineer relational features from bureau history, previous applications, and payment history.
5. Train a full 5-fold CatBoost model and evaluate with OOF ROC-AUC.
6. Perform error analysis on false positives, false negatives, and feature-family importance.

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
- `outputs/figures/01_model_comparison.png`
- `outputs/figures/02_project_pipeline_overview.png`

## 7. How to Reproduce
1. Clone the repository.
2. Place the full Kaggle competition data into `data/raw/` (or use your own sample data).
3. Install dependencies from `requirements.txt` via:
   ```bash
   pip install -r requirements.txt
   ```
4. **Interactive Exploration Mode**: You can walk through the data manually using the Jupyter notebooks.
   - `01_eda_and_audit.ipynb`
   - `02_feature_engineering_and_modeling.ipynb`
   - `03_error_analysis_and_submission.ipynb`
   
5. **Full Pipeline Mode**: You can auto-generate the complete feature set and train the algorithms automatically by executing the entrypoint script.
   ```bash
   make run
   # Or simply: python main.py
   ```
6. The fully trained predictions will generate in `outputs/submissions/final_submission_lgbm.csv`.

## Notes
- Do not commit raw data to GitHub (`data/` and `outputs/` are gitignored).
- Formatting is governed by `black` and `isort` which can be run using `make format`.
- The strongest gains in this project come from relational feature engineering, specifically capturing lateness and limit behaviors within the client financial histories.