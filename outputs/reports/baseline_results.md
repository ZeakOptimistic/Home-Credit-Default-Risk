# Baseline Results

## Dataset
- Train shape: (307511, 123)
- Test shape: (48744, 122)
- Positive rate: 0.0807
- Number of categorical columns: 16
- Number of numeric columns: 105

## Cross-validation
- Number of folds: 5
- Fold AUC scores: [0.755092, 0.765237, 0.758302, 0.76516, 0.755821]
- Mean fold AUC: 0.759922
- OOF AUC: 0.759895

## Outputs
- Submission file: c:\Coding\Home-Credit-Default-Risk\outputs\submissions\baseline_main_table_catboost.csv
- Feature importance figure: c:\Coding\Home-Credit-Default-Risk\outputs\figures\baseline_main_table_feature_importance.png

## Notes
- This baseline uses only the main application table.
- This benchmark will be compared against later feature-engineered models.