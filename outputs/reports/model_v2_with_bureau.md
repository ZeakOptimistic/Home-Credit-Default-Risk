# Model V2 Results - Main Table + Bureau

## Dataset
- Train shape: (307511, 122)
- Test shape: (48744, 121)
- Bureau feature table shape: (356255, 37)
- Final merged train feature matrix: (307511, 157)
- Final merged test feature matrix: (48744, 157)
- Number of categorical columns: 16
- Number of numeric columns: 141
- Number of bureau-derived columns used in model: 36

## Cross-validation
- Number of folds: 5
- Fold AUC scores: [0.760744, 0.770255, 0.764195, 0.770144, 0.761541]
- Mean fold AUC: 0.765376
- OOF AUC: 0.765354

## Baseline Comparison
- Baseline mean fold AUC: 0.759922
- Baseline OOF AUC: 0.759895
- OOF lift vs baseline: 0.005459

## Outputs
- Submission file: C:\Coding\Home-Credit-Default-Risk\outputs\submissions\model_v2_bureau.csv
- Feature importance figure: C:\Coding\Home-Credit-Default-Risk\outputs\figures\model_v2_bureau_feature_importance.png

## Notes
- This model adds bureau and bureau_balance aggregated features on top of the main application table.
- Compare whether external credit history improves discrimination over the baseline-only model.