# Model V3 Results - Main Table + Bureau + Previous Application

## Dataset
- Train shape: (307511, 122)
- Test shape: (48744, 121)
- Bureau feature table shape: (356255, 37)
- Previous application feature table shape: (356255, 39)
- Final merged train feature matrix: (307511, 195)
- Final merged test feature matrix: (48744, 195)
- Number of categorical columns: 16
- Number of numeric columns: 179
- Number of bureau-derived columns used in model: 36
- Number of previous-derived columns used in model: 38

## Cross-validation
- Number of folds: 5
- Fold AUC scores: [0.769614, 0.778521, 0.77039, 0.777536, 0.768885]
- Mean fold AUC: 0.772989
- OOF AUC: 0.772974

## Comparison
- Baseline mean fold AUC: 0.759922
- Baseline OOF AUC: 0.759895
- Bureau model mean fold AUC: 0.765376
- Bureau model OOF AUC: 0.765354
- OOF lift vs baseline: 0.013079
- OOF lift vs bureau-only model: 0.007620

## Outputs
- Submission file: C:\Coding\Home-Credit-Default-Risk\outputs\submissions\model_v3_bureau_previous.csv
- Feature importance figure: C:\Coding\Home-Credit-Default-Risk\outputs\figures\model_v3_bureau_previous_feature_importance.png

## Notes
- This model adds previous_application aggregated features on top of the main table and bureau features.
- The goal is to measure whether previous Home Credit application history improves discrimination beyond bureau-only history.