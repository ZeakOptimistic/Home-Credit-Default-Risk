# Model V4 Results - Main Table + Bureau + Previous + Payment History

## Dataset
- Train shape: (307511, 122)
- Test shape: (48744, 121)
- Bureau feature table shape: (356255, 37)
- Previous application feature table shape: (356255, 39)
- Payment history feature table shape: (356255, 115)
- Final merged train feature matrix: (307511, 310)
- Final merged test feature matrix: (48744, 309)
- Number of categorical columns: 16
- Number of numeric columns: 292

## Cross-validation
- Number of folds: 5
- Fold AUC scores: [0.781515, 0.790077, 0.783888, 0.78959, 0.781873]
- Mean fold AUC: 0.785389
- OOF AUC: 0.785274

## Comparison
- Baseline OOF AUC: 0.759895
- Bureau + Previous OOF AUC: 0.772974
- OOF lift vs baseline: 0.025379
- OOF lift vs bureau + previous: 0.012300

## Outputs
- Submission file: C:\Coding\Home-Credit-Default-Risk\outputs\submissions\model_v4_with_payments.csv
- OOF predictions file: C:\Coding\Home-Credit-Default-Risk\outputs\reports\model_v4_with_payments_oof.csv
- Feature importance figure: C:\Coding\Home-Credit-Default-Risk\outputs\figures\model_v4_with_payments_feature_importance.png
- Feature importance table: C:\Coding\Home-Credit-Default-Risk\outputs\reports\model_v4_with_payments_feature_importance.csv

## Top 20 Features
- EXT_SOURCE_3: 8.769960
- EXT_SOURCE_2: 7.133875
- EXT_SOURCE_1: 3.924690
- DAYS_BIRTH: 3.192080
- AMT_CREDIT: 2.676793
- AMT_ANNUITY: 2.397871
- AMT_GOODS_PRICE: 2.199153
- BUREAU_DEBT_CREDIT_RATIO_MEAN: 1.837825
- DAYS_EMPLOYED: 1.721038
- POS_CNT_INSTALMENT_FUTURE_MEAN: 1.330537
- DAYS_ID_PUBLISH: 1.214422
- CODE_GENDER: 1.192433
- NAME_EDUCATION_TYPE: 1.089340
- BUREAU_DAYS_CREDIT_MAX: 1.084598
- BUREAU_AMT_CREDIT_SUM_MEAN: 1.083243
- PREV_PREV_CREDIT_GOODS_RATIO_MEAN: 0.962676
- ORGANIZATION_TYPE: 0.953958
- REGION_POPULATION_RELATIVE: 0.950052
- DAYS_REGISTRATION: 0.922683
- INS_LATE_RATIO: 0.870674