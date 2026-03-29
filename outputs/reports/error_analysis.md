# Error Analysis

## Model quality
- OOF AUC: 0.785274
- Threshold used for error analysis: 0.464240

## Error group counts
- True Negative: 200828
- False Positive: 81858
- False Negative: 6938
- True Positive: 17887

## Top 10 most important features
1. EXT_SOURCE_3 (8.769960)
2. EXT_SOURCE_2 (7.133875)
3. EXT_SOURCE_1 (3.924690)
4. DAYS_BIRTH (3.192080)
5. AMT_CREDIT (2.676793)
6. AMT_ANNUITY (2.397871)
7. AMT_GOODS_PRICE (2.199153)
8. BUREAU_DEBT_CREDIT_RATIO_MEAN (1.837825)
9. DAYS_EMPLOYED (1.721038)
10. POS_CNT_INSTALMENT_FUTURE_MEAN (1.330537)

## Feature family contribution in top 50
- main_application: count=21, total_importance=43.046229, mean_importance=2.049820
- payment_history: count=15, total_importance=10.007692, mean_importance=0.667179
- bureau: count=7, total_importance=6.512974, mean_importance=0.930425
- previous_application: count=7, total_importance=4.417790, mean_importance=0.631113

## Files created
- C:\Coding\Home-Credit-Default-Risk\outputs\figures\error_analysis\01_error_group_counts.png
- C:\Coding\Home-Credit-Default-Risk\outputs\figures\error_analysis\02_oof_prediction_distribution.png
- C:\Coding\Home-Credit-Default-Risk\outputs\figures\error_analysis\03_top_feature_importance.png
- C:\Coding\Home-Credit-Default-Risk\outputs\figures\error_analysis\04_feature_family_contribution.png
- C:\Coding\Home-Credit-Default-Risk\outputs\reports\error_group_feature_means.csv
- C:\Coding\Home-Credit-Default-Risk\outputs\reports\hardest_false_negatives.csv
- C:\Coding\Home-Credit-Default-Risk\outputs\reports\hardest_false_positives.csv