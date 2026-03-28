# Main Table EDA Summary

## Dataset
- Train shape: (307511, 131)
- Test shape: (48744, 130)

## Target
- TARGET=0 count: 282,686
- TARGET=1 count: 24,825
- TARGET=1 proportion: 0.0807

## Missing Values
- Top 10 columns by missing percentage:
  - COMMONAREA_MEDI: 69.87%
  - COMMONAREA_AVG: 69.87%
  - COMMONAREA_MODE: 69.87%
  - NONLIVINGAPARTMENTS_MODE: 69.43%
  - NONLIVINGAPARTMENTS_AVG: 69.43%
  - NONLIVINGAPARTMENTS_MEDI: 69.43%
  - FONDKAPREMONT_MODE: 68.39%
  - LIVINGAPARTMENTS_MODE: 68.35%
  - LIVINGAPARTMENTS_AVG: 68.35%
  - LIVINGAPARTMENTS_MEDI: 68.35%

## Key Findings
- The target is imbalanced, so accuracy is not the right metric.
- Several columns have substantial missingness and should be handled carefully.
- Core financial variables are strongly skewed and contain outliers.
- DAYS_EMPLOYED contains the anomaly value 365243 and should be treated separately.
- Ratio features look more meaningful than some raw variables.

## Candidate Features
- AMT_INCOME_TOTAL
- AMT_CREDIT
- AMT_ANNUITY
- AGE_YEARS
- DAYS_EMPLOYED_CLEAN
- DAYS_EMPLOYED_ANOM
- INCOME_CREDIT_RATIO
- ANNUITY_INCOME_RATIO
- CREDIT_GOODS_RATIO
- EMPLOYMENT_AGE_RATIO