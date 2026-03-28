# Data Audit Summary

## Objective
Understand the structure of the Home Credit Default Risk dataset before EDA and modeling.

## Main findings
- `application_train` and `application_test` are the core modeling tables.
- The dataset is relational and contains multiple historical tables.
- `bureau_balance` must be linked through `bureau` using `SK_ID_BUREAU`.
- `installments_payments`, `POS_CASH_balance`, and `credit_card_balance` are history tables connected to previous applications.
- Most auxiliary tables require aggregation before merging into the main table.

## Key identifiers
- `SK_ID_CURR`: customer/application identifier
- `SK_ID_PREV`: previous Home Credit application identifier
- `SK_ID_BUREAU`: bureau credit identifier

## Planned next step
Perform EDA on `application_train` and build a baseline model using only the main table.