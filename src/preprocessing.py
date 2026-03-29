import numpy as np

from src.config import ID_COL, TARGET_COL


def handle_days_employed_anomaly(df):
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df


def prepare_main_table_baseline(train, test):
    train = handle_days_employed_anomaly(train)
    test = handle_days_employed_anomaly(test)

    X = train.drop(columns=[TARGET_COL, ID_COL]).copy()
    y = train[TARGET_COL].copy()

    X_test = test.drop(columns=[ID_COL]).copy()
    test_ids = test[ID_COL].copy()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    for df in [X, X_test]:
        for col in categorical_cols:
            df[col] = df[col].fillna("MISSING").astype(str)

    return X, y, X_test, test_ids, categorical_cols, numeric_cols

def prepare_main_table_with_bureau(train, test, bureau_features):
    train = handle_days_employed_anomaly(train)
    test = handle_days_employed_anomaly(test)

    if not bureau_features[ID_COL].is_unique:
        raise ValueError("bureau_features must be unique at SK_ID_CURR level")

    train_merged = train.merge(bureau_features, on=ID_COL, how="left")
    test_merged = test.merge(bureau_features, on=ID_COL, how="left")

    X = train_merged.drop(columns=[TARGET_COL, ID_COL]).copy()
    y = train_merged[TARGET_COL].copy()

    X_test = test_merged.drop(columns=[ID_COL]).copy()
    test_ids = test_merged[ID_COL].copy()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    for df in [X, X_test]:
        for col in categorical_cols:
            df[col] = df[col].fillna("MISSING").astype(str)

    return X, y, X_test, test_ids, categorical_cols, numeric_cols