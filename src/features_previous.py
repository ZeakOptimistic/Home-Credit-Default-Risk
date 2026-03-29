import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_previous_application_data():
    previous = pd.read_csv(RAW_DATA_DIR / "previous_application.csv")
    train_ids = pd.read_csv(RAW_DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    test_ids = pd.read_csv(RAW_DATA_DIR / "application_test.csv", usecols=["SK_ID_CURR"])

    base = pd.concat([train_ids, test_ids], axis=0, ignore_index=True)
    base = base.drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)

    return base, previous


def preprocess_previous_application(previous: pd.DataFrame) -> pd.DataFrame:
    prev = previous.copy()

    # Common anomalous placeholder values in this dataset
    day_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for col in day_cols:
        if col in prev.columns:
            prev[col] = prev[col].replace(365243, np.nan)

    # Simple decision flags
    prev["NAME_CONTRACT_STATUS"] = prev["NAME_CONTRACT_STATUS"].fillna("MISSING")
    prev["PREV_IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["PREV_IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["PREV_IS_CANCELED"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
    prev["PREV_IS_UNUSED_OFFER"] = (prev["NAME_CONTRACT_STATUS"] == "Unused offer").astype(int)

    # Amount-based engineered features
    prev["PREV_APP_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["PREV_APP_CREDIT_RATIO"] = (
        prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    ).replace([np.inf, -np.inf], np.nan)

    prev["PREV_CREDIT_GOODS_RATIO"] = (
        prev["AMT_CREDIT"] / prev["AMT_GOODS_PRICE"]
    ).replace([np.inf, -np.inf], np.nan)

    prev["PREV_ANNUITY_CREDIT_RATIO"] = (
        prev["AMT_ANNUITY"] / prev["AMT_CREDIT"]
    ).replace([np.inf, -np.inf], np.nan)

    prev["PREV_DOWNPAYMENT_CREDIT_RATIO"] = (
        prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"]
    ).replace([np.inf, -np.inf], np.nan)

    # Timing / recency
    prev["PREV_DAYS_DECISION_ABS"] = prev["DAYS_DECISION"].abs()

    # Approved-only useful features
    prev["PREV_APPROVED_AMT_CREDIT"] = np.where(
        prev["PREV_IS_APPROVED"] == 1, prev["AMT_CREDIT"], np.nan
    )
    prev["PREV_APPROVED_AMT_ANNUITY"] = np.where(
        prev["PREV_IS_APPROVED"] == 1, prev["AMT_ANNUITY"], np.nan
    )
    prev["PREV_APPROVED_APP_CREDIT_RATIO"] = np.where(
        prev["PREV_IS_APPROVED"] == 1, prev["PREV_APP_CREDIT_RATIO"], np.nan
    )

    return prev


def aggregate_previous_to_curr(previous: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "SK_ID_PREV": "count",
        "PREV_IS_APPROVED": "sum",
        "PREV_IS_REFUSED": "sum",
        "PREV_IS_CANCELED": "sum",
        "PREV_IS_UNUSED_OFFER": "sum",
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_DOWN_PAYMENT": ["mean", "max"],
        "AMT_GOODS_PRICE": ["mean", "max"],
        "CNT_PAYMENT": ["mean", "max"],
        "RATE_DOWN_PAYMENT": ["mean", "max"],
        "DAYS_DECISION": ["mean", "max"],
        "PREV_DAYS_DECISION_ABS": ["mean", "min"],
        "PREV_APP_CREDIT_DIFF": ["mean", "max"],
        "PREV_APP_CREDIT_RATIO": ["mean", "max"],
        "PREV_CREDIT_GOODS_RATIO": "mean",
        "PREV_ANNUITY_CREDIT_RATIO": "mean",
        "PREV_DOWNPAYMENT_CREDIT_RATIO": "mean",
        "PREV_APPROVED_AMT_CREDIT": ["mean", "max"],
        "PREV_APPROVED_AMT_ANNUITY": ["mean", "max"],
        "PREV_APPROVED_APP_CREDIT_RATIO": "mean",
    }

    prev_agg = previous.groupby("SK_ID_CURR").agg(agg_dict)

    prev_agg.columns = [
        "PREV_" + "_".join(col).upper() if isinstance(col, tuple) else "PREV_" + col.upper()
        for col in prev_agg.columns
    ]

    prev_agg = prev_agg.reset_index()

    prev_agg["PREV_APPROVAL_RATE"] = (
        prev_agg["PREV_PREV_IS_APPROVED_SUM"] / prev_agg["PREV_SK_ID_PREV_COUNT"]
    )

    prev_agg["PREV_REFUSAL_RATE"] = (
        prev_agg["PREV_PREV_IS_REFUSED_SUM"] / prev_agg["PREV_SK_ID_PREV_COUNT"]
    )

    prev_agg["PREV_CANCEL_RATE"] = (
        prev_agg["PREV_PREV_IS_CANCELED_SUM"] / prev_agg["PREV_SK_ID_PREV_COUNT"]
    )

    return prev_agg


def build_previous_application_features(save: bool = True) -> pd.DataFrame:
    base, previous = load_previous_application_data()
    previous = preprocess_previous_application(previous)
    previous_features = aggregate_previous_to_curr(previous)

    final_features = base.merge(previous_features, on="SK_ID_CURR", how="left")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_features.to_parquet(
            PROCESSED_DATA_DIR / "previous_application_features.parquet",
            index=False,
        )

    return final_features