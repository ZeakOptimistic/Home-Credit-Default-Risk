import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_bureau_data():
    bureau = pd.read_csv(RAW_DATA_DIR / "bureau.csv")
    bureau_balance = pd.read_csv(RAW_DATA_DIR / "bureau_balance.csv")
    train_ids = pd.read_csv(RAW_DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    test_ids = pd.read_csv(RAW_DATA_DIR / "application_test.csv", usecols=["SK_ID_CURR"])

    base = pd.concat([train_ids, test_ids], axis=0, ignore_index=True)
    base = base.drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)

    return base, bureau, bureau_balance


def build_bureau_balance_features(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb = bureau_balance.copy()

    status_dummies = pd.get_dummies(bb["STATUS"], prefix="BB_STATUS")
    bb = pd.concat([bb, status_dummies], axis=1)

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_MONTHS_MAX=("MONTHS_BALANCE", "max"),
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
    )

    status_cols = [c for c in bb.columns if c.startswith("BB_STATUS_")]

    bb_status_agg = bb.groupby("SK_ID_BUREAU")[status_cols].agg(["mean", "sum"])
    bb_status_agg.columns = [
        f"{col}_{stat}".upper() for col, stat in bb_status_agg.columns
    ]
    bb_status_agg = bb_status_agg.reset_index()

    bb_agg = bb_agg.reset_index().merge(bb_status_agg, on="SK_ID_BUREAU", how="left")
    return bb_agg


def merge_bureau_with_balance(bureau: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    bureau_full = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bureau_full["CREDIT_ACTIVE_IS_ACTIVE"] = (
        bureau_full["CREDIT_ACTIVE"] == "Active"
    ).astype(int)

    bureau_full["CREDIT_ACTIVE_IS_CLOSED"] = (
        bureau_full["CREDIT_ACTIVE"] == "Closed"
    ).astype(int)

    bureau_full["HAS_DEBT"] = (
        bureau_full["AMT_CREDIT_SUM_DEBT"].fillna(0) > 0
    ).astype(int)

    bureau_full["HAS_OVERDUE"] = (
        bureau_full["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0
    ).astype(int)

    bureau_full["DEBT_CREDIT_RATIO"] = (
        bureau_full["AMT_CREDIT_SUM_DEBT"] / bureau_full["AMT_CREDIT_SUM"]
    ).replace([np.inf, -np.inf], np.nan)

    bureau_full["OVERDUE_DEBT_RATIO"] = (
        bureau_full["AMT_CREDIT_SUM_OVERDUE"] / bureau_full["AMT_CREDIT_SUM_DEBT"]
    ).replace([np.inf, -np.inf], np.nan)

    return bureau_full


def aggregate_bureau_to_curr(bureau_full: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "SK_ID_BUREAU": "count",
        "CREDIT_ACTIVE_IS_ACTIVE": "sum",
        "CREDIT_ACTIVE_IS_CLOSED": "sum",
        "HAS_DEBT": "sum",
        "HAS_OVERDUE": "sum",
        "DAYS_CREDIT": ["mean", "max"],
        "DAYS_CREDIT_UPDATE": "mean",
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": "max",
        "DEBT_CREDIT_RATIO": "mean",
        "OVERDUE_DEBT_RATIO": "mean",
        "BB_MONTHS_COUNT": ["mean", "max"],
    }

    bb_status_mean_cols = [
        c for c in bureau_full.columns
        if c.startswith("BB_STATUS_") and c.endswith("_MEAN")
    ]

    bb_status_sum_cols = [
        c for c in bureau_full.columns
        if c.startswith("BB_STATUS_") and c.endswith("_SUM")
    ]

    for col in bb_status_mean_cols:
        agg_dict[col] = "mean"

    for col in bb_status_sum_cols:
        agg_dict[col] = "sum"

    bureau_curr_agg = bureau_full.groupby("SK_ID_CURR").agg(agg_dict)

    bureau_curr_agg.columns = [
        "BUREAU_" + "_".join(col).upper() if isinstance(col, tuple) else "BUREAU_" + col.upper()
        for col in bureau_curr_agg.columns
    ]

    bureau_curr_agg = bureau_curr_agg.reset_index()

    bureau_curr_agg["BUREAU_ACTIVE_RATIO"] = (
        bureau_curr_agg["BUREAU_CREDIT_ACTIVE_IS_ACTIVE_SUM"] /
        bureau_curr_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    )

    bureau_curr_agg["BUREAU_CLOSED_RATIO"] = (
        bureau_curr_agg["BUREAU_CREDIT_ACTIVE_IS_CLOSED_SUM"] /
        bureau_curr_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    )

    bureau_curr_agg["BUREAU_OVERDUE_RATIO"] = (
        bureau_curr_agg["BUREAU_HAS_OVERDUE_SUM"] /
        bureau_curr_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    )

    return bureau_curr_agg


def build_bureau_features(save: bool = True) -> pd.DataFrame:
    base, bureau, bureau_balance = load_bureau_data()
    bb_agg = build_bureau_balance_features(bureau_balance)
    bureau_full = merge_bureau_with_balance(bureau, bb_agg)
    bureau_curr_agg = aggregate_bureau_to_curr(bureau_full)

    bureau_features = base.merge(bureau_curr_agg, on="SK_ID_CURR", how="left")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        bureau_features.to_parquet(PROCESSED_DATA_DIR / "bureau_features.parquet", index=False)

    return bureau_features