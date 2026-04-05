import pandas as pd
import numpy as np
import gc
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ----------------- BUREAU FEATURES -----------------

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
    bb_status_agg.columns = [f"{col}_{stat}".upper() for col, stat in bb_status_agg.columns]
    bb_status_agg = bb_status_agg.reset_index()

    bb_agg = bb_agg.reset_index().merge(bb_status_agg, on="SK_ID_BUREAU", how="left")
    return bb_agg

def merge_bureau_with_balance(bureau: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    bureau_full = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau_full["CREDIT_ACTIVE_IS_ACTIVE"] = (bureau_full["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau_full["CREDIT_ACTIVE_IS_CLOSED"] = (bureau_full["CREDIT_ACTIVE"] == "Closed").astype(int)
    bureau_full["HAS_DEBT"] = (bureau_full["AMT_CREDIT_SUM_DEBT"].fillna(0) > 0).astype(int)
    bureau_full["HAS_OVERDUE"] = (bureau_full["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0).astype(int)
    bureau_full["DEBT_CREDIT_RATIO"] = (bureau_full["AMT_CREDIT_SUM_DEBT"] / bureau_full["AMT_CREDIT_SUM"]).replace([np.inf, -np.inf], np.nan)
    bureau_full["OVERDUE_DEBT_RATIO"] = (bureau_full["AMT_CREDIT_SUM_OVERDUE"] / bureau_full["AMT_CREDIT_SUM_DEBT"]).replace([np.inf, -np.inf], np.nan)
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

    for col in bureau_full.columns:
        if col.startswith("BB_STATUS_") and col.endswith("_MEAN"): agg_dict[col] = "mean"
        elif col.startswith("BB_STATUS_") and col.endswith("_SUM"): agg_dict[col] = "sum"

    bureau_curr_agg = bureau_full.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_curr_agg.columns = ["BUREAU_" + "_".join(col).upper() if isinstance(col, tuple) else "BUREAU_" + col.upper() for col in bureau_curr_agg.columns]
    bureau_curr_agg = bureau_curr_agg.reset_index()

    bureau_curr_agg["BUREAU_ACTIVE_RATIO"] = bureau_curr_agg["BUREAU_CREDIT_ACTIVE_IS_ACTIVE_SUM"] / bureau_curr_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    bureau_curr_agg["BUREAU_CLOSED_RATIO"] = bureau_curr_agg["BUREAU_CREDIT_ACTIVE_IS_CLOSED_SUM"] / bureau_curr_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    bureau_curr_agg["BUREAU_OVERDUE_RATIO"] = bureau_curr_agg["BUREAU_HAS_OVERDUE_SUM"] / bureau_curr_agg["BUREAU_SK_ID_BUREAU_COUNT"]
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


# ----------------- PREVIOUS APPLICATION FEATURES -----------------

def load_previous_application_data():
    previous = pd.read_csv(RAW_DATA_DIR / "previous_application.csv")
    train_ids = pd.read_csv(RAW_DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    test_ids = pd.read_csv(RAW_DATA_DIR / "application_test.csv", usecols=["SK_ID_CURR"])

    base = pd.concat([train_ids, test_ids], axis=0, ignore_index=True)
    base = base.drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)
    return base, previous

def preprocess_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    day_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    for col in day_cols:
        if col in prev.columns: prev[col] = prev[col].replace(365243, np.nan)

    prev["NAME_CONTRACT_STATUS"] = prev["NAME_CONTRACT_STATUS"].fillna("MISSING")
    prev["PREV_IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["PREV_IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["PREV_IS_CANCELED"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
    prev["PREV_IS_UNUSED_OFFER"] = (prev["NAME_CONTRACT_STATUS"] == "Unused offer").astype(int)

    prev["PREV_APP_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["PREV_APP_CREDIT_RATIO"] = (prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]).replace([np.inf, -np.inf], np.nan)
    prev["PREV_CREDIT_GOODS_RATIO"] = (prev["AMT_CREDIT"] / prev["AMT_GOODS_PRICE"]).replace([np.inf, -np.inf], np.nan)
    prev["PREV_ANNUITY_CREDIT_RATIO"] = (prev["AMT_ANNUITY"] / prev["AMT_CREDIT"]).replace([np.inf, -np.inf], np.nan)
    prev["PREV_DOWNPAYMENT_CREDIT_RATIO"] = (prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"]).replace([np.inf, -np.inf], np.nan)

    prev["PREV_DAYS_DECISION_ABS"] = prev["DAYS_DECISION"].abs()

    prev["PREV_APPROVED_AMT_CREDIT"] = np.where(prev["PREV_IS_APPROVED"] == 1, prev["AMT_CREDIT"], np.nan)
    prev["PREV_APPROVED_AMT_ANNUITY"] = np.where(prev["PREV_IS_APPROVED"] == 1, prev["AMT_ANNUITY"], np.nan)
    prev["PREV_APPROVED_APP_CREDIT_RATIO"] = np.where(prev["PREV_IS_APPROVED"] == 1, prev["PREV_APP_CREDIT_RATIO"], np.nan)

    return prev

def aggregate_previous_to_curr(previous: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "SK_ID_PREV": "count", "PREV_IS_APPROVED": "sum", "PREV_IS_REFUSED": "sum", "PREV_IS_CANCELED": "sum", "PREV_IS_UNUSED_OFFER": "sum",
        "AMT_APPLICATION": ["mean", "max"], "AMT_CREDIT": ["mean", "max"], "AMT_ANNUITY": ["mean", "max"], "AMT_DOWN_PAYMENT": ["mean", "max"], "AMT_GOODS_PRICE": ["mean", "max"],
        "CNT_PAYMENT": ["mean", "max"], "RATE_DOWN_PAYMENT": ["mean", "max"], "DAYS_DECISION": ["mean", "max"], "PREV_DAYS_DECISION_ABS": ["mean", "min"],
        "PREV_APP_CREDIT_DIFF": ["mean", "max"], "PREV_APP_CREDIT_RATIO": ["mean", "max"], "PREV_CREDIT_GOODS_RATIO": "mean", "PREV_ANNUITY_CREDIT_RATIO": "mean", "PREV_DOWNPAYMENT_CREDIT_RATIO": "mean",
        "PREV_APPROVED_AMT_CREDIT": ["mean", "max"], "PREV_APPROVED_AMT_ANNUITY": ["mean", "max"], "PREV_APPROVED_APP_CREDIT_RATIO": "mean",
    }
    prev_agg = previous.groupby("SK_ID_CURR").agg(agg_dict)
    prev_agg.columns = ["PREV_" + "_".join(col).upper() if isinstance(col, tuple) else "PREV_" + col.upper() for col in prev_agg.columns]
    prev_agg = prev_agg.reset_index()
    prev_agg["PREV_APPROVAL_RATE"] = prev_agg["PREV_PREV_IS_APPROVED_SUM"] / prev_agg["PREV_SK_ID_PREV_COUNT"]
    prev_agg["PREV_REFUSAL_RATE"] = prev_agg["PREV_PREV_IS_REFUSED_SUM"] / prev_agg["PREV_SK_ID_PREV_COUNT"]
    prev_agg["PREV_CANCEL_RATE"] = prev_agg["PREV_PREV_IS_CANCELED_SUM"] / prev_agg["PREV_SK_ID_PREV_COUNT"]
    return prev_agg

def build_previous_application_features(save: bool = True) -> pd.DataFrame:
    base, previous = load_previous_application_data()
    previous = preprocess_previous_application(previous)
    previous_features = aggregate_previous_to_curr(previous)
    final_features = base.merge(previous_features, on="SK_ID_CURR", how="left")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_features.to_parquet(PROCESSED_DATA_DIR / "previous_application_features.parquet", index=False)
    return final_features

# ----------------- PAYMENTS FEATURES (MEMORY EFFICIENT) -----------------

def build_installments_features(save: bool = True) -> pd.DataFrame:
    base = pd.read_csv(RAW_DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    base = pd.concat([base, pd.read_csv(RAW_DATA_DIR / "application_test.csv", usecols=["SK_ID_CURR"])], ignore_index=True)
    base = base.drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)
    
    inst = pd.read_csv(RAW_DATA_DIR / "installments_payments.csv")
    inst["PAYMENT_PERC"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["DPD"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["DPD"] = inst["DPD"].apply(lambda x: x if x > 0 else 0)
    inst["DBD"] = inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]
    inst["DBD"] = inst["DBD"].apply(lambda x: x if x > 0 else 0)
    
    agg_dict = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"]
    }
    
    inst_agg = inst.groupby("SK_ID_CURR").agg(agg_dict)
    inst_agg.columns = ["INSTAL_" + "_".join(col).upper() for col in inst_agg.columns]
    inst_agg.reset_index(inplace=True)
    del inst
    gc.collect()
    
    final_features = base.merge(inst_agg, on="SK_ID_CURR", how="left")
    for col in final_features.columns:
        if final_features[col].dtype == "float64": final_features[col] = final_features[col].astype("float32")
            
    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_features.to_parquet(PROCESSED_DATA_DIR / "installments_features.parquet", index=False)
    return final_features

def build_pos_cash_features(save: bool = True) -> pd.DataFrame:
    base = pd.read_csv(RAW_DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    base = pd.concat([base, pd.read_csv(RAW_DATA_DIR / "application_test.csv", usecols=["SK_ID_CURR"])], ignore_index=True)
    base = base.drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)
    
    pos = pd.read_csv(RAW_DATA_DIR / "POS_CASH_balance.csv")
    pos_agg = pos.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"]
    })
    pos_agg.columns = ["POS_" + "_".join(col).upper() for col in pos_agg.columns]
    pos_agg.reset_index(inplace=True)
    del pos
    gc.collect()
    
    final_features = base.merge(pos_agg, on="SK_ID_CURR", how="left")
    for col in final_features.columns:
        if final_features[col].dtype == "float64": final_features[col] = final_features[col].astype("float32")
            
    if save:
        final_features.to_parquet(PROCESSED_DATA_DIR / "pos_cash_features.parquet", index=False)
    return final_features

def build_credit_card_features(save: bool = True) -> pd.DataFrame:
    base = pd.read_csv(RAW_DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    base = pd.concat([base, pd.read_csv(RAW_DATA_DIR / "application_test.csv", usecols=["SK_ID_CURR"])], ignore_index=True)
    base = base.drop_duplicates(subset=["SK_ID_CURR"]).reset_index(drop=True)
    
    cc = pd.read_csv(RAW_DATA_DIR / "credit_card_balance.csv")
    # Exclude non-numeric for easy aggregation
    cc_numeric = cc.drop(columns=["SK_ID_PREV", "NAME_CONTRACT_STATUS"], errors="ignore")
    cc_agg = cc_numeric.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = ["CC_" + "_".join(col).upper() for col in cc_agg.columns]
    cc_agg.reset_index(inplace=True)
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size().values
    del cc
    del cc_numeric
    gc.collect()
    
    final_features = base.merge(cc_agg, on="SK_ID_CURR", how="left")
    for col in final_features.columns:
        if final_features[col].dtype == "float64": final_features[col] = final_features[col].astype("float32")
            
    if save:
        final_features.to_parquet(PROCESSED_DATA_DIR / "credit_card_features.parquet", index=False)
    return final_features
