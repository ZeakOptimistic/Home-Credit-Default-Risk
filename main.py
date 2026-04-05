import gc
import pandas as pd
from pathlib import Path

from src.data_loader import load_application_data
from src.features import (
    build_bureau_features,
    build_previous_application_features,
    build_installments_features,
    build_pos_cash_features,
    build_credit_card_features,
)
from src.preprocessing import prepare_main_table
from src.train import train_lightgbm_cv
from src.inference import build_submission
from src.config import SUBMISSIONS_DIR

def run_pipeline():
    print("1/7 Loading base application data...")
    train, test, sample_sub, data_dir = load_application_data()
    
    print("2/7 Building Bureau features...")
    bureau_feats = build_bureau_features(save=False)
    
    print("3/7 Building Previous Application features...")
    prev_feats = build_previous_application_features(save=False)

    print("4/7 Building Fast Memory-Safe Payment features...")
    inst_feats = build_installments_features(save=False)
    pos_feats = build_pos_cash_features(save=False)
    cc_feats = build_credit_card_features(save=False)
    
    print("5/7 Merging features into main training table...")
    additional_features = [bureau_feats, prev_feats, inst_feats, pos_feats, cc_feats]
    X, y, X_test, test_ids, categorical_cols, numeric_cols = prepare_main_table(train, test, additional_features)
    
    del train, test, bureau_feats, prev_feats, inst_feats, pos_feats, cc_feats
    gc.collect()
    
    print(f"6/7 Training LightGBM on {X.shape[0]} rows and {X.shape[1]} features...")
    results = train_lightgbm_cv(X, y, X_test, categorical_cols)
    print(f"---> Cross-Validated OOF AUC: {results['overall_auc']:.6f} <---")
    
    print("7/7 Generating Kaggle submission...")
    submission = build_submission(test_ids, results["test_preds"])
    sub_path = SUBMISSIONS_DIR / "final_submission_lgbm.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Success! Submission file ready for Kaggle upload at: {sub_path}")

if __name__ == "__main__":
    run_pipeline()
