import numpy as np
import pandas as pd
import lightgbm as lgb

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def train_catboost_cv(
    X,
    y,
    X_test,
    categorical_cols,
    random_state=42,
    n_splits_full=5,
):
    positive_count = int(y.sum())
    n_splits = 3 if positive_count < 10 else n_splits_full

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    feature_importance_list = []
    models = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_cols,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=100,
        )

        valid_pred = model.predict_proba(X_valid)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_valid, valid_pred)

        oof_preds[valid_idx] = valid_pred
        test_preds += test_pred / n_splits
        fold_scores.append(fold_auc)
        models.append(model)

        fold_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.get_feature_importance(),
            "fold": fold,
        })
        feature_importance_list.append(fold_importance)

        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_preds)

    feature_importance = (
        pd.concat(feature_importance_list, axis=0)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    return {
        "models": models,
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "fold_scores": fold_scores,
        "overall_auc": overall_auc,
        "n_splits": n_splits,
        "feature_importance": feature_importance,
    }

def train_lightgbm_cv(
    X,
    y,
    X_test,
    categorical_cols,
    random_state=42,
    n_splits_full=5,
):
    positive_count = int(y.sum())
    n_splits = 3 if positive_count < 10 else n_splits_full

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    feature_importance_list = []
    models = []

    for df in [X, X_test]:
        for col in categorical_cols:
            df[col] = df[col].astype("category")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        valid_pred = model.predict_proba(X_valid)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
        fold_auc = roc_auc_score(y_valid, valid_pred)

        oof_preds[valid_idx] = valid_pred
        test_preds += test_pred / n_splits
        fold_scores.append(fold_auc)
        models.append(model)

        fold_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
            "fold": fold,
        })
        feature_importance_list.append(fold_importance)
        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_preds)
    feature_importance = (
        pd.concat(feature_importance_list, axis=0)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    return {
        "models": models,
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "fold_scores": fold_scores,
        "overall_auc": overall_auc,
        "n_splits": n_splits,
        "feature_importance": feature_importance,
    }