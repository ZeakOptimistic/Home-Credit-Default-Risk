import pandas as pd


def build_submission(test_ids, test_preds, id_col="SK_ID_CURR"):
    submission = pd.DataFrame({
        id_col: test_ids,
        "TARGET": test_preds,
    })
    return submission


def build_baseline_report(
    train_shape,
    test_shape,
    positive_rate,
    n_categorical,
    n_numeric,
    n_splits,
    fold_scores,
    overall_auc,
    submission_path,
    fig_path,
):
    report_lines = [
        "# Baseline Results",
        "",
        "## Dataset",
        f"- Train shape: {train_shape}",
        f"- Test shape: {test_shape}",
        f"- Positive rate: {positive_rate:.4f}",
        f"- Number of categorical columns: {n_categorical}",
        f"- Number of numeric columns: {n_numeric}",
        "",
        "## Cross-validation",
        f"- Number of folds: {n_splits}",
        f"- Fold AUC scores: {[round(score, 6) for score in fold_scores]}",
        f"- Mean fold AUC: {sum(fold_scores)/len(fold_scores):.6f}",
        f"- OOF AUC: {overall_auc:.6f}",
        "",
        "## Outputs",
        f"- Submission file: {submission_path}",
        f"- Feature importance figure: {fig_path}",
        "",
        "## Notes",
        "- This baseline uses only the main application table.",
        "- This benchmark will be compared against later feature-engineered models.",
    ]
    return "\n".join(report_lines)