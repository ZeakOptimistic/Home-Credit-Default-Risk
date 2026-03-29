# Final Project Summary

## Best Model
- Best model: Main + Bureau + Previous + Payments
- Best OOF AUC: 0.785274
- Final submission: C:\Coding\Home-Credit-Default-Risk\outputs\submissions\final_submission.csv
- Final model artifact: C:\Coding\Home-Credit-Default-Risk\outputs\models\final_model.joblib

## Model Comparison
| model_name | mean_fold_auc | oof_auc | lift_vs_previous | lift_vs_baseline |
| --- | --- | --- | --- | --- |
| Baseline (Main Table) | 0.759922 | 0.759895 | N/A | 0.000000 |
| Main + Bureau | 0.765376 | 0.765354 | 0.005459 | 0.005459 |
| Main + Bureau + Previous | 0.772989 | 0.772974 | 0.007620 | 0.013079 |
| Main + Bureau + Previous + Payments | 0.785389 | 0.785274 | 0.012300 | 0.025379 |

## Generated Artifacts
- Comparison CSV: C:\Coding\Home-Credit-Default-Risk\outputs\reports\final_model_comparison.csv
- Comparison figure: C:\Coding\Home-Credit-Default-Risk\outputs\figures\final_report\01_model_comparison.png
- Pipeline figure: C:\Coding\Home-Credit-Default-Risk\outputs\figures\final_report\02_project_pipeline_overview.png
- README: c:\Coding\Home-Credit-Default-Risk\README.md