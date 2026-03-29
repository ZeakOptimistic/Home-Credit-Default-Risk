from pathlib import Path
import pandas as pd

from src.config import PROJECT_ROOT, RAW_DATA_DIR, SAMPLE_DATA_DIR, ID_COL


def find_data_dir():
    candidates = [
        RAW_DATA_DIR,
        SAMPLE_DATA_DIR,
        Path("/mnt/data"),
        PROJECT_ROOT,
    ]

    for path in candidates:
        if (path / "application_train.csv").exists() and (path / "application_test.csv").exists():
            return path

    raise FileNotFoundError("Could not find application_train.csv and application_test.csv")


def load_application_data():
    data_dir = find_data_dir()

    train_path = data_dir / "application_train.csv"
    test_path = data_dir / "application_test.csv"
    sample_sub_path = data_dir / "sample_submission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
    else:
        sample_sub = pd.DataFrame({ID_COL: test[ID_COL], "TARGET": 0.0})

    return train, test, sample_sub, data_dir