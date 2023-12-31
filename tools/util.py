import os
import pandas as pd

from pathlib import Path


def save_file(data, root, title, subtitle, file_name):
    dir_path = os.path.join(root, title, subtitle)
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(dir_path, file_name)
    data.to_csv(file_path, index=True)


def load_file(root, title, subtitle, file_name):
    dir_path = os.path.join(root, title, subtitle)
    file_path = os.path.join(dir_path, file_name)

    if Path(file_path).exists():
        return pd.read_csv(file_path, index_col=["name", "iter_num"]) \
                 .rename(columns=lambda x: float(x))
    else:
        return None
