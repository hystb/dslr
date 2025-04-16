import numpy as np
from src.preprocessing.file.dataset import DataSet
from src.preprocessing.file.loader import Loader
import src.maths as mts
import sys

def truncate_string(s):
    max_lenght = 10

    if len(s) > max_lenght:
        return s[:max_lenght - 2] + ".."
    return s

def remove_none(column: np.ndarray):
    return column[column != None] 

def do_describe(dataset: DataSet):
    headers = list(dataset.get_headers().keys())

    numerics_headers = headers[dataset.numeric_data_index:]
    numerics_columns = dataset.get_numerics().T

    values = {
        "count": [],
        "mean": [],
        "std": [],
        "min": [],
        "p25": [],
        "p50": [],
        "p75": [],
        "max": [],
        "samp_var": [],
        "iqr": [],
        "skew": []
    }

    for col in numerics_columns:
        col = remove_none(col)
        values.get("count").append(mts.count(col))
        values.get("mean").append(mts.mean(col))
        values.get("std").append(mts.std(col))
        values.get("min").append(mts.min(col))
        values.get("p25").append(mts.percentile(col, 25))
        values.get("p50").append(mts.percentile(col, 50))
        values.get("p75").append(mts.percentile(col, 75))
        values.get("max").append(mts.max(col))
        values.get("samp_var").append(mts.sample_var(col))
        values.get("iqr").append(mts.percentile(col, 75) - mts.percentile(col, 25))
        values.get("skew").append(mts.skew(col))

    print(f"{'':<10}", end="")
    for h in numerics_headers:
        print(f"{truncate_string(h):<12}", end="")
    print()

    for stat, vals in values.items():
        print(f"{stat:<10}", end="")
        for v in vals:
            print(f"{v:<12.3g}", end="")
        print()

def main():
    if (len(sys.argv) == 2):
        dataset = Loader().load(sys.argv[1])
        do_describe(dataset)

if __name__ == "__main__":
    main()
