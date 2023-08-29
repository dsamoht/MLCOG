"""parse .pkl files in a given directory;
   throws the `;` separated matrix to stdout
"""
import glob
from pathlib import Path
import pickle
import sys

import pandas as pd

sys.path.append("../mlcog")


res_dir = sys.argv[1]

if __name__ == "__main__":

    for task in ["test"]:
        lines = []
        result_files = glob.glob(str(Path(res_dir).joinpath(f"task-{task}*.pkl")))
        for _file in result_files:
            res = {}
            res["datatype"] = _file.split("datatype-")[1].split("_algo")[0]
            res["algo"] = _file.split("algo-")[1].split("_selection")[0]
            res["fs_method"] = _file.split("selection-")[1].split(".pkl")[0]
            d = pickle.load(open(_file, "rb"))
            res.update(dict(d["results_matrix"].iloc[d["best_index"],:]))
            lines.append(res)
        res_df = pd.DataFrame(lines)

    res_df.to_csv(sys.stdout, sep=";")
