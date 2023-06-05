import pandas as pd
import glob

base_folder = "datasets-experiments-results/"
files = glob.glob(base_folder + "*")

files.sort()
data = pd.DataFrame(columns=["dataset", "model", "loss", "split_method", "n_splits", "validation_size_perc", "exec"])

for file in files:
    split = file.split("/")[-1].split("_")
    df = pd.read_parquet(file)
    df.insert(len(df.columns), "dataset", [split[0]] * len(df.index))
    df.insert(len(df.columns), "model", [split[1]] * len(df.index))
    df.insert(len(df.columns), "loss", [split[2] + "_" + split[3]] * len(df.index))
    df.insert(len(df.columns), "split_method", [split[4]] * len(df.index))
    df.insert(len(df.columns), "n_splits", [split[5]] * len(df.index))
    df.insert(len(df.columns), "validation_size_perc", [eval(split[6])] * len(df.index))
    df.insert(len(df.columns), "exec", [split[7].split(".")[0]] * len(df.index))

    data = pd.concat([data, df])

data.reset_index(inplace=True, drop=True)
data.to_parquet("datasets-benchmark-experiment-results.parquet", index=False)
