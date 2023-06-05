from copy import deepcopy
import pandas as pd
import glob

files = glob.glob("ci-tests/*.parquet")
files = sorted(files)

agg_df = pd.DataFrame(columns=["dataset", "llp_variant", "follow_dgm", "b-indep-y", "x-indep-b", "x-indep-y-given-b", "x-indep-b-given-y", "b-indep-y-given-x"])
for file in files:
    df = pd.read_parquet(file)
    dataset, ci_test = file.split("/")[1].split(".")[0].split("_")
    print("Dataset: {}\n CI test: {}".format(dataset, ci_test))
    # If dataset is not in agg_df, add it
    if dataset not in agg_df["dataset"].values:
        agg_df = pd.concat([agg_df, df], ignore_index=True)
    else:
        # Update the row
        row = deepcopy(agg_df[agg_df["dataset"] == dataset])
        row[ci_test] = df[ci_test].values[0]

        # Update the follow dgm
        row["follow_dgm"] = row["follow_dgm"].values[0] and df["follow_dgm"].values[0]
        # Update agg_df
        agg_df[agg_df["dataset"] == dataset] = row

agg_df.to_csv("ci-tests/ci-results.csv", index=False)