import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import argparse
from fcit.fcit import fcit

RANDOM_STATE = 74210992
VARIANTS = ["naive", "simple", "intermediate", "hard"]
FOLLOW_DGM = {
    "naive": {
        "b-indep-y": True,
        "x-indep-b": True,
        "x-indep-y-given-b": False,
        "x-indep-b-given-y": True,
        "b-indep-y-given-x": True,
    },
    "simple": {
        "b-indep-y": False,
        "x-indep-b": False,
        "x-indep-y-given-b": False,
        "x-indep-b-given-y": True,
        "b-indep-y-given-x": False,
    },
    "intermediate": {
        "b-indep-y": False,
        "x-indep-b": False,
        "x-indep-y-given-b": False,
        "x-indep-b-given-y": False,
        "b-indep-y-given-x": True,
    },
    "hard": {
        "b-indep-y": False,
        "x-indep-b": False,
        "x-indep-y-given-b": False,
        "x-indep-b-given-y": False,
        "b-indep-y-given-x": False,
    }
}
DATASETS_FOLDER = "datasets-ci"
EXPERIMENTS_FOLDER = "ci-tests"
try:
    N_JOBS = eval(os.getenv('NSLOTS'))
except:
    N_JOBS = -1
print("Using {} cores".format(N_JOBS))

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, required=True)
parser.add_argument("--ci_test", "-ci", type=str, required=True)

args = parser.parse_args()

dataset = args.dataset
ci_test = args.ci_test

df_ci_tests = pd.DataFrame(columns=["dataset", "llp_variant", "follow_dgm", "b-indep-y", "x-indep-b", "x-indep-y-given-b", "x-indep-b-given-y", "b-indep-y-given-x"])

print("Dataset: {}".format(dataset))

# Extracting variant
llp_variant = [variant for variant in VARIANTS if variant in dataset][0]
print("LLP variant: {}".format(llp_variant))
    
# Extracting base dataset
base_dataset = dataset.split(llp_variant)[0]
base_dataset = base_dataset[:-1]

# Reading X, y (base dataset) and bags (dataset)
df = pd.read_parquet("{}/{}.parquet".format(DATASETS_FOLDER, base_dataset))
X = df.drop(["y"], axis=1).values
y = df["y"].values.reshape(-1, 1)

df = pd.read_parquet("{}/{}.parquet".format(DATASETS_FOLDER, dataset))
bags = df["bag"].values

print(f"CI test: {ci_test}")

if ci_test == "b-indep-y":
    crosstab_y_bags = pd.crosstab(y.reshape(-1), bags.reshape(-1), margins=False)
    p_value = chi2_contingency(crosstab_y_bags.values)[1]

elif ci_test == "x-indep-b":
    p_value = fcit.test(X, bags.reshape(-1, 1), num_perm=30, verbose=False, random_state=RANDOM_STATE, n_jobs=N_JOBS)
else:
    # One-hot encoding bags
    bags = pd.get_dummies(bags).values

    if ci_test == "x-indep-y-given-b":
        p_value = fcit.test(X, y, bags, num_perm=30, verbose=False, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif ci_test == "x-indep-b-given-y":
        p_value = fcit.test(X, bags, y, num_perm=30, verbose=False, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    else:
        p_value = fcit.test(bags, y, X, num_perm=30, verbose=False, random_state=RANDOM_STATE, n_jobs=N_JOBS)

# Checking if the DGM is followed
expected_follow_dgm = FOLLOW_DGM[llp_variant][ci_test]
observed_follow_dgm = p_value > 0.05
follow_dgm = (observed_follow_dgm == expected_follow_dgm)

# Saving results
df_ci_tests = pd.concat([df_ci_tests, pd.DataFrame({
    "dataset": [dataset],
    "llp_variant": [llp_variant],
    "follow_dgm": [follow_dgm],
    ci_test: [np.round(p_value, 4)]
})])

df_ci_tests.follow_dgm = df_ci_tests.follow_dgm.astype(bool)

print("P-value: {}".format(p_value))
print("Follow DGM: {}".format(follow_dgm))
print("\n------------------------\n")

df_ci_tests = df_ci_tests.reset_index(drop=True)

filename = "{}/{}_{}.parquet".format(EXPERIMENTS_FOLDER, dataset, ci_test)
df_ci_tests.to_parquet(filename)
print("Execution finished!!!")
        

    



