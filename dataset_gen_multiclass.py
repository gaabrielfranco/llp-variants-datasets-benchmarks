import argparse
import os
from datasets import llp_variant_generation
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from datasets import llp_variant_generation
from datasets_gen_info import CANDIDATES
from sklearn.datasets import fetch_covtype

def compute_proportions_multiclass(bags, y):
    """Compute the proportions (multiclass) for each bag given.
    Parameters
    ----------
    bags : {array-like}
    y : {array-like}

    Returns
    -------
    proportions : {array}
        An array of type np.float
    """
    num_bags = int(max(bags)) + 1
    n_classes = len(np.unique(y))
    proportions = np.empty((num_bags, n_classes), dtype=float)
    for i in range(num_bags):
        bag = np.where(bags == i)[0]
        for j in range(n_classes):
            proportions[i, j] = np.count_nonzero(y[bag] == j) / len(bag)
    return proportions    
    
def load_base_dataset(base_dataset, random_state=None):
    """
    Load the base dataset and return the X and y.

    Parameters
    ----------
    base_dataset : {str}
        The base dataset to load.
    random_state : {int}
        The random state to use.

    Returns
    -------
    X : {array-like}
        The features.
    y : {array-like}
        The labels.
    """
    if base_dataset == "covtype":
        covtype = fetch_covtype(random_state=random_state, shuffle=True)
        X = covtype.data
        y = covtype.target
        y -= 1 # We want the labels to be in [0, 6]
    elif base_dataset == "cifar-10-grey":
        df = pd.read_parquet("base-datasets/cifar-10-grey.parquet")
        df = df.sample(frac=1, random_state=random_state)
        X = deepcopy(df.drop(columns=["label"]).values)
        y = deepcopy(df["label"].values)
    elif base_dataset == "adult":
        df = pd.read_parquet("base-datasets/adult.parquet")
        df.drop(columns=["educational-num"], inplace=True)
        df.gender.replace({"Male": 0, "Female": 1}, inplace=True)
        df.income.replace({"<=50K": 0, ">50K": 1}, inplace=True)
        df = pd.get_dummies(df, columns=
                            ["age", "workclass", "education", "marital-status", \
                             "occupation", "relationship", "race", "native-country"])
        df = df.sample(frac=1, random_state=random_state)
        X = deepcopy(df.drop(columns=["income"]).values)
        y = deepcopy(df["income"].values)
    elif base_dataset == "cifar-10-grey-animal-vehicle":
        df = pd.read_parquet("base-datasets/cifar-10-grey.parquet")
        df = df.sample(frac=1, random_state=random_state)
        df.label = df.label.apply(lambda x: 0 if x in [0, 1, 8, 9] else 1)
        X = deepcopy(df.drop(columns=["label"]).values)
        y = deepcopy(df["label"].values)
    else:
        raise ValueError("Base dataset not found.")
    
    X = MinMaxScaler().fit_transform(X)

    return X, y

def get_clusters(X, clustering_method, n_clusters, random_state):
    if clustering_method == "kmeans":
        clusters = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(X)
    else:
        raise ValueError("Clustering method has to be kmeans.")
    
    # If some clusters are empty, we terminate the program
    for i in range(n_clusters):
        if np.count_nonzero(clusters == i) == 0:
            print("Some clusters are empty. Please try again with a different clustering method.")
            exit()

    return clusters

def generate_llp_dataset(X, y, clusters, proportions_target, bags_sizes_target, llp_variant, random_state):
    if llp_variant == "naive":
        bags = llp_variant_generation(X, llp_variant="naive", bags_size_target=bags_sizes_target, 
                                    random_state=random_state)
    elif llp_variant == "simple":
        bags = llp_variant_generation(X, y, llp_variant="simple", bags_size_target=bags_sizes_target, 
                                    proportions_target=proportions_target, 
                                    random_state=random_state)
    elif llp_variant == "intermediate":
        bags = llp_variant_generation(X, y, llp_variant="intermediate", bags_size_target=bags_sizes_target, 
                                    proportions_target=proportions_target, clusters=clusters, 
                                    random_state=random_state)
    elif llp_variant == "hard":
        bags = llp_variant_generation(X, y, llp_variant="hard", bags_size_target=bags_sizes_target, 
                                    proportions_target=proportions_target, clusters=clusters, 
                                    random_state=random_state)
    return bags

random_state=6738921
random = np.random.RandomState(random_state)
n_jobs=-1
VARIANTS = ["naive", "simple", "intermediate", "hard"]

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_dataset", "-bd", type=str, required=True, choices=["covtype", "cifar-10-grey", "cifar-10-grey-animal-vehicle", "adult"])
parser.add_argument("--clustering_method", "-cm", type=str, required=True, choices=["kmeans"])
parser.add_argument("--n_clusters", "-nc", type=int, required=True)
args = parser.parse_args()

base_dataset = args.base_dataset
clustering_method = args.clustering_method
n_clusters = args.n_clusters

try:
    os.mkdir("datasets-ci-multiclass")
except:
    pass

X, y = load_base_dataset(base_dataset, random_state=random_state)
clusters = get_clusters(X, clustering_method, n_clusters, random_state)

# Setting the proportions and bags sizes target
n_bags = 1000
n_items = X.shape[0]
n_classes = len(np.unique(y))
n_bags_type = "massive"
bags_size_type = "equal"
proportions_type = "close-global"
#(1000, 10)

proportions_global = np.zeros((n_classes), dtype=float)
for j in np.unique(y):
    proportions_global[j] = np.count_nonzero(y == j) / len(y)

proportions_target = np.array([proportions_global] * n_bags)
bags_sizes_target = np.array([n_items // n_bags] * n_bags)
bags_sizes_target_naive = np.array([n_items // n_bags] * n_bags)

# Check if proportions_target is valid
print(f"Does the proportions target have stochastic rows?: {np.isclose(proportions_target.sum(axis=1), 1).all()}\n")

# Saving the base datasets (it will be the same for all variants)
df_base = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
df_base["y"] = y
df_base["y"] = df_base["y"].astype(int)
df_base.to_parquet("datasets-ci-multiclass/{}.parquet".format(base_dataset), index=False)

# Rescale the bags sizes to sum the same as the number of instances
bags_sizes_target = [int(bags_sizes_target[i] * X.shape[0] / sum(bags_sizes_target)) for i in range(len(bags_sizes_target))]
bags_sizes_target_naive = [int(bags_sizes_target_naive[i] * X.shape[0] / sum(bags_sizes_target_naive)) for i in range(len(bags_sizes_target_naive))]
n_bags = len(proportions_target)

# Generating the datasets 
for llp_variant in VARIANTS:
    bags = generate_llp_dataset(X, y, clusters, proportions_target, bags_sizes_target if llp_variant != "naive" else bags_sizes_target_naive, llp_variant, random_state)
    unique_bags, bags_sizes = np.unique(bags, return_counts=True)
    if len(unique_bags) != n_bags:
        print("\n\n")
        print(base_dataset)
        print(llp_variant)
        print(n_bags_type)
        print(bags_size_type)
        print(proportions_type)
        print(unique_bags)
        print(n_bags)
        raise Exception("ERROR: The number of bags is not correct")

    # Saving only the bags (saving space)
    df = pd.DataFrame(bags, columns=["bag"], dtype=int)
    
    filename = "datasets-ci-multiclass/{}-{}-{}-{}-{}-cluster-{}-{}.parquet".format(base_dataset, llp_variant, n_bags_type, \
                        bags_size_type, 
                        proportions_type if llp_variant != "naive" else "None", \
                        clustering_method if llp_variant in ["intermediate", "hard"] else "None", \
                        n_clusters if llp_variant in ["intermediate", "hard"] else "None")
    df.to_parquet(filename, index=False)
    print("Dataset {} generated".format(filename))
    print("Proportions: {}".format(compute_proportions_multiclass(bags, y)))
    print("Bag sizes: {}".format(np.bincount(bags)))
    print("\n------------------------\n")