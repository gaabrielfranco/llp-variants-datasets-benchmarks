import argparse
from copy import deepcopy
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from scipy.stats import ttest_ind
from llp_learn.util import compute_proportions

def get_dataset_variant(dataset):
    if "naive" in dataset:
         return "Naive"
    elif "simple" in dataset:
        return "Simple"
    elif "intermediate" in dataset:
        return "Intermediate"
    elif "hard" in dataset:
        return "Hard"
    else:
        return "unknown"

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--plot_type', "-p", type=str, required=True, help="Plot to generate")
args = parser.parse_args()

VARIANTS = ["naive", "simple", "intermediate", "hard"]

dataset_title_map = {}
error_legend_map = {
    "error_bag_abs": "Abs loss",
     "error_accuracy_validation": "Oracle"
}

"""
Attention:
    This can change depending on the experiments - e.g. n_folds and validation perc size
"""
split_method_map = {
    'split-bag-bootstrap': "SB\nBS",
    'split-bag-shuffle': "SB\nSH",
    'split-bag-k-fold': "SB\nKF",
    'full-bag-stratified-k-fold': "FB\nKF",
}

error_metric_map = {
    "error_bag_abs": "Abs loss",
    "hypergeo": "Prob. based loss"
}

# TODO: think about this - maybe beta experiments in a separate dataframe
final_results = pd.read_parquet("datasets-benchmark-experiment-results.parquet")

final_results.rename(columns={"metric": "error_metric"}, inplace=True)

final_results["error_metric"].replace(error_legend_map, inplace=True)
final_results["split_method"].replace(split_method_map, inplace=True)

final_results["split_method"] = final_results["split_method"] + "\n" + final_results["validation_size_perc"].astype(str)
final_results["split_method"] = final_results["split_method"].str.replace("nan", "")

final_results["dataset"] = final_results["dataset"] + "-" + final_results["n_splits"] + "folds"
final_results["error_metric"].replace(error_metric_map, inplace=True)

base_datasets = ["adult", "cifar-10-grey"]

base_datasets_type = {
    "cifar-10-grey": "Image-Objects",
    "adult": "Tabular"
}

model_map = {
    "lmm": "LMM",
    "llp-svm-lin": "Alter-SVM",
    "kdd-lr": "EM/LR",
    "mm": "MM",
    "dllp": "DLLP",
    "amm": "AMM"
}

# Getting the infos about the datasets
final_results["n_bags"] = final_results.dataset.apply(lambda x: "large" if "large" in x else "small")
final_results["bag_sizes"] = final_results.dataset.apply(lambda x: "not-equal" if "not-equal" in x else "equal")
final_results["proportions"] = final_results.dataset.apply(lambda x: "close-global" if "close-global" in x else "far-global" if "far-global" in x else "mixed" if "mixed" in x else "none")

final_results["model"].replace(model_map, inplace=True)
# Creating a column with the dataset variant
final_results["dataset_variant"] = final_results["dataset"].apply(get_dataset_variant)

# Creating a columns with the base dataset
final_results["base_dataset"] = "None"
for dataset in base_datasets:
    final_results.loc[final_results.dataset.str.contains(dataset), "base_dataset"] = dataset

final_results["dataset_type"] = "None"
for base_dataset in base_datasets_type:
    final_results.loc[final_results.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]

# We have a total of 72 datasets (80 - 8 that are not close-global for the intermediate variant of CIFAR-10)
if args.plot_type == "check-n-experiments":
    total_models = len(final_results)
    print("Total trained models: ", total_models)
    for model in final_results["model"].unique():
        print(model, len(final_results[final_results["model"] == model]))
    print("")
    # Checking number of experiments
    n_experiments_df = final_results.groupby(["model", "dataset", "split_method"]).size().reset_index(name='counts').sort_values(by="counts", ascending=False)
    print("Total number of experiments:", len(n_experiments_df))
    print("Experiments per split_method")
    print(n_experiments_df["split_method"].value_counts())
elif args.plot_type == "aggregate-category-results":
    for base_dataset in base_datasets:
        for size in ["small", "large"]:
            for proportions_type in ["equal", "not-equal"]:
                for llp_variant in VARIANTS:
                    df_dataset = final_results[(final_results.dataset.str.contains(base_dataset)) & (final_results.dataset.str.contains(llp_variant)) & (final_results.dataset.str.contains(f"{size}-{proportions_type}-"))]
                    min_acc = df_dataset.accuracy_test.min()
                    max_acc = df_dataset.accuracy_test.max()
                    y_lim = (min_acc - 0.05, max_acc + 0.05)
                    g = sns.FacetGrid(df_dataset, col="model", row="dataset", ylim=y_lim, height=5, aspect=1.5)
                    g.map(sns.pointplot, "split_method", "accuracy_test", dodge=True, join=False, 
                            capsize=.2, errorbar=("se", 1.96), order=sorted(df_dataset.split_method.unique()))
                    g.set_axis_labels("", "Accuracy")
                    g.set_titles("{row_name}\n{col_name}")
                    plt.tight_layout()
                    filename = f"plots/agg-category-plots/{base_dataset}-{size}-{proportions_type}-{llp_variant}.pdf"
                    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
                    plt.close()
elif args.plot_type == "datasets-info":
    dataset_info = pd.DataFrame(columns=["Dataset", "Number of bags", "Proportions", "Bag sizes"])

    files = glob.glob("../datasets-ci/*.parquet")
    files = sorted(files)
    # Removing base datasets

    files = [file for file in files if "adult.parquet" not in file]
    files = [file for file in files if "cifar-10-grey-animal-vehicle.parquet" not in file]

    for file in files:

        dataset = file.split("/")[-1].split(".")[0]

        # Extracting variant
        llp_variant = [variant for variant in VARIANTS if variant in dataset][0]
        print("LLP variant: {}".format(llp_variant))
        
        # Extracting base dataset
        base_dataset = dataset.split(llp_variant)[0]
        base_dataset = base_dataset[:-1]

        # Reading X, y (base dataset) and bags (dataset)
        df = pd.read_parquet("{}/{}.parquet".format("../datasets-ci", base_dataset))
        X = df.drop(["y"], axis=1).values
        y = df["y"].values.reshape(-1, 1)

        df = pd.read_parquet(file)
        bags = df["bag"].values

        proportions = compute_proportions(bags, y)
        proportions = [round(x, 2) for x in proportions]
        bags_sizes = np.bincount(bags)
        list2str = lambda x: ("(" + ",".join([str(y) for y in x]) + ")").replace(",)", ")")
        dataset_info = dataset_info.append({"Dataset": dataset, "Number of bags": len(np.unique(bags)), "Proportions": list2str(proportions), "Bag sizes": list2str(bags_sizes)}, ignore_index=True)
    dataset_info.sort_values(by=["Dataset"], inplace=True)
    with pd.option_context("max_colwidth", 10000):
        dataset_info.to_latex(buf="plots/dataset-info-plots/table-datasets-info", index=False, escape=False)
elif args.plot_type == "table-best-methods":
    df_best_methods = pd.DataFrame(columns=["base_dataset", "dataset_variant", "n_bags", "bag_sizes", "proportions", "best_hyperparam_method", "best_algorithm", "best_in_both"])
    for base_dataset in sorted(final_results.base_dataset.unique()):
        for llp_variant in sorted(final_results.dataset_variant.unique()):
            for n_bags in sorted(final_results.n_bags.unique()):
                for bag_sizes in sorted(final_results.bag_sizes.unique()):
                    for proportions in sorted(final_results.proportions.unique()):
                        if proportions == "none" and llp_variant != "Naive":
                            continue # Skip the none proportion (naive case)

                        best_method = deepcopy(final_results[(final_results.base_dataset == base_dataset) & (final_results.dataset_variant == llp_variant) & (final_results.n_bags == n_bags) & (final_results.bag_sizes == bag_sizes) & (final_results.proportions == proportions)])

                        # Combination doesn't exist (case of CIFAR-10 intermediate that are not close-global)
                        if best_method.shape[0] == 0:
                            continue
                        
                        # Removing the \n from the split method (used to make the table more readable)
                        best_method["split_method"] = best_method.split_method.apply(lambda x: x.replace("\n", " "))
                        best_method["split_method"] = best_method.split_method.apply(lambda x: x.replace("KF ", "KF"))

                        # Get the overall best method
                        x = best_method.groupby(["split_method", "model"]).mean(numeric_only=True).f1_test.sort_values(ascending=False)
                        best_global_combination = set()

                        # The top (split_method, model) combination is always included in the best global
                        best_global_combination.add((x.index[0][0], x.index[0][1]))
                        for i in range(1, len(x.index)):
                            split_method_1, model_1 = x.index[0] 
                            split_method_2, model_2 = x.index[i]

                            acc_1 = best_method[(best_method.split_method == split_method_1) & (best_method.model == model_1)].f1_test.values
                            acc_2 = best_method[(best_method.split_method == split_method_2) & (best_method.model == model_2)].f1_test.values

                            best_models_test = ttest_ind(acc_1, acc_2, equal_var=False, random_state=73921)
                            if best_models_test.pvalue <= 0.05:
                                # The top (split_method, model) is better than the i-th (split_method, model) combination
                                break
                            else:
                                # split_method_1, model_1 are already in the best global combination.
                                # Then, add split_method_2, model_2
                                best_global_combination.add((split_method_2, model_2))

                        # Get the best model (algorithm) for this combination of parameters
                        accuracy_models = {}
                        avg_accuracy_models = {}
                        for model in best_method.model.unique():
                            #accuracy_models[model] = deepcopy(best_method[best_method.model == model].accuracy_test.values)
                            accuracy_models[model] = deepcopy(best_method[best_method.model == model].f1_test.values)
                            avg_accuracy_models[model] = np.mean(accuracy_models[model])

                        avg_accuracy_models = sorted(avg_accuracy_models.items(), key=lambda x: x[1], reverse=True)
                        best_models = set()

                        # The top model is always be in the best model set
                        best_models.add(avg_accuracy_models[0][0])

                        for i in range(1, len(avg_accuracy_models)):
                            best_models_test = ttest_ind(accuracy_models[avg_accuracy_models[0][0]],
                                accuracy_models[avg_accuracy_models[i][0]],
                                equal_var=False, random_state=73921)
                            if best_models_test.pvalue <= 0.05:                                
                                break
                            else:
                                best_models.add(avg_accuracy_models[i][0])

                        # Get the best hyperparameter method
                        # Each model "votes" for the best hyperparameter method
                        best_split_method_votes = {}
                        for split_method in best_method.split_method.unique():
                            best_split_method_votes[split_method] = 0

                        for model in best_method.model.unique():
                            accuracy_split_method = {}
                            avg_accuracy_split_method = {}
                            # Get the best hyperparameter method for this model
                            best_method_model = best_method[best_method.model == model]
                            for split_method in best_method_model.split_method.unique():
                                #accuracy_split_method[split_method] = deepcopy(best_method_model[best_method_model.split_method == split_method].accuracy_test.values)
                                accuracy_split_method[split_method] = deepcopy(best_method_model[best_method_model.split_method == split_method].f1_test.values)
                                avg_accuracy_split_method[split_method] = np.mean(accuracy_split_method[split_method])

                            avg_accuracy_split_method = sorted(avg_accuracy_split_method.items(), key=lambda x: x[1], reverse=True)
                            best_split_method = set()

                            # The top split method is always be in the best split method set
                            best_split_method.add(avg_accuracy_split_method[0][0])

                            for i in range(1, len(avg_accuracy_split_method)):
                                best_split_method_test = ttest_ind(accuracy_split_method[avg_accuracy_split_method[0][0]],
                                    accuracy_split_method[avg_accuracy_split_method[i][0]],
                                    equal_var=False, random_state=73921)
                                if best_split_method_test.pvalue <= 0.05:
                                    break
                                else:
                                    best_split_method.add(avg_accuracy_split_method[i][0])

                            for split_method in best_split_method:
                                best_split_method_votes[split_method] += 1

                        # Get the best hyperparameter method
                        best_split_method_votes = sorted(best_split_method_votes.items(), key=lambda x: x[1], reverse=True)
                        
                        # Getting the split methods with max values from the tuple (split_method, votes)
                        max_votes = best_split_method_votes[0][1]
                        best_split_method = set()
                        for split_method, votes in best_split_method_votes:
                            if votes == max_votes:
                                best_split_method.add(split_method)
                            else:
                                break

                        df_best_methods = pd.concat([df_best_methods, pd.DataFrame({
                           "base_dataset": base_dataset,
                           "dataset_variant": llp_variant,
                           "n_bags": n_bags,
                           "bag_sizes": bag_sizes,
                           "proportions": proportions,
                           "best_hyperparam_method": str(sorted(best_split_method)),
                           "best_algorithm": str(sorted(best_models)),
                           "best_in_both": str(sorted(best_global_combination))
                        }, index=[0])], ignore_index=True)

    # Categorizing the best_hyperparam_method 
    def get_best_hyperparam_method_cat(x):
        if "SB" in x and "FB" in x:
            return "SB+FB"
        elif "SB" in x:
            return "SB"
        elif "FB" in x:
            return "FB"
        
    df_best_methods["best_hyperparam_method_cat"] = df_best_methods.best_hyperparam_method.apply(get_best_hyperparam_method_cat)

    # Print best hyperparams per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_hyperparam_method_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_hyperparam_method_cat": "Best Hyperparam. Selection Method",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    #sns.set(font_scale=0.7)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)

    g = sns.catplot(row="Base Dataset", y="Frequency", x="Best Hyperparam. Selection Method", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=1.1, aspect=1.1, sharex=True)
    g.set_titles("{row_name}\n{col_name}")
    g.set_xlabels("")
    plt.tight_layout()
    filename = "plots/best-hyperparam-methods-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_algorithm.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D["best_algorithm"] = D.best_algorithm.apply(lambda x: x.translate({ord("["): "", ord("]"): "", ord("'"): "", ord(" "): ""}))
    ba = D["best_algorithm"].unique()
    D["best_algorithm_legend"] = D["best_algorithm"].apply(lambda x: np.where(ba == x)[0][0])
    D.rename(columns={"best_algorithm": "Best Algorithm(s)",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "best_algorithm_legend": "Best Algorithm Index",
                      "count": "Frequency"}, inplace=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)

    g = sns.catplot(row="Base Dataset", y="Frequency", x="Best Algorithm Index", hue="Best Algorithm(s)", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=1.1, aspect=1.1, sharex=True, dodge=False)#, width=1)
    g.set_titles("{row_name}\n{col_name}")
    g.set_xlabels("")
    g.set_xticklabels("")
    for ax in g.axes.flat:
        ax.set_xticks([])
    plt.legend(bbox_to_anchor=(1.1, 1.8), loc=2, borderaxespad=0., fontsize=5)
    plt.tight_layout()
    filename = "plots/best-algorithms-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()