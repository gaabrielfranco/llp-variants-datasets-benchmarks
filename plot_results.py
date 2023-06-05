import argparse
from copy import deepcopy
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import matplotlib
from scipy.stats import ttest_ind
import sys

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

#TODO: fix this

def plot_winning_figure(winning_df, filename, plot_type=None):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    #_ = plt.figure(figsize=(2, 1))
    plt.rc('font', size=6)

    # split_method_map = {
    #     "Split-\nBag\nBootstrap": "SP-BS",
    #     "Split-\nbag\nShuffle": "SP-SH",
    #     "Split-\nbag\nK-fold": "SP-KF",
    #     "SB\nBS",
    #     "SB\nSH",
    #     "SB\nKF",
    # 'full-bag-stratified-k-fold': "FB\nKF",
    # }

    if plot_type == "algorithms":
        fig, axes = plt.subplots(nrows=1, ncols=len(winning_df.index.levels[0]), sharey=True, gridspec_kw={"width_ratios": [2, 3, 3]}, figsize=(3.5, 3.5)) # width, height
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(winning_df.index.levels[0]), sharey=True, figsize=(3.5, 3.5)) # width, height
    if plot_type == "variants":
        it_array = list(winning_df.index.levels[0])
        it_array[2], it_array[3] = it_array[3], it_array[2]
        rotation_ticks = 45
    else:
        it_array = winning_df.index.levels[0]
        rotation_ticks = 0

    #print(winning_df.index.levels[1])

    # Rename level 1 index
    #winning_df.index = winning_df.index.set_levels([split_method_map[x] for x in winning_df.index.levels[1]], level=1)

    for i, row in enumerate(it_array):
        ax = axes[i]
        winning_df.loc[row, :].plot(ax=ax, kind='bar', width=.8, stacked=True, color=["tab:blue", "tab:red",  "tab:grey"], fontsize=6, legend=None)
        
        ax.set_xlabel(row, weight='bold')
        ax.set_axisbelow(True)

        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation_ticks)

        if i == 0:
            ax.tick_params(axis=u'x', which=u'both', length=0)
            ax.set_ylabel("Fraction of experiments", fontsize=6)
        else:
            ax.tick_params(axis=u'both', which=u'both', length=0)

    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=6)
    plt.tight_layout()
    # remove spacing in between
    fig.subplots_adjust(wspace=0)  # space between plots
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def compute_proportions(bags, y):
    """Compute the proportions for each bag given.
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
    proportions = np.empty(num_bags, dtype=float)
    for i in range(num_bags):
        bag = np.where(bags == i)[0]
        proportions[i] = np.count_nonzero(y[bag] == 1) / len(bag)
    return proportions


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
final_results = pd.read_parquet("../datasets-experiments-results/llp-datasets-experiments-results.parquet")
final_results = pd.concat([final_results, pd.read_parquet("../beta-experiments-results/beta-experiments-results.parquet")], ignore_index=True)

final_results.rename(columns={"metric": "error_metric"}, inplace=True)

final_results["error_metric"].replace(error_legend_map, inplace=True)
final_results["split_method"].replace(split_method_map, inplace=True)

final_results["split_method"] = final_results["split_method"] + "\n" + final_results["validation_size_perc"].astype(str)
final_results["split_method"] = final_results["split_method"].str.replace("nan", "")

final_results["dataset"] = final_results["dataset"] + "-" + final_results["n_splits"] + "folds"

# Removing some error metrics (using only abs in the paper)
final_results = final_results[final_results["error_metric"].isin(["abs"])]
final_results["error_metric"].replace(error_metric_map, inplace=True)

base_datasets = ["adult", "cifar-10-grey"]

base_datasets_type = {
    "cifar-10-grey": "Image-Objects",
    "adult": "Tabular"
}

dataset_map = {}

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
final_results["dataset"].replace(dataset_map, inplace=True)
# Creating a column with the dataset variant
final_results["dataset_variant"] = final_results["dataset"].apply(get_dataset_variant)

# Creating a columns with the base dataset
final_results["base_dataset"] = "None"
for dataset in base_datasets:
    final_results.loc[final_results.dataset.str.contains(dataset), "base_dataset"] = dataset

final_results["dataset_type"] = "None"
for base_dataset in base_datasets_type:
    final_results.loc[final_results.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]

# Removing CIFAR-10 intermediate that are not close-global
final_results = final_results[~((final_results.base_dataset == "cifar-10-grey") & \
                                (final_results.proportions != "close-global") & \
                                (final_results.dataset_variant == "Intermediate"))]

# We have a total of 72 datasets (80 - 8 that are not close-global for the intermediate variant of CIFAR-10)

if args.plot_type == "check-execs":
    #Test to check if the results are correct (n execs for kdd-lr/lmm)
    for model in ["EM/LR", "LMM", "MM"]:
        print(model, "\n")
        X = final_results[final_results["model"] == model]
        print(len(X))
        X = X.groupby(["dataset", "model", "loss", "split_method", "n_splits", "validation_size_perc"])

        for a, b in X:
            execs = b.exec
            execs = execs.astype(int)
            try:
                v = (sorted(execs) == np.array(range(30))).all()
            except:
                print("Error")
                print(a)
                print(b)
                print(execs)
                break
            if not v:
                print("Not all execs are present")
                print(a)
                print(execs)
                print("\n")
        print("End for model {}\n".format(model))
elif args.plot_type == "individual-results":
    # # Plotting individual results
    for dataset in final_results.dataset.unique():
        for model in final_results.model.unique():
            X = deepcopy(final_results[(final_results.dataset == dataset) & (final_results.model == model)])
            X["accuracy_test"] = X["accuracy_test"].astype(float)
            if len(X) == 0:
                continue
            ax = sns.pointplot(x="split_method", y="accuracy_test", hue="error_metric",
                data=X, 
                dodge=True, join=False, capsize=.2, errorbar=("se", 1.96))
            ax.set_ylabel("Accuracy", fontsize=6)
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=6)
            #ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=6)
            plt.tight_layout()
            plt.suptitle("Dataset {}\nAlgorithm {}".format(dataset, model), fontsize=6)
            filename = "plots/individual-results/{}_{}.pdf".format(dataset, model.replace("/", "-"))
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
            plt.close()
elif args.plot_type == "aggregate-results":
    for dataset in base_datasets:
        df_dataset = final_results[final_results.dataset.str.contains(dataset)]
        min_acc = df_dataset.accuracy_test.min()
        max_acc = df_dataset.accuracy_test.max()
        y_lim = (min_acc - 0.05, max_acc + 0.05)
        g = sns.FacetGrid(df_dataset, col="model", row="dataset", ylim=y_lim)
        #g = sns.FacetGrid(df_dataset, col="dataset", row="model", ylim=y_lim, height=5, aspect=1.5)
        g.map(sns.pointplot, "split_method", "accuracy_test", dodge=True, join=False, 
                capsize=.2, errorbar=("se", 1.96), order=sorted(df_dataset.split_method.unique()))
        g.set_axis_labels("", "Accuracy")
        plt.tight_layout()
        filename = "plots/agg_plots/{}.pdf".format(dataset)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
        plt.close()

    # Not sharing y
    for dataset in base_datasets:
        df_dataset = final_results[final_results.dataset.str.contains(dataset)]
        min_acc = df_dataset.accuracy_test.min()
        max_acc = df_dataset.accuracy_test.max()
        y_lim = (min_acc - 0.05, max_acc + 0.05)
        #g = sns.FacetGrid(df_dataset, col="dataset", row="model", sharey=False, height=5, aspect=1.5)
        g = sns.FacetGrid(df_dataset, col="model", row="dataset", sharey=False, height=5, aspect=1.5)
        g.map(sns.pointplot, "split_method", "accuracy_test", dodge=True, join=False, 
                capsize=.2, errorbar=("se", 1.96), order=sorted(df_dataset.split_method.unique()))
        g.set_axis_labels("", "Accuracy")
        plt.tight_layout()
        filename = "plots/agg_plots/{}-not-share-y.pdf".format(dataset)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
        plt.close()
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
        #dataset_info = dataset_info.append({"Dataset": dataset_map[dataset].replace("\n", ""), "Number of bags": len(df.bag.unique()), "Proportions and bag sizes": "\\begin{tabular}[c]{@{}l@{}}" + list2str(proportions) + "\\\\ " + list2str(bags_sizes) + "\\end{tabular}"}, ignore_index=True)
        dataset_info = dataset_info.append({"Dataset": dataset, "Number of bags": len(np.unique(bags)), "Proportions": list2str(proportions), "Bag sizes": list2str(bags_sizes)}, ignore_index=True)
    dataset_info.sort_values(by=["Dataset"], inplace=True)
    with pd.option_context("max_colwidth", 10000):
        dataset_info.to_latex(buf="plots/dataset-info-plots/table-datasets-info", index=False, escape=False)
elif args.plot_type == "datasets-info-plot":
    for base_dataset in base_datasets:
        for size in ["small", "large"]:
            N = 10 if size == "large" else 5
            for proportions_type in ["equal", "not-equal"]:
                X = final_results[(final_results.dataset.str.contains(base_dataset)) & (final_results.dataset.str.contains(f"{size}-{proportions_type}-"))]
                X = X[~X.dataset.str.contains("naive")]

                theta = radar_factory(N, frame='polygon')

                data = [
                    [f"Bag {i}" for i in range(N)],
                ]

                for dataset in X.dataset.unique():
                    
                    # Remove the 10folds or 5folds from the end of the dataset name
                    old_dataset = deepcopy(dataset)
                    dataset = dataset.split("-10folds" if "10folds" in dataset else "-5folds")[0]

                    # Reading X, y (base dataset) and bags (dataset)
                    df = pd.read_parquet("{}/{}.parquet".format("../datasets-ci", base_dataset))
                    X = df.drop(["y"], axis=1).values
                    y = df["y"].values.reshape(-1, 1)

                    df = pd.read_parquet("{}/{}.parquet".format("../datasets-ci", dataset))
                    bags = df["bag"].values

                    proportions = compute_proportions(bags, y)
                    proportions = [round(x, 2) for x in proportions]

                    old_dataset = "\n".join(old_dataset.split("-cluster-"))
                    data.append((old_dataset, [proportions, [round(y.sum() / len(y), 2)] * len(proportions)]))

                spoke_labels = data.pop(0)

                fig, axs = plt.subplots(figsize=(12, 12), nrows=3, ncols=3,
                                        subplot_kw=dict(projection='radar'), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

                # Find the min and max of values to set the rgrids
                vmin = round(min(np.min(d) for case_data in data for d in case_data[1]), 1)
                vmax = round(max(np.max(d) for case_data in data for d in case_data[1]), 1)

                r_grid_values = np.arange(vmin, vmax, 0.1)

                colors = ['b', 'r']
                for ax, (title, case_data) in zip(axs.flat, data):
                    ax.set_rgrids(r_grid_values)
                    ax.set_title(title, weight='bold', size='small', position=(0.5, 1.1),
                                horizontalalignment='center', verticalalignment='center')
                    for d, color in zip(case_data, colors):
                        ax.plot(theta, d, color=color)
                        ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
                    ax.set_varlabels(spoke_labels)
                    ax.tick_params(axis='both', labelsize=8)

                # add legend relative to top-left plot
                # labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
                # legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                #                          labelspacing=0.1, fontsize='small')

                plt.tight_layout()
                plt.savefig(f"plots/dataset-info-plots/radar-info-{base_dataset}-{size}-{proportions_type}.pdf", bbox_inches='tight', dpi=800)
                plt.close()
elif args.plot_type == "winning-figures":
    # Computing the W/L/D
    winning_df_shuffle = pd.DataFrame(columns=["dataset", "algorithm", "win", "lose", "draw"])
    winning_df_bootstrap = pd.DataFrame(columns=["dataset", "algorithm", "win", "lose", "draw"])
    winning_df_kfold = pd.DataFrame(columns=["dataset", "algorithm", "win", "lose", "draw"])

    for dataset in final_results.dataset.unique():
        for model in final_results.model.unique():
            accuracy_test_split_method = {}
            for split_method in final_results.split_method.unique():
                X = final_results[(final_results.dataset == dataset) & (final_results.model == model) & (final_results.split_method == split_method)]
                # Just add if X is not empty
                if len(X) > 0:
                    accuracy_test_split_method[split_method] = X.accuracy_test.values
            
            # Verify if accuracy_test_split_method is empty
            if len(accuracy_test_split_method) == 0:
                continue
            
            # If Full-bag K-fold is not in the accuracy_test_split_method, continue
            if "FB\nKF\n" not in accuracy_test_split_method:
                continue
            
            # Test if the results are statistically significant

            # Bootstrap
            if "SB\nBS\n0.5" in accuracy_test_split_method:
                sp_bootstrap_test = ttest_ind(accuracy_test_split_method["FB\nKF\n"],
                                                accuracy_test_split_method["SB\nBS\n0.5"], 
                                                equal_var=False, random_state=73921)
                if sp_bootstrap_test.pvalue <= 0.05:
                    # Win the one with the highest average accuracy
                    if np.mean(accuracy_test_split_method["FB\nKF\n"]) > np.mean(accuracy_test_split_method["SB\nBS\n0.5"]):
                        # Full bag wins
                        winning_df_bootstrap = pd.concat([pd.DataFrame([[dataset, model, 0, 1, 0]], columns=winning_df_bootstrap.columns), 
                                                            winning_df_bootstrap], ignore_index=True)
                    else:
                        # Split bag wins
                        winning_df_bootstrap = pd.concat([pd.DataFrame([[dataset, model, 1, 0, 0]], columns=winning_df_bootstrap.columns), 
                                                            winning_df_bootstrap], ignore_index=True)
                else:
                    # Draw
                    winning_df_bootstrap = pd.concat([pd.DataFrame([[dataset, model, 0, 0, 1]], columns=winning_df_bootstrap.columns), 
                                                        winning_df_bootstrap], ignore_index=True)

            # Shuffle
            if "SB\nSH\n0.5" in accuracy_test_split_method:
                sp_shuffle_test = ttest_ind(accuracy_test_split_method["FB\nKF\n"],
                                                accuracy_test_split_method["SB\nSH\n0.5"],
                                                equal_var=False, random_state=73921)
                if sp_shuffle_test.pvalue <= 0.05:
                    # Win the one with the highest average accuracy
                    if np.mean(accuracy_test_split_method["FB\nKF\n"]) > np.mean(accuracy_test_split_method["SB\nSH\n0.5"]):
                        # Full bag wins
                        winning_df_shuffle = pd.concat([pd.DataFrame([[dataset, model, 0, 1, 0]], columns=winning_df_shuffle.columns), 
                                                            winning_df_shuffle], ignore_index=True)
                    else:
                        # Split bag wins
                        winning_df_shuffle = pd.concat([pd.DataFrame([[dataset, model, 1, 0, 0]], columns=winning_df_shuffle.columns), 
                                                            winning_df_shuffle], ignore_index=True)
                else:
                    # Draw
                    winning_df_shuffle = pd.concat([pd.DataFrame([[dataset, model, 0, 0, 1]], columns=winning_df_shuffle.columns), 
                                                        winning_df_shuffle], ignore_index=True)

            # K-fold
            if "SB\nKF\n" in accuracy_test_split_method:
                sp_kfold_test = ttest_ind(accuracy_test_split_method["FB\nKF\n"],
                                                accuracy_test_split_method["SB\nKF\n"],
                                                equal_var=False, random_state=73921)
                if sp_kfold_test.pvalue <= 0.05:
                    # Win the one with the highest average accuracy
                    if np.mean(accuracy_test_split_method["FB\nKF\n"]) > np.mean(accuracy_test_split_method["SB\nKF\n"]):
                        # Full bag wins
                        winning_df_kfold = pd.concat([pd.DataFrame([[dataset, model, 0, 1, 0]], columns=winning_df_kfold.columns), 
                                                            winning_df_kfold], ignore_index=True)
                    else:
                        # Split bag wins
                        winning_df_kfold = pd.concat([pd.DataFrame([[dataset, model, 1, 0, 0]], columns=winning_df_kfold.columns), 
                                                            winning_df_kfold], ignore_index=True)
                else:
                    # Draw
                    winning_df_kfold = pd.concat([pd.DataFrame([[dataset, model, 0, 0, 1]], columns=winning_df_kfold.columns), 
                                                        winning_df_kfold], ignore_index=True)
    # Creating a column with the dataset variant
    winning_df_bootstrap["dataset_variant"] = winning_df_bootstrap["dataset"].apply(get_dataset_variant)
    winning_df_shuffle["dataset_variant"] = winning_df_shuffle["dataset"].apply(get_dataset_variant)
    winning_df_kfold["dataset_variant"] = winning_df_kfold["dataset"].apply(get_dataset_variant)

    # Creating a columns with the base dataset
    winning_df_bootstrap["base_dataset"] = "None"
    winning_df_shuffle["base_dataset"] = "None"
    winning_df_kfold["base_dataset"] = "None"
    for dataset in base_datasets:
        winning_df_bootstrap.loc[winning_df_bootstrap.dataset.str.contains(dataset), "base_dataset"] = dataset
        winning_df_shuffle.loc[winning_df_shuffle.dataset.str.contains(dataset), "base_dataset"] = dataset
        winning_df_kfold.loc[winning_df_kfold.dataset.str.contains(dataset), "base_dataset"] = dataset

    # Creating a column with the dataset type using base_datasets_type
    winning_df_bootstrap["dataset_type"] = "None"
    winning_df_shuffle["dataset_type"] = "None"
    winning_df_kfold["dataset_type"] = "None"
    for base_dataset in base_datasets_type:
        winning_df_bootstrap.loc[winning_df_bootstrap.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]
        winning_df_shuffle.loc[winning_df_shuffle.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]
        winning_df_kfold.loc[winning_df_kfold.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]
        
    winning_df_bootstrap["split_method"] = split_method_map["split-bag-bootstrap"]
    winning_df_shuffle["split_method"] = split_method_map["split-bag-shuffle"]
    winning_df_kfold["split_method"] = split_method_map["split-bag-k-fold"]

    winning_df = pd.concat([winning_df_bootstrap, winning_df_shuffle, winning_df_kfold], ignore_index=True)
    N_per_variant = winning_df.dataset_variant.value_counts()
    N_per_dataset_type = winning_df.dataset_type.value_counts()
    N_per_algorithm = winning_df.algorithm.value_counts()

    # Print the overall fraction of winning in the significant tests
    significant_tests = winning_df[(winning_df.win == 1) | (winning_df.lose == 1)]
    frac_win = significant_tests[significant_tests.win == 1].shape[0] / significant_tests.shape[0]
    print("Overall fraction of winning in the significant tests: %.2f" % (frac_win))

    # Main figure of the paper (W/L/D percentage per dataset variant)
    winning_df_variant_sm = winning_df.groupby(["dataset_variant", "split_method"]).sum()[["win", "lose", "draw"]]
    # Rename the index from variant to variant + N (number of datasets)
    for variant in winning_df_variant_sm.index.get_level_values(0):    
        winning_df_variant_sm.rename(index={variant: variant + "\n(N=" + str(N_per_variant[variant]) + ")"}, inplace=True)
    winning_df_variant_sm = winning_df_variant_sm.div(winning_df_variant_sm.sum(axis=1), axis=0)
    winning_df_variant_sm.rename(columns={"win": "S > F", "lose": "S < F", "draw": "No significance"}, inplace=True)

    filename = "plots/winning-figures/aggregate_result_per_variant.pdf"
    plot_winning_figure(winning_df_variant_sm, filename, plot_type="variants")
    
    # # # Per dataset type
    # winning_df_type_sm = winning_df.groupby(["dataset_type", "split_method"]).sum()[["win", "lose", "draw"]]
    # for dataset_type in winning_df_type_sm.index.get_level_values(0):    
    #     winning_df_type_sm.rename(index={dataset_type: dataset_type + "\n(N=" + str(N_per_dataset_type[dataset_type]) + ")"}, inplace=True)
    # winning_df_type_sm = winning_df_type_sm.div(winning_df_type_sm.sum(axis=1), axis=0)
    # winning_df_type_sm.rename(columns={"win": "S > F", "lose": "S < F", "draw": "No significance"}, inplace=True)

    # filename = "plots/winning-figures/aggregate_result_per_dataset_type.pdf"
    # plot_winning_figure(winning_df_type_sm, filename)

    # Per algorithm
    winning_df_algorithm_sm = winning_df.groupby(["algorithm", "split_method"]).sum()[["win", "lose", "draw"]]
    for algorithm in winning_df_algorithm_sm.index.get_level_values(0):
        winning_df_algorithm_sm.rename(index={algorithm: algorithm + "\n(N=" + str(N_per_algorithm[algorithm]) + ")"}, inplace=True)
    winning_df_algorithm_sm = winning_df_algorithm_sm.div(winning_df_algorithm_sm.sum(axis=1), axis=0)
    winning_df_algorithm_sm.rename(columns={"win": "S > F", "lose": "S < F", "draw": "No significance"}, inplace=True)

    filename = "plots/winning-figures/aggregate_result_per_algorithm.pdf"
    plot_winning_figure(winning_df_algorithm_sm, filename, plot_type="algorithms")

elif args.plot_type == "table-best-methods":
    df_best_methods = pd.DataFrame(columns=["base_dataset", "dataset_variant", "n_bags", "bag_sizes", "proportions", "best_hyperparam_method", "best_algorithm", "best_in_both"])
    # Excluding Naive variant
    for base_dataset in sorted(final_results.base_dataset.unique()):
        for llp_variant in sorted(final_results.dataset_variant.unique()):
            # if llp_variant == "Naive":
            #     continue
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
                        #x = best_method.groupby(["split_method", "model"]).mean(numeric_only=True).accuracy_test.sort_values(ascending=False)
                        x = best_method.groupby(["split_method", "model"]).mean(numeric_only=True).f1_test.sort_values(ascending=False)
                        best_global_combination = set()

                        # The top (split_method, model) combination is always included in the best global
                        best_global_combination.add((x.index[0][0], x.index[0][1]))
                        for i in range(1, len(x.index)):
                            split_method_1, model_1 = x.index[0] #x.index[i]
                            split_method_2, model_2 = x.index[i] #x.index[i+1]

                            #acc_1 = best_method[(best_method.split_method == split_method_1) & (best_method.model == model_1)].accuracy_test.values
                            #acc_2 = best_method[(best_method.split_method == split_method_2) & (best_method.model == model_2)].accuracy_test.values
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

    # Categorizing the best_in_both
    def get_best_in_both_cat(x):
        if "SB" in x and "FB" in x:
            prefix = "SB+FB"
        elif "SB" in x:
            prefix = "SB"
        elif "FB" in x:
            prefix = "FB"

        if "AMM" in x and "LMM" in x and "DLLP" in x and "MM" in x and "EM/LR" in x:
            return f"{prefix} and ALL"
        elif ("AMM" in x or "LMM" in x or "MM" in x) and not ("DLLP" in x or "EM/LR" in x):
            return f"{prefix} and MM family"
        elif "DLLP" in x and not ("AMM" in x or "LMM" in x or "MM" in x or "EM/LR" in x):
            return f"{prefix} and DLLP"
        elif "EM/LR" in x and not ("AMM" in x or "LMM" in x or "MM" in x or "DLLP" in x):
            return f"{prefix} and EM/LR"
        else:
            return f"{prefix} and mixed"
        
    df_best_methods["best_in_both_cat"] = df_best_methods.best_in_both.apply(get_best_in_both_cat)

    # Categorizinh the best_algorithm
    def get_best_algorithm_cat(x):
        if "AMM" in x and "LMM" in x and "DLLP" in x and "MM" in x and "EM/LR" in x:
            return "All"
        elif ("AMM" in x or "LMM" in x or "MM" in x) and not ("DLLP" in x or "EM/LR" in x):
            return "MM family"
        elif "DLLP" in x and not ("AMM" in x or "LMM" in x or "MM" in x or "EM/LR" in x):
            return "DLLP"
        elif "EM/LR" in x and not ("AMM" in x or "LMM" in x or "MM" in x or "DLLP" in x):
            return "EM/LR"
        else:
            return "Mixed"
        
    df_best_methods["best_algorithm_cat"] = df_best_methods.best_algorithm.apply(get_best_algorithm_cat)

    # Print best hyperparams (global)
    D = df_best_methods.best_hyperparam_method_cat.value_counts()
    # Normalize the count
    D = D/float(D.sum())
    D = D.reset_index(name="count")
    D.rename(columns={"index": "Best Hyperparam. Selection Method",
                        "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Best Hyperparam. Selection Method", y="Frequency", data=D, kind="bar", errorbar=None)
    filename = "plots/table-best-methods/best-hyperparam-methods-global.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm (global)
    D = df_best_methods.best_algorithm.value_counts()
    # Normalize the count
    D = D/float(D.sum())
    D = D.reset_index(name="count")
    D.rename(columns={"index": "Best Algorithm",
                        "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Best Algorithm", y="Frequency", data=D, kind="bar", errorbar=None)
    plt.xticks(rotation=45)
    filename = "plots/table-best-methods/best-algorithms-global.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm cat (global)
    D = df_best_methods.best_algorithm_cat.value_counts()
    # Normalize the count
    D = D/float(D.sum())
    D = D.reset_index(name="count")
    D.rename(columns={"index": "Best Algorithm (Category)",
                        "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Best Algorithm (Category)", y="Frequency", data=D, kind="bar", errorbar=None)
    plt.xticks(rotation=45)
    filename = "plots/table-best-methods/best-algorithms-category-global.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best in both (global)
    D = df_best_methods.best_in_both_cat.value_counts()
    # Normalize the count
    D = D/float(D.sum())
    D = D.reset_index(name="count")
    D.rename(columns={"index": "Best (HS, Alg.) combination",
                        "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Best (HS, Alg.) combination", y="Frequency", data=D, kind="bar", errorbar=None)
    plt.xticks(rotation=45)
    filename = "plots/table-best-methods/best-in-both-global.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best hyperparams per dataset variant
    D = df_best_methods.groupby(["dataset_variant"]).best_hyperparam_method_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=0,group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_hyperparam_method_cat": "Best Hyperparam. Selection Method",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Dataset Variant", y="Frequency", hue="Best Hyperparam. Selection Method", data=D, kind="bar", errorbar=None)
    filename = "plots/table-best-methods/best-hyperparam-methods-per-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm per dataset variant
    D = df_best_methods.groupby(["dataset_variant"]).best_algorithm.value_counts()
    # Normalize the count
    D = D.groupby(level=0,group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_algorithm": "Best Algorithm",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Dataset Variant", y="Frequency", hue="Best Algorithm", data=D, kind="bar", errorbar=None)
    filename = "plots/table-best-methods/best-algorithms-per-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm (categorical) per dataset variant
    D = df_best_methods.groupby(["dataset_variant"]).best_algorithm_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=0,group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_algorithm_cat": "Best Algorithm (Category)",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Dataset Variant", y="Frequency", hue="Best Algorithm (Category)", data=D, kind="bar", errorbar=None)
    filename = "plots/table-best-methods/best-algorithms-category-per-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best in both per dataset variant
    D = df_best_methods.groupby(["dataset_variant"]).best_in_both_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=0,group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_in_both_cat": "Best (HS, Alg.) combination",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Dataset Variant", y="Frequency", hue="Best (HS, Alg.) combination", data=D, kind="bar", errorbar=None)
    filename = "plots/table-best-methods/best-in-both-per-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best hyperparams per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_hyperparam_method_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_hyperparam_method_cat": "Best Hyperparam. Selection Method",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.7)
    g = sns.catplot(x="Base Dataset", y="Frequency", hue="Best Hyperparam. Selection Method", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=2, aspect=0.75)
    g.set_titles("Variant:\n{col_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=6)
    plt.tight_layout()
    filename = "plots/table-best-methods/best-hyperparam-methods-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_algorithm.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_algorithm": "Best Algorithm",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.7)
    g = sns.catplot(x="Base Dataset", y="Frequency", hue="Best Algorithm", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=2, aspect=0.75)
    g.set_titles("Variant:\n{col_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=6)
    plt.tight_layout()
    filename = "plots/table-best-methods/best-algorithms-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_algorithm_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_algorithm_cat": "Best Algorithm (Category)",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Base Dataset", y="Frequency", hue="Best Algorithm (Category)", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8)
    plt.tight_layout()
    filename = "plots/table-best-methods/best-algorithms-category-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best in both per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_in_both_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_in_both_cat": "Best (HS, Alg.) combination",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "count": "Frequency"}, inplace=True)
    sns.set(font_scale=0.8)
    sns.catplot(x="Base Dataset", y="Frequency", hue="Best (HS, Alg.) combination", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8)
    plt.tight_layout()
    filename = "plots/table-best-methods/best-in-both-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    df_best_methods.to_csv("plots/table-best-methods/best_methods_table.csv", index=False)

elif args.plot_type == "beta-experiments":

    split_method_beta = ['SB\nBS\n[0.21, 0.25, 0.27, 0.33, 0.3, 0.37, 0.48, 0.49, 0.5, 0.5]',
    'SB\nSH\n[0.21, 0.25, 0.27, 0.33, 0.3, 0.37, 0.48, 0.49, 0.5, 0.5]',
    'SB\nBS\n[0.22, 0.27, 0.28, 0.34, 0.33, 0.38, 0.49, 0.49, 0.5, 0.5]',
    'SB\nSH\n[0.22, 0.27, 0.28, 0.34, 0.33, 0.38, 0.49, 0.49, 0.5, 0.5]',
    'SB\nBS\n[0.25, 0.26, 0.32, 0.32, 0.37, 0.36, 0.49, 0.49, 0.5, 0.5]',
    'SB\nSH\n[0.25, 0.26, 0.32, 0.32, 0.37, 0.36, 0.49, 0.49, 0.5, 0.5]',
    'SB\nBS\n[0.17, 0.29, 0.11, 0.15, 0.28, 0.41, 0.5, 0.5, 0.5, 0.5]',
    'SB\nSH\n[0.17, 0.29, 0.11, 0.15, 0.28, 0.41, 0.5, 0.5, 0.5, 0.5]']

    split_method_beta_reverse = ['SB\nSH\n[0.75, 0.74, 0.68, 0.68, 0.63, 0.64, 0.51, 0.51, 0.5, 0.5]',
    'SB\nBS\n[0.75, 0.74, 0.68, 0.68, 0.63, 0.64, 0.51, 0.51, 0.5, 0.5]',
    'SB\nSH\n[0.78, 0.73, 0.72, 0.66, 0.67, 0.62, 0.51, 0.51, 0.5, 0.5]',
    'SB\nBS\n[0.78, 0.73, 0.72, 0.66, 0.67, 0.62, 0.51, 0.51, 0.5, 0.5]',
    'SB\nSH\n[0.79, 0.75, 0.73, 0.67, 0.7, 0.63, 0.52, 0.51, 0.5, 0.5]',
    'SB\nBS\n[0.79, 0.75, 0.73, 0.67, 0.7, 0.63, 0.52, 0.51, 0.5, 0.5]',
    'SB\nSH\n[0.83, 0.71, 0.89, 0.85, 0.72, 0.59, 0.5, 0.5, 0.5, 0.5]',
    'SB\nBS\n[0.83, 0.71, 0.89, 0.85, 0.72, 0.59, 0.5, 0.5, 0.5, 0.5]']

    split_method_beta_extreme_case = ['SB\nBS\n0.05', 'SB\nSH\n0.05']
    split_method_beta_standard_case = ['SB\nBS\n0.2', 'SB\nSH\n0.2']

    datasets_beta = ["cifar-10-grey-animal-vehicle-naive-large-not-equal-None-cluster-None-None-10folds",
                     "cifar-10-grey-animal-vehicle-simple-large-not-equal-mixed-cluster-None-None-10folds",
                     "cifar-10-grey-animal-vehicle-intermediate-large-not-equal-close-global-cluster-kmeans-5-10folds",
                     "cifar-10-grey-animal-vehicle-hard-large-not-equal-close-global-cluster-kmeans-5-10folds"]

    X = final_results[final_results.dataset.isin(datasets_beta)]

    X = X[X.split_method.isin(split_method_beta + split_method_beta_extreme_case)]

    # Plot the distribution
    def f(x):
        if "BS" in x:
            if "[" in x:
                return "BS-opt"
            else:
                return "BS-extreme"
        elif "SH" in x:
            if "[" in x:
                return "SH-opt"
            else:
                return "SH-extreme"


    X["split_method"] = X.split_method.apply(f)

    X_gb = X.groupby(["dataset", "model"])

    df_difs_f1 = pd.DataFrame(columns=["dataset", "model", "split_method", "f1_test_diff"])

    for info, data in X_gb:
        a = data[data.split_method == "BS-opt"].f1_test.values
        b = data[data.split_method == "BS-extreme"].f1_test

        df_difs_f1 = pd.concat([df_difs_f1, pd.DataFrame({
            "dataset": info[0],
            "model": info[1],
            "split_method": "BS",
            "f1_test_diff": a - b
        })], ignore_index=True)

        a = data[data.split_method == "SH-opt"].f1_test.values
        b = data[data.split_method == "SH-extreme"].f1_test

        df_difs_f1 = pd.concat([df_difs_f1, pd.DataFrame({
            "dataset": info[0],
            "model": info[1],
            "split_method": "SH",
            "f1_test_diff": a - b
        })], ignore_index=True)


    # Plot the distribution
    sns.set(font_scale=0.8)


    g = sns.FacetGrid(df_difs_f1, col="dataset", row="model", hue="split_method", sharex=True, sharey=False, margin_titles=True, height=2, aspect=1.5)
    g.map(sns.kdeplot, "f1_test_diff", cumulative=False)
    g.add_legend()
    plt.show()
    exit()

    X_beta = X[X.split_method.isin(split_method_beta)]
    X_beta = X_beta.groupby(["dataset", "model", "split_method"])

    print("Optimal beta\n")
    for info, data in X_beta:
        dataset, model, split_method = info
        #split_method_orig = "\n".join(split_method.split("\n")[:2]) + "\n0.5"
        split_method_orig = "\n".join(split_method.split("\n")[:2]) + "\n0.05"
        X_orig = X[(X.dataset == dataset) & (X.model == model) & (X.split_method == split_method_orig)]
        if len(X_orig) != 30 or len(data) != 30:
            print("Not enough data for", dataset, model, split_method)
            exit()
                    
        best_method = ttest_ind(X_orig.f1_test.values, data.f1_test.values,
            equal_var=False, random_state=73921)
        if best_method.pvalue <= 0.05:
            # Getting the extreme beta case
            # beta_extreme_case = "\n".join(split_method.split("\n")[:2]) + "\n0.05"
            # X_extreme = X[(X.dataset == dataset) & (X.model == model) & (X.split_method == beta_extreme_case)]
            # if len(X_extreme) != 30:
            #     print("Not enough data for", dataset, model, beta_extreme_case)
            #     exit()
            
            # best_method_extreme = ttest_ind(data.f1_test.values, X_extreme.f1_test.values,
            #     equal_var=False, random_state=73921)

            print("Significant difference for", dataset, model, split_method.replace("\n", "-"))
            print("p-value:", best_method.pvalue)
            print("F1-Score")
            #print("Original (Beta=0.5): %.4f" % X_orig.f1_test.mean(), end="\t")
            print("Extreme: %.4f" % X_orig.f1_test.mean(), end="\t")
            print("Beta opt.: %.4f" % data.f1_test.mean(), end="\t")
            #print("Beta=0.05: %.4f" % X_extreme.f1_test.mean(), end="\t")
            #print("p-value (Beta opt. vs Beta=0.05): %.4f" % best_method_extreme.pvalue)
            print("\n")
        # else:
        #     print("No significant difference for", dataset, model, split_method.replace("\n", " "))
        #     print("p-value:", best_method.pvalue)
        #     print("\n-------------------\n")
    exit()
    # Reverse case
    X_beta = X[X.split_method.isin(split_method_beta_reverse)]
    X_beta = X_beta.groupby(["dataset", "model", "split_method"])

    print("Reverse optimal beta\n")
    for info, data in X_beta:
        dataset, model, split_method = info
        split_method_orig = "\n".join(split_method.split("\n")[:2]) + "\n0.5"
        X_orig = X[(X.dataset == dataset) & (X.model == model) & (X.split_method == split_method_orig)]
        if len(X_orig) != 30 or len(data) != 30:
            print("Not enough data for", dataset, model, split_method)
            exit()
                    
        best_method = ttest_ind(X_orig.f1_test.values, data.f1_test.values,
            equal_var=False, random_state=73921)
        if best_method.pvalue <= 0.05:
            # Getting the extreme beta case
            beta_extreme_case = "\n".join(split_method.split("\n")[:2]) + "\n0.05"
            X_extreme = X[(X.dataset == dataset) & (X.model == model) & (X.split_method == beta_extreme_case)]
            if len(X_extreme) != 30:
                print("Not enough data for", dataset, model, beta_extreme_case)
                exit()
            
            best_method_extreme = ttest_ind(data.f1_test.values, X_extreme.f1_test.values,
                equal_var=False, random_state=73921)

            print("Significant difference for", dataset, model, split_method.replace("\n", "-"))
            print("p-value:", best_method.pvalue)
            print("F1-Score")
            print("Original (Beta=0.5): %.4f" % X_orig.f1_test.mean(), end="\t")
            print("Beta opt.: %.4f" % data.f1_test.mean(), end="\t")
            print("Beta=0.05: %.4f" % X_extreme.f1_test.mean(), end="\t")
            print("p-value (Beta opt. vs Beta=0.05): %.4f" % best_method_extreme.pvalue)
            print()
        # else:
        #     print("No significant difference for", dataset, model, split_method.replace("\n", " "))
        #     print("p-value:", best_method.pvalue)
        #     print("\n-------------------\n")

    # Extreme case
    print("\nExtreme case\n")
    X_beta_wc = X[X.split_method.isin(split_method_beta_extreme_case)]
    X_beta_wc = X_beta_wc.groupby(["dataset", "model", "split_method"])

    for info, data in X_beta_wc:
        dataset, model, split_method = info
        split_method_orig = "\n".join(split_method.split("\n")[:2]) + "\n0.5"
        X_orig = X[(X.dataset == dataset) & (X.model == model) & (X.split_method == split_method_orig)]
        if len(X_orig) != 30 or len(data) != 30:
            print("Not enough data for", dataset, model, split_method)
            exit()
                    
        best_method = ttest_ind(X_orig.f1_test.values, data.f1_test.values,
            equal_var=False, random_state=73921)
        if best_method.pvalue <= 0.05:
            print("Significant difference for", dataset, model, split_method.replace("\n", "-"))
            print("p-value:", best_method.pvalue)
            print("F1-Score")
            print("Original (Beta=0.5): %.4f" % X_orig.f1_test.mean(), end="\t")
            print("Beta=0.05: %.4f" % data.f1_test.mean())
            print()
        # else:
        #     print("No significant difference for", dataset, model, split_method.replace("\n", " "))
        #     print("p-value:", best_method.pvalue)
        #     print("\n-------------------\n")

    # Standard case
    print("\nStandard case\n")
    X_beta_wc = X[X.split_method.isin(split_method_beta_standard_case)]
    X_beta_wc = X_beta_wc.groupby(["dataset", "model", "split_method"])

    for info, data in X_beta_wc:
        dataset, model, split_method = info
        split_method_orig = "\n".join(split_method.split("\n")[:2]) + "\n0.5"
        X_orig = X[(X.dataset == dataset) & (X.model == model) & (X.split_method == split_method_orig)]
        if len(X_orig) != 30 or len(data) != 30:
            print("Not enough data for", dataset, model, split_method)
            exit()
                    
        best_method = ttest_ind(X_orig.f1_test.values, data.f1_test.values,
            equal_var=False, random_state=73921)
        if best_method.pvalue <= 0.05:
            print("Significant difference for", dataset, model, split_method.replace("\n", "-"))
            print("p-value:", best_method.pvalue)
            print("F1-Score")
            print("Original (Beta=0.5): %.4f" % X_orig.f1_test.mean(), end="\t")
            print("Beta=0.2: %.4f" % data.f1_test.mean())
            print()
        # else:
        #     print("No significant difference for", dataset, model, split_method.replace("\n", " "))
        #     print("p-value:", best_method.pvalue)
        #     print("\n-------------------\n")

