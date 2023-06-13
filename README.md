# Evaluating LLP Methods: Challenges and Approaches

Repository for reproducing the results of the paper "Evaluating LLP Methods: Challenges and Approaches"

## Requirements
- Python 3.8 or higher (developed on Python 3.8)
- R version 4.2.1

```sh
pip3 install -r requirements.txt
```

To use MM[^1], LMM, and AMM[^2] it is necessary to get their code:

```sh
git clone https://github.com/giorgiop/almostnolabel.git
```
[^1]: Quadrianto, Novi, et al. "Estimating labels from label proportions." Proceedings of the 25th international conference on Machine learning. 2008.

[^2]: Patrini, Giorgio, et al. "(Almost) no label no cry." Advances in Neural Information Processing Systems 27 (2014).

To install the R libraries:
```sh
python3 install_r_libraries.py
```

Regarding the CI tests, we made some changes in FCIT[^3] to allow reproducibility. To get its code:
```sh
git clone https://github.com/gaabrielfranco/fcit.git
```

[^3]: Chalupka, Krzysztof, Pietro Perona, and Frederick Eberhardt. "Fast conditional independence test for vector variables with large sample sizes." arXiv preprint arXiv:1804.02747 (2018).

## Get and pre-processing the base datasets

First, go to the base datasets folder:

```sh
cd base-datasets
```

### Adult download and pre-processing:
```sh
wget https://archive.ics.uci.edu/static/public/2/adult.zip
python3 adult_preprocessing.py
```

The base dataset is saved as ```adult.parquet```

### CIFAR-10 download and pre-processing:

```sh
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
python3 cifar-10_preprocessing.py
```

The base dataset is saved as ```cifar-10-grey.parquet```

## Generate the datasets

In the ```llp-variants-datasets-benchmarks``` folder, run the following script:

```sh
./run_gen_datasets.sh
```

All the datasets are saved in the ```datasets-ci``` folder. The base datasets are also processed and saved in this folder, as ```adult.parquet``` and ```cifar-10-grey-animal-vehicle.parquet```.

## Run the CI tests

In the ```llp-variants-datasets-benchmarks``` folder, run the following script:

```sh
./run_ci_tests.sh
```

All the CI tests are saved in the ```ci-tests``` folder.

## Run an single experiment

```sh
python3 run_experiments.py -d {dataset_name} -m {model} -l {loss} -n {n_splits} -v {validation_size_percentage} -s {splitter} -e {execution_number}
```

As an example, we have:
```sh
python3 run_experiments -d adult-hard-large-equal-close-global-cluster-kmeans-5 -m lmm -l abs -n 3 -v 0.5 -s split-bag-bootstrap -e 0
```

For $k$-fold based methods, the *validation_size_percentage* is not used
```sh
python3 kdd_experiment.py -d adult-hard-large-equal-close-global-cluster-kmeans-5 -m lmm -l abs -n 3 -s split-bag-k-fold -e 0
```

## Run all the paper experiments

```sh
./run_all_experiments.sh
```

Each execution produces one ```parquet``` file. After running all the experiments, they can be combined into one single file (```datasets-benchmark-experiment-results.parquet```) as following:

```sh
python3 aggregate_results.py
```

## Produce all the plots in the paper

```sh
python3 plot_results.py -p best-methods
```

The plots are saved in the ```plots``` folder.

## Produce the results and extra information about the datasets in LaTeX table format
```sh
python3 plot_results.py -p table-all-results
python3 plot_results.py -p datasets-info
python3 plot_results.py -p table-ci-tests
```

The tables are saved in the ```tables``` folder.
