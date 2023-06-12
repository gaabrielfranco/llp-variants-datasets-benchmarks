for dataset in "adult-hard-large-equal-close-global-cluster-kmeans-5" "adult-hard-large-equal-far-global-cluster-kmeans-5" "adult-hard-large-equal-mixed-cluster-kmeans-5" "adult-hard-large-not-equal-close-global-cluster-kmeans-5" "adult-hard-large-not-equal-far-global-cluster-kmeans-5" "adult-hard-large-not-equal-mixed-cluster-kmeans-5" "adult-intermediate-large-equal-close-global-cluster-kmeans-5" "adult-intermediate-large-equal-far-global-cluster-kmeans-5" "adult-intermediate-large-equal-mixed-cluster-kmeans-5" "adult-intermediate-large-not-equal-close-global-cluster-kmeans-5" "adult-intermediate-large-not-equal-far-global-cluster-kmeans-5" "adult-intermediate-large-not-equal-mixed-cluster-kmeans-5" "adult-naive-large-equal-None-cluster-None-None" "adult-naive-large-not-equal-None-cluster-None-None" "adult-simple-large-equal-close-global-cluster-None-None" "adult-simple-large-equal-far-global-cluster-None-None" "adult-simple-large-equal-mixed-cluster-None-None" "adult-simple-large-not-equal-close-global-cluster-None-None" "adult-simple-large-not-equal-far-global-cluster-None-None" "adult-simple-large-not-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-hard-large-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-not-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-not-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-large-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-large-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-naive-large-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-naive-large-not-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-not-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-not-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-not-equal-mixed-cluster-None-None"
do
    for model in "kdd-lr" "lmm" "mm" "dllp" "amm"
    do
        for n_splits in "10"
        do
            for splitter in "full-bag-stratified-k-fold" "split-bag-k-fold"
            do
                for ((exec=0; exec<30; exec++))
                do
                    python3 run_experiments.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -e $exec
                done
            done
            for splitter in "split-bag-bootstrap" "split-bag-shuffle"
            do
                for ((exec=0; exec<30; exec++))
                do
                    python3 run_experiments.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -v 0.5 -e $exec
                done
            done
        done
    done
done

for dataset in "adult-hard-small-equal-close-global-cluster-kmeans-5" "adult-hard-small-equal-far-global-cluster-kmeans-5" "adult-hard-small-equal-mixed-cluster-kmeans-5" "adult-hard-small-not-equal-close-global-cluster-kmeans-5" "adult-hard-small-not-equal-far-global-cluster-kmeans-5" "adult-hard-small-not-equal-mixed-cluster-kmeans-5" "adult-intermediate-small-equal-close-global-cluster-kmeans-5" "adult-intermediate-small-equal-far-global-cluster-kmeans-5" "adult-intermediate-small-equal-mixed-cluster-kmeans-5" "adult-intermediate-small-not-equal-close-global-cluster-kmeans-5" "adult-intermediate-small-not-equal-far-global-cluster-kmeans-5" "adult-intermediate-small-not-equal-mixed-cluster-kmeans-5" "adult-naive-small-equal-None-cluster-None-None" "adult-naive-small-not-equal-None-cluster-None-None" "adult-simple-small-equal-close-global-cluster-None-None" "adult-simple-small-equal-far-global-cluster-None-None" "adult-simple-small-equal-mixed-cluster-None-None" "adult-simple-small-not-equal-close-global-cluster-None-None" "adult-simple-small-not-equal-far-global-cluster-None-None" "adult-simple-small-not-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-hard-small-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-not-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-not-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-small-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-small-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-naive-small-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-naive-small-not-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-not-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-not-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-not-equal-mixed-cluster-None-None"
do
    for model in "kdd-lr" "lmm" "mm" "dllp" "amm"
    do
        for n_splits in "5"
        do
            for splitter in "full-bag-stratified-k-fold" "split-bag-k-fold"
            do
                for ((exec=0; exec<30; exec++))
                do
                    python3 run_experiments.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -e $exec
                done
            done
            for splitter in "split-bag-bootstrap" "split-bag-shuffle"
            do
                for ((exec=0; exec<30; exec++))
                do
                    python3 run_experiments.py -d $dataset -m $model -l "abs" -n $n_splits -s $splitter -v 0.5 -e $exec
                done
            done
        done
    done
done