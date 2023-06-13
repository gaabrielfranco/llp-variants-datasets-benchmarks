# Script to generate all datasets calling dataset_gen.py

for base_dataset in "adult"
do
    for cluster in "kmeans"
    do
        for n_clusters in "5"
        do
            for n_bags in "small" "large"
            do
                for bags_size in "equal" "not-equal"
                do
                    for proportions in "close-global" "far-global" "mixed"
                    do
                        python3 dataset_gen.py --base_dataset $base_dataset --cluster $cluster --n_clusters $n_clusters --n_bags $n_bags --bags_size $bags_size --proportions $proportions
                    done
                done
            done
        done
    done
done

for base_dataset in "cifar-10-grey-animal-vehicle"
do
    for cluster in "kmeans"
    do
        for n_clusters in "5"
        do
            for n_bags in "small" "large"
            do
                for bags_size in "equal" "not-equal"
                do
                    for proportions in "close-global" "far-global" "mixed"
                    do
                        python3 dataset_gen.py --base_dataset $base_dataset --cluster $cluster --n_clusters $n_clusters --n_bags $n_bags --bags_size $bags_size --proportions $proportions
                    done
                done
            done
        done
    done
done
