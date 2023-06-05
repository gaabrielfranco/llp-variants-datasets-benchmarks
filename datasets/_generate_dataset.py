from copy import deepcopy
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from ipfn import ipfn

def derivative_f(B, C, A): 
    """
    Computes the gradient of f = ||C - BA||_F^2 w.r.t. A.

    Arguments:
        B: a numpy array
        C: a numpy array
        A: a numpy array
    Returns:
        the gradient of f w.r.t. A
    """
    return (-2 * (B.T @ C)) + (2 * (B.T @ B @ A))


def gd(B, C, n_clusters, n_bags, alpha=10e-8, max_iter = 10000, tol = 10e-6, n_runs=10, random_state=None):
    """
    Find A s.t. ||C - BA||_F^2 is minimized using gradient descent.

    Arguments:
        B: a numpy array
        C: a numpy array
        n_clusters: number of clusters
        n_bags: number of bags
        alpha: learning rate
        max_iter: maximum number of iterations
        tol: tolerance
        n_runs: number of runs to find the best A
        random_state: random state
    Returns:
        A: a numpy array
    """
    random = np.random.RandomState(random_state)

    best_A = None
    best_f = 10e5

    for i in range(n_runs):
        # Feasible A for starting
        A = random.normal(0.5, 0.5, size=(n_clusters, n_bags))
        A[A < 0] = 0
        A[A > 1] = 1
        A_sum = A.sum(axis=1, keepdims=True)
        A_sum[A_sum == 0] = 1
        A = A / A_sum
        
        old_A = np.full(A.shape, 10e5, float)

        convergence = False
        for i in range(max_iter):
            # Step 1) GD
            current_gradient_value = derivative_f(B, C, A)
            A = A - alpha * current_gradient_value

            # Step 2) Projection
            A[A < 0] = 0
            A[A > 1] = 1
            A_sum = A.sum(axis=1, keepdims=True)
            A_sum[A_sum == 0] = 1
            A = A / A_sum

            if (np.linalg.norm(A - old_A) / np.linalg.norm(A)) <= tol:
                convergence = True
                
                # Update best A
                if np.linalg.norm(C - B @ A) < best_f:
                    best_f = np.linalg.norm(C - B @ A)
                    best_A = deepcopy(A)
                break
            else:
                old_A = deepcopy(A)

    if not convergence:
        warnings.warn("max_iter: %d reached without convergence in %d runs" % (max_iter, n_runs))

    return best_A

def llp_variant_generation(X, y=None, llp_variant="naive", bags_size_target=np.array([]), proportions_target=np.array([]), clusters=np.array([]), random_state=None):
    valid_llp_variants = [
        "naive",
        "simple",
        "intermediate",
        "hard"
    ]

    if not llp_variant in valid_llp_variants:
        raise Exception("LLP variant is not valid")

    if isinstance(proportions_target, list):
        proportions_target = np.array(proportions_target)

    if isinstance(bags_size_target, list):
        bags_size_target = np.array(bags_size_target)

    if bags_size_target.size == 0:
            raise Exception("bags_size_target must be not empty")

    if (llp_variant=="intermediate" or llp_variant=="hard") and clusters.size == 0:
        raise Exception("clusters must be not empty for intermediate and hard variants")
    
    if clusters.size != 0:
        n_clusters = np.max(clusters) + 1

    random = np.random.RandomState(random_state)
    n_bags = len(bags_size_target)
    p_bags_size_target = bags_size_target / bags_size_target.sum()

    if llp_variant == "naive":
        bags = random.choice(n_bags, size=len(X), p=p_bags_size_target)
    elif llp_variant == "simple":
        proportions_global = len(y[y == 1]) / len(y)
        p_positives = (proportions_target * p_bags_size_target) / proportions_global
        p_negatives = ((1 - proportions_target) * p_bags_size_target) / (1 - proportions_global)

        if not np.isclose(p_positives.sum(), 1).all():
            warnings.warn("p_positives.sum() != 1 - it will be normalized")
            p_positives /= p_positives.sum()
        if not np.isclose(p_negatives.sum(), 1).all():
            warnings.warn("p_negatives.sum() != 1 - it will be normalized")
            p_negatives /= p_negatives.sum()

        if np.isclose(p_positives, p_negatives).all():
            raise Exception("p_positives must not be equal to p_negatives. Change the proportions_target or bags_size_target")
         
        idx_negatives = np.where(y == 0)[0]
        idx_positives = np.where(y == 1)[0]

        bags = np.empty(len(y), int)
        bags[idx_negatives] = random.choice(range(n_bags), size=len(idx_negatives), p=p_negatives)
        bags[idx_positives] = random.choice(range(n_bags), size=len(idx_positives), p=p_positives)
    elif llp_variant == "intermediate":        
        n_positives_target = np.round(proportions_target * bags_size_target).astype(int)
        n_negatives_target = (bags_size_target - n_positives_target).astype(int)

        P_yx = np.zeros((2, n_clusters), dtype=int)
        P_yb = np.array([n_negatives_target, n_positives_target], dtype=int)

        for cluster in range(n_clusters):
            cluster_idx = np.where(clusters == cluster)[0]
            y_cluster = y[cluster_idx]
            P_yx[0, cluster] = int(np.round(np.sum(y_cluster == 0)))
            P_yx[1, cluster] = int(np.round(np.sum(y_cluster == 1)))

        A = gd(P_yx, P_yb, n_clusters, n_bags, alpha=10e-12, max_iter = 10000, tol = 0.0001, random_state=random_state)

        if A is None or (A.sum(axis=0) == 0).any():
            raise Exception("The solution found has at least one empty bag. Try to change the proportions_target and/or bags_size_target")

        # Sampling step
        bags = np.empty(len(X), dtype=int)

        for cluster in range(n_clusters):
            cluster_idx = np.where(clusters == cluster)[0]
            bags[cluster_idx] = random.choice(range(n_bags), size=len(cluster_idx), replace=True, p=A[cluster, :])
    elif llp_variant == "hard":
        L = len(bags_size_target)
        K = n_clusters
        C = 2

        p_x = np.array([float(len(np.where(clusters == i)[0])) for i in range(n_clusters)])
        p_x /= p_x.sum()
        p_y = np.array([len(y[y == 0]) / len(y), len(y[y == 1]) / len(y)])
        p_b = bags_size_target / bags_size_target.sum()
        p_y_given_x = np.empty(K)
        for i in range(K):
            cluster_idx = np.where(clusters == i)[0]
            p_y_given_x[i] = y[cluster_idx].sum() / len(cluster_idx)
        p_y_b = np.array([(1 - proportions_target) * p_b, proportions_target * p_b])
        p_y_x = np.array([(1 - p_y_given_x) * p_x, p_y_given_x * p_x])

        A = random.uniform(0, 1, size=(C, K, L))
        A /= A.sum()

        aggregates = [p_y, p_x, p_b, p_y_x, p_y_b]
        dimensions = [[0], [1], [2], [0, 1], [0, 2]]

        IPF = ipfn.ipfn(A, aggregates, dimensions, max_iteration=1000)
        A = IPF.iteration()

        P_b_xy = deepcopy(A)
        for x in range(K):
            for label in range(C):
                P_b_xy[label, x, :] /= P_b_xy[label, x, :].sum()

        bags = np.empty(len(X), dtype=int)
        for cluster in range(K):
            idx_cluster = np.where(clusters == cluster)[0]
            y_cluster = y[idx_cluster]
            for label in range(C):
                idx_cluster_label = np.where(y_cluster == label)[0]
                p_label = P_b_xy[label, cluster, :]
                bags[idx_cluster[idx_cluster_label]] = random.choice(range(L), size=len(idx_cluster_label), replace=True, p=p_label)
    else:
        raise Exception("llp_variant must be one of ['naive', 'simple', 'intermediate', 'hard']")

    return bags

