## All codes written by Paul Platzer.
# These codes are attached to the following publication:
# Paul Platzer, Arthur Avenas, Bertrand Chapron, Lucas Drumetz, Alexis Mouche, Pierre Tandeo, Léo Vinour, Distance Learning for Analog Methods. 2024. ⟨hal-04841334⟩
# The codes in this file aim at reproducing the methodology described in Alessandrini et al. (2018) used to find optimal weights given to different variables used in analog distance definition. Although the iterative form of this method differs from pure grid-search, the term "grid_search" is used lightly in this code to refer to the method of Alessandrini et al. (2018).

import numpy as np
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.neighbors import NearestNeighbors
import itertools
from joblib import Parallel, delayed
import sys
from analogs import apply_transform, find_analogues, compute_weights, compute_diffs, compute_mae_mad, compute_error

## Generate combination of p-weights so that:
# 1. each weight is in [0, 0.1, 0.2, ... , 0.9, 1]
# 2. the sum of all p weights equals 1
def generate_weight_combinations(p):
    def find_partitions(n, k, prefix=()):
        """Recursively generate partitions of n into k parts (values in [0,10])"""
        if k == 1:
            if n <= 10:
                yield prefix + (n,)
            return
        for i in range(min(n, 10) + 1):
            yield from find_partitions(n - i, k - 1, prefix + (i,))

    valid_combinations = set()  # Use a set to avoid duplicate permutations
    for partition in find_partitions(10, p):
        # Convert from integer partition to weight partition
        weights = tuple(x / 10 for x in partition)
        valid_combinations.add(weights)  # Store unique combinations

    return list(valid_combinations)

## Sub-function to allow parallel computation over different weight combinations
def process_variable(ind_new_var, dataset_x, dataset_y, Itrain, Itest, k, loo, corr_length_train, ind_vars_selected, nn_algo, n_jobs):
    """Function to process a single variable in parallel."""
    # New variable set including tested new variable
    ind_new_vars_selection = np.append(ind_vars_selected, ind_new_var)

    # All possible weight combinations for the new number of selected variables
    weight_combinations = generate_weight_combinations(len(ind_new_vars_selection))

    CRPS_train_combinations = []
    CRPS_test_combinations = []
    transform_diagonal_test_combinations = []

    for weights_test in weight_combinations:
        # Define transformation from variable weights
        transform_diagonal_test = np.zeros(dataset_x.shape[1], dtype='float64')
        transform_diagonal_test[ind_new_vars_selection] = weights_test
        transform_diagonal_test_combinations.append(transform_diagonal_test)

        # Transform training explanatory variables
        dataset_X = apply_transform(dataset_x, transform_diagonal_test, np.arange(len(dataset_x)))
        dataset_X = dataset_X[:, ind_new_vars_selection]

        # Initiate nearest neighbors algorithm
        nn = NearestNeighbors( algorithm = nn_algo , n_neighbors = k + int(loo==True) + 2*corr_length_train, n_jobs = n_jobs )
        nn.fit( dataset_X[Itrain] )

        # Compute error on train set
        error_train = compute_error(dataset_X[Itrain], dataset_y, Itrain, Itrain, k, nn, loo=loo, corr_length_train=corr_length_train, vector_out=False, error_type='CRPS')
        
        # Compute error on independent test set if provided (no LOO procedure)
        if len(Itest)>0:
            test_X = dataset_X[Itest]    
            error_test = compute_error(test_X, dataset_y, Itrain, Itest, k, nn, loo=False, corr_length_train=0, error_type='CRPS')
        else:
            error_test = np.array(np.nan)
        
        # Store errors
        CRPS_test_combinations.append( error_test.copy() )
        CRPS_train_combinations.append( error_train.copy() )

    # Select best combination
    ind_best_combination = np.argmin(np.array(CRPS_train_combinations))
    return (CRPS_train_combinations[ind_best_combination],
            CRPS_test_combinations[ind_best_combination],
            transform_diagonal_test_combinations[ind_best_combination])


# Main function to use parallel processing
def grid_search_CRPS_TC(dataset_x, dataset_y, Itrain, Itest, k=200, nn_algo='kd_tree',
                        thresh_CRPS_gain=0.03, Nvars=np.inf, loo=False, corr_length_train=0, n_jobs_variables=1, n_jobs_nnsearch=1):
    # TO COMMENT
    
    CRPS_train = []
    CRPS_test = []
    transform_diagonals = [np.zeros(dataset_x.shape[1], dtype='float64')]
    CRPS_gain = 9999.9

    if Nvars < np.inf:
        thresh_CRPS_gain = 0

    nvars = 0
    while CRPS_gain > thresh_CRPS_gain and nvars < Nvars:
        print(f'No. of vars selected = {len(CRPS_train) + 1}/{dataset_x.shape[1]}')
        ind_vars_not_selected = np.where(transform_diagonals[-1] == 0)[0]
        ind_vars_selected = np.where(transform_diagonals[-1] != 0)[0]

        # Parallel search over the incrementally-added new variable, searched inside all possibilities from "ind_vars_not_selected" ("not YET selected")
        batch_size = max(1, len(ind_vars_not_selected) // (2 * n_jobs_variables))  # Adjust batch size dynamically

        results = []
        with tqdm_joblib(tqdm(desc="Processing Variables", total=len(ind_vars_not_selected), leave=True)) as progress_bar:
            for i in range(0, len(ind_vars_not_selected), batch_size):
                batch_results = Parallel(n_jobs=n_jobs_variables)(
                    delayed(process_variable)(ind_new_var, dataset_x, dataset_y, Itrain, Itest, k, loo, 
                                              corr_length_train, ind_vars_selected, nn_algo, n_jobs_nnsearch)
                    for ind_new_var in ind_vars_not_selected[i:i + batch_size]
                )
                results.extend(batch_results)


        # Extract results
        CRPS_train_newvars, CRPS_test_newvars, transform_diagonal_newvars = zip(*results)

        ind_best_var = np.argmin(np.array(CRPS_train_newvars))
        transform_diagonals.append(transform_diagonal_newvars[ind_best_var])
        CRPS_train.append(CRPS_train_newvars[ind_best_var])
        CRPS_test.append(CRPS_test_newvars[ind_best_var])

        nvars = np.sum(transform_diagonals[-1] != 0)

        if len(CRPS_train) == 1:
            CRPS_gain = 9999.9
        elif len(CRPS_train) == dataset_x.shape[1]:
            break
        else:
            CRPS_gain = (CRPS_train[-2] - CRPS_train[-1]) / CRPS_train[-2]

    return np.array(transform_diagonals), np.array(CRPS_train), np.array(CRPS_test)
