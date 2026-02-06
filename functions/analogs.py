## All codes written by Paul Platzer.
# These codes are attached to the following publication:
# Paul Platzer, Arthur Avenas, Bertrand Chapron, Lucas Drumetz, Alexis Mouche, Pierre Tandeo, Léo Vinour, Distance Learning for Analog Methods. 2024. ⟨hal-04841334⟩

import numpy as np
# import numexpr as ne # -> useless in new version
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors


#############################################################################
### Routines necessary for both MSE-based and CRPS-based gradient descent ###
#############################################################################

def apply_transform(dataset_x, transform, Ind1, Ind2=[]):
    ## Sub-function to apply linear transformation to explanatory variables in order to build catalogue and target sets ##

    ## Inputs:
    # "dataset_x" : input variables from dataset, shape = (number_of_samples, number_of_input_features)
    # "transform" : linear transformation to be applied to input features,
    #               shape = either (number_of_features, number_of_features) [[square matrix]] or (number_of_features) [[diagonal matrix]]
    # "Ind1" : sample indices from "dataset_x" to which the transformation must be applied, for instance "Icat" or "Itest"
    # "Ind2" : optional second set of indices from "dataset_x" to which the transformation must be applied, for instance "Itar"

    ## Outputs:
    # "set1_X", "set2_X" : transformed input variables at the desired indices Ind1 and (optional) Ind2

    
    # Find-out if the linear transformation is diagonal
    is_diag = [None,True,False][len(transform.shape)]

    # Apply linear transformation
    if is_diag :
        set1_x = dataset_x[Ind1]
        set1_X = set1_x * transform
        if len(Ind2) != 0:
            set2_x = dataset_x[Ind2]
            set2_X = set2_x * transform
    else :
        set1_X = np.matmul( dataset_x[Ind1] , transform.T )
        if len(Ind2) != 0:
            set2_X = np.matmul( dataset_x[Ind2] , transform.T )

    if len(Ind2) != 0:
        return set1_X, set2_X
    else:
        return set1_X


def find_analogues(nn, tar_X, k, loo=False, corr_length_train=0, ind_tar=[]):
    ## Sub-function based on scikit-learn's "NearestNeighbors" to find analogues and related analog-to-target distances ##

    ## Inputs:
    # "nn" : Output NearestNeighbor algorithm fitted on catalogue "cat_X".
    # "k": Number of analogues to use in algorithm
    # "tar_X" : Targets of which analogues must be searched. shape = (number_of_targets , number_of_features)
    # "loo" : Boolean to indicate if leave-one-out proceure has to be applied. This is in case the targets "tar_X" are included in the catalogue used to build the nearest neighbor function "nn"
    # "corr_length_train" : Auto-correlation-time of the catalogue/target set (only used if loo==True). This is to be sure that we do not select "analogues" that are only time-neighbors of the target (i.e. using yesterday's weather as an analogue of today's weather to make a forecast...)
    # "ind_tar" : Indices of the targets in the catalog. Useful if corr_length_train>0. If the target set equals exactly the catalog, then ind_tar does not have to be specified and is set to np.arange(len(tar_X)).
    # /!\ Note that if corr_length_train>0, then the catalog used to build "nn" must be time-ordered !

    ## Outputs:
    # "dist" : Analog-to-target Euclidean distances, shape = (number_of_targets, number_of_analogs)
    # "ind" : Analog indices, shape = (number_of_targets, number_of_analogs)

    # Find analogues + possibly time-neighbors
    dist, ind = nn.kneighbors( tar_X , n_neighbors = k + int(loo==True) + 2*corr_length_train, return_distance=True )

    # Leave-one-out procedure: if target set is included in catalog, the first "analog" is the target itself and must be removed
    dist = dist[:,int(loo==True):]; ind = ind[:,int(loo==True):]
    
    # Leave-out samples that are below the correlation time-scale of the data
    if corr_length_train > 0:
        # Collect indices of targets. The indices of the analogues in the catalog must be at least corr_length_train away from ind_tar
        if len(ind_tar)==0:
            ind_tar = np.arange(len(ind))
        dist[ np.abs( ind - np.repeat(ind_tar[:,np.newaxis], k+2*corr_length_train, axis=1) )
              < corr_length_train ] = 10**12
        sort = np.argsort(dist , axis=1)
        dist = np.take_along_axis(dist , sort , axis=1)
        ind = np.take_along_axis(ind , sort , axis=1)
        ind = ind[:,:k] ; dist = dist[:,:k]

    return dist, ind


def compute_weights(dist):
    ## Sub-function to compute probabilities given to analogues from the set of analogue-to-target distances

    ## Input:
    # "dist" : analog-to-target Euclidean distances, shape = (number_of_targets, number_of_analogues)

    ## Output:
    # "p" : probabilities (weights) given to each analogue, shape = (number_of_targets, number_of_analogues)
    # note: p is build so that the sum over the analogues equals 1, i.e. np.sum(p, axis=1) is an array of ones
    
    # Square distances
    dist_squared = dist**2
    
    # Add the smallest distance to prevent from having zeros
    w = np.exp( - dist_squared.T + dist_squared[:,0] ).T 
    
    # Convert to sum-1 probabilities
    p = ( w.T / np.sum( w , axis = -1 ) ).T

    return p


def compute_diffs(dataset_y, Itar, Icat, ind_tar, ind_cat, k):
    ## Sub-function to compute differences between analogues and targets, and between different analogues of a same target ##
    ## Used to compute CRPS and gradient of CRPS ##

    ## Inputs:
    # "dataset_y": output features of the dataset, shape = (number_of_samples, number_of_output_features)
    # "Itar": indices of target-set, shape = (number_of_targets)
    # "ind_tar": e.g. ind_tar=ind_batch during mini-batch gradient-descent
    # "ind_cat": e.g. ind_cat=ind when "ind" is the set of analogues
    # "k": number of analogues to use in algorithm

    ## Outputs:
    # "diff_y" : differences in output space between analogues and target, shape = (batch_size , k_number_of_analogues, number_of_output_features)
    # "diff_ana_y" : differences in output space between analogues of a same target, shape = (batch_size, k_number_of_analogues , k_number_of_analogues, number_of_output_features)

    diff_y = np.abs(dataset_y[Itar[ind_tar], None, :] - dataset_y[Icat[ind_cat]])S

    ana_y = dataset_y[Icat[ind_cat]]
    diff_ana_y = np.abs(ana_y[:, None, :, :] - ana_y[:, :, None, :])
    
    return diff_y, diff_ana_y


def compute_mae_mad(p, diff_y, diff_ana_y):
    ## TO COMMENT
    
    # Compute mean absolute error and difference (here we already perform the sum over output features)
    mae = np.sum(p[..., None] * diff_y, axis=(1,2))
    mad = np.sum( p[:, :, None, None] * np.transpose( p[:, :, None, None] * diff_ana_y , axes = [0,2,1,3] ), axis=(1, 2, 3) )

    return mae, mad
    

def compute_error(test_X, dataset_y, Icat, Itest, k, nn, loo=False, corr_length_train=0, vector_out=False, error_type='CRPS'):
    ## Sub-function to compute MSE or CRPS of analogue ensemble ##

    ## Inputs:
    # "test_X": transformed input features over the test set which is used to compute CRPS or MSE
    # "dataset_y": output features of the dataset, shape = (number_of_samples, number_of_output_features)
    # "Itar": indices of target-set, shape = (number_of_targets)
    # "Icat": indices of catalogue-set, shape = (number_of_samples_in_catalogue)
    # "k": number of analogues to use in algorithm
    # "nn" : output NearestNeighbor algorithm fitted on catalogue "cat_X"
    # "loo" : Boolean to indicate if leave-one-out proceure has to be applied. This is in case the targets "tar_X" are included in the catalogue used to build the nearest neighbor function "nn"
    # "corr_length_train" : Auto-correlation-time of the catalogue/target set (only used if loo==True). This is to be sure that we do not select "analogues" that are only time-neighbors of the target (i.e. using yesterday's weather as an analogue of today's weather to make a forecast...)
    # "vector_out" : Boolean. If "True", and error_type='MSE', average analogue estimate is given as output. If 'True and error_type='CRPS',  If "False", only CRPS or MSE is given as output.
    
    ## Outputs:
    # "CRPS": scalar value of average CRPS (averaged over all samples and all output features)
    # "mae": Mean-absolute error (averaged over all output features), shape = (number_of_targets)
    # "mad": Mean-absolute dfference (averaged over all output features), shape = (number_of_targets)
    # "MSE": scalar value of average MSE (averaged over all samples and all output features)
    # "y_av": average analogue estimate, shape = (number_of_targets, number_of_output_features)
    
    # Find analogues and related distances
    dist, ind = find_analogues(nn, test_X, k, loo, corr_length_train)
        
    # Compute weights
    p = compute_weights(dist)

    if error_type=='MSE':
        # Compute average analogue estimate
        y_av = np.sum((dataset_y[Icat][ind].T*p.T).T, axis=1)
    
        # Compute MSE
        MSE = np.mean( np.sum( ( y_av - dataset_y[Itest] )**2 , axis=1 ) )

        if vector_out:
            return MSE, y_av
        else:
            return MSE
    
    elif error_type=='CRPS':
        # Compute differences in output space between analogues and target and analogues themselves
        diff_y, diff_ana_y = compute_diffs(dataset_y, Itest, Icat, range(len(Itest)), ind, k)
    
        # Compute mean absolute error and difference (here we already perform the sum over output features (index "m" in einsum))
        mae, mad = compute_mae_mad(p, diff_y, diff_ana_y)
    
        # Compute CRPS
        CRPS = np.mean( mae - .5 * mad )
    
        if vector_out:
            return CRPS, mae, mad
        else:
            return CRPS
