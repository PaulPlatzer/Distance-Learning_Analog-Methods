## All codes written by Paul Platzer.
# These codes are attached to the following publication:
# Paul Platzer, Arthur Avenas, Bertrand Chapron, Lucas Drumetz, Alexis Mouche, Pierre Tandeo, Léo Vinour, Distance Learning for Analog Methods. 2024. ⟨hal-04841334⟩

import numpy as np
# import numexpr as ne
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors
from collections.abc import Sequence

import sys
from analogs import apply_transform, find_analogues, compute_weights, compute_diffs, compute_mae_mad, compute_error


#####################################
### Gradient computation routines ###
#####################################


def compute_gradient_MSE(dataset_x, dataset_y, Itar, Icat, ind_tar, ind_cat, k, y_av, p, transform):
    # TO COMMENT
    
    is_diag = [None,True,False][len(transform.shape)]

    diff_x = np.repeat(dataset_x[Itar[ind_tar], np.newaxis, :], k, axis=1) - dataset_x[Icat][ind_cat]
    # diff_x = dataset_x[Itar[ind_tar], None, :] - dataset_x[Icat][ind_cat] # MODIFIED -> TO CHECK
    
    if is_diag:
        diff_x_2 = diff_x**2 # for diagonal matrices
    else:
        diff_x_2 = np.einsum( 'ijk,ijl->ijkl', diff_x, diff_x) # for non-diagonal matrices
        # diff_x_2 = diff_x[..., None] * diff_x[:, :, None, :] # MODIFIED -> TO CHECK

    diff_yhat_yana = np.repeat(y_av[:,np.newaxis], k, axis=1) - dataset_y[Icat[ind_cat]] # noted "\hat{y}_i - y_j" in manuscript
    diff_yhat_ytrue = np.repeat( ( y_av - dataset_y[Itar[ind_tar]] ) [:, np.newaxis], k, axis=1) # noted "\hat{y}_i - y_i" in manuscript
    # diff_yhat_yana = y_av[:, None] - dataset_y[Icat[ind_cat]] # MODIFIED -> TO CHECK
    # diff_yhat_ytrue = (y_av - dataset_y[Itar[ind_tar]])[:, None] # MODIFIED -> TO CHECK
    
    Vij = np.einsum( 'ijm->ij' , diff_yhat_yana * diff_yhat_ytrue )
    # Vij = np.sum(diff_yhat_yana * diff_yhat_ytrue, axis=-1) # MODIFIED -> TO CHECK
    
    if is_diag:
        V = (2/len(Itar[ind_tar])) * np.einsum( 'ij,ij,ijl->l' , Vij , p , diff_x_2 ) # for diagonal matrices
        # V = (2 / len(Itar[ind_tar])) * np.sum(Vij * p[..., None] * diff_x_2, axis=(0, 1)) # MODIFIED -> TO CHECK
        grad = transform * V
        
    else:
        V = (2/len(Itar[ind_tar])) * np.einsum( 'ij,ij,ijkl->kl' , Vij , p , diff_x_2 ) # for non-diagonal matrices
        # V = (2 / len(Itar[ind_tar])) * np.sum(Vij[..., None, None] * p[..., None, None] * diff_x_2, axis=(0, 1)) # MODIFIED -> TO CHECK
        grad = np.matmul(transform, V)
        
    return grad


def compute_gradient_CRPS(dataset_x, Itar, Icat, ind_tar, ind_cat, k, mae, mad, diff_y, diff_ana_y, p, transform):
    # TO COMMENT

    is_diag = [None,True,False][len(transform.shape)]
        
    # diff_x = (np.repeat(dataset_x[Itar[ind_tar], np.newaxis, :], k, axis=1)
    #           - dataset_x[Icat][ind_cat])
    diff_x = dataset_x[Itar[ind_tar], None, :] - dataset_x[Icat][ind_cat] # MODIFIED -> IT WORKS

    if is_diag:
        diff_x_2 = diff_x**2 # for diagonal matrices
    else:
        diff_x_2 = np.einsum( 'ijk,ijl->ijkl', diff_x, diff_x) # for non-diagonal matrices
        # diff_x_2 = diff_x[..., None] * diff_x[:, :, None, :] # MODIFIED -> TO CHECK ON LORENZ

    Vij = np.repeat( (mae - mad)[:, np.newaxis], k, axis=1 )
    # Vij = (mae - mad)[:, None] # MODIFIED -> DOESN'T WORK -> IS IT EVEN USEFUL ?

    # Vij -= np.einsum('ijm->ij',
    #                  ( diff_y - np.einsum( 'ik,ikjm->ijm' , p , diff_ana_y ) ) )
    Vij -= np.sum( diff_y - np.sum( p[..., None, None] * diff_ana_y , axis=1 ) , axis=2 ) # MODIFIED -> IT WORKS
    # Vij -= np.sum(diff_y - np.einsum('ik,ikjm->ijm', p, diff_ana_y), axis=-1) # MODIFIED -> TO CHECK -> NOT WORKING
    # Vij -= np.sum(diff_y - np.einsum( 'ik,ikjm->ijm' , p , diff_ana_y ), axis=-1) # MODIFIED -> TO CHECK 
    # Vij -= np.sum(diff_y - np.sum(p[:, None, None] * diff_ana_y, axis=1), axis=-1) # MODIFIED -> TO CHECK -> NOT WORKING


    if is_diag:
        # V = (1/len(Itar[ind_tar])) * np.einsum( 'ij,ij,ijl->l' , p , Vij , diff_x_2 ) # for diagonal matrices
        V = (1 / len(Itar[ind_tar])) * np.sum( (Vij * p)[..., None] * diff_x_2, axis=(0, 1)) # MODIFIED -> IT WORKS
        grad = transform * V
        
    else:
        V = (1/len(Itar[ind_tar])) * np.einsum( 'ij,ij,ijkl->kl' , p , Vij , diff_x_2 )
        # V = (1 / len(Itar[ind_tar])) * np.sum(Vij[..., None, None] * p[..., None, None] * diff_x_2, axis=(0, 1)) # MODIFIED -> TO CHECK ON LORENZ
        grad = np.matmul(transform, V)
        
    return grad


def compute_regularization(transform, regul_coef=[0], regul_type='l1/l2'):
    # TOCOMMENT
    
    if len(np.array(regul_coef)) == 1: # if one regularization coefficients is given, apply one of the following regularizations
        
        if regul_coef != 0:
            
            if regul_type == 'l1/l2':
                norm1_ = np.sum(np.abs(transform))
                norm2_ = np.sqrt(np.sum(transform**2))
                regul_nocoef = np.sign(transform)/norm2_ - (norm1_/norm2_**3)*transform
                
            elif regul_type == 'l1':
                regul_nocoef = np.sign(transform)
                
            elif regul_type == 'l2':
                regul_nocoef = transform
                
            regul_term = regul_coef[0] * regul_nocoef

        else:
        
            regul_term = 0
            
    elif len(np.array(regul_coef)) == 2: # if two regularization coefficients are given, apply elastic-net regularization
        
        if np.abs(regul_coef[0]) + np.abs(regul_coef[1]) != 0:
            regul_term = regul_coef[0]*np.sign(transform) + regul_coef[1]*transform

        else:

            regul_term = 0
            
    return regul_term


#################################################################
### Final routine to apply gradient-descent distance learning ###
#################################################################


def learn_distance(dataset_x, dataset_y, transform, Icat, Itar, Itest, 
                   k = 200, nn_algo='auto', error_type='CRPS',
                    learning_rate = 0.1, regul_coef = [0.01], regul_type = 'l1/l2',
                   loo=False, corr_length_train=0,
                     n_epoch = None, n_iter = None, batch_size = 100, compute_error_train = True,
                         n_jobs=1, verbose_batch = True, verbose_epoch = True):
    # TO COMMENT

    # Initialize empty lists
    ERROR_batch = []
    ERROR_train = []
    ERROR_test = []
    Epoch = []
    transforms_ = []

    # Initialize transform matrix
    transform_ = transform.copy()

    # Determine if transformation is diagonal
    is_diag = [None,True,False][len(transform.shape)]

    # Ensure only one of n_epoch or n_iter is provided
    if (n_epoch is not None) and (n_iter is not None):
        raise ValueError("Provide only one of n_epoch or n_iter, not both.")

    # If n_iter is given, compute the minimum required n_epoch
    if n_iter is not None:
        n_epoch = max(1, int(np.ceil(n_iter * batch_size / len(Itar))))
    
    # Initialize iteration counter
    iteration_count = 0

    # Checks for the provided regularization coefficient and regularization type
    if isinstance(regul_coef, (float, int)):  # Allow single float or int
        regul_coef = [regul_coef]
    elif isinstance(regul_coef, Sequence) and all(isinstance(x, (float, int)) for x in regul_coef):
        # If one regularization coeffcient is given, regularization type must be either "l1" or "l2" or "l1/l2"
        if len(regul_coef) == 1:
            if not regul_type in {'l1', 'l2', 'l1/l2'}:
                print('ERROR: One regularization coefficient imposes regularization type to be either "l1" or "l2" or "l1/l2".')
                exit(1)
        # If two regularization coeffcients are given, set regularization type to "elastic-net"
        if len(regul_coef) == 2:
            regul_type = 'elastic-net'
    else:
        print('ERROR: The regularization coefficient is not valid. Provied either a float, int, or a two-element array-like object.')  # Not a valid input
        exit(1)

    # If two regularization coeffcients are given, set regularization type to "elastic-net"
    if len(regul_coef) == 2:
        regul_type = 'elastic-net'

    # If "catalogue" and "target" sets are equal, then impose leave-one-out with loo=True
    train_equals_target = False # initialize boolean variable
    if len(Icat) == len(Itar):
        if (Icat==Itar).sum() == len(Icat):
            train_equals_target = True
            if loo==False:
                print('/!\ If catalogue and target sets are equal, leave-one-out condition must be set to "loo=True"')
                loo=True

    
    print('Starting distance-learning algorithm with the following parameters:')
    print('Error type = ' + error_type)
    print('Transformation type = ' + is_diag * 'diagonal matrix (weighting of coordinates)'
          + np.invert(is_diag) * 'matrix (general linear transformation)')
    print('Number of analogues = ' + str(k))
    if train_equals_target:
        print('Catalogue and target sets are equal (leave-one-out procedure'
              +(corr_length_train>0)*(' & leaving time-correlated analogues below '
                                      +str(corr_length_train)+' time-increments')+').')
    print('Learning rate = '+str(learning_rate))
    print('Number of Epochs = '+str(n_epoch))
    print('Mini-batch size = '+str(batch_size))
    print('Regularization = ' + (np.sum(np.abs(regul_coef))>0) * ( regul_type + ', ' ) + str(regul_coef) )
    print('Errors computed = mini-batch'+ compute_error_train*', train' + (len(Itest)>0)*', test' + '.')
    print('Errors not computed = ' + (1-compute_error_train)*'train ,' + (len(Itest)==0)*'test ,')
    
    for epoch in tqdm(range(n_epoch)):
        
        rgn_perm_epoch = np.random.default_rng( 13121312 + 1312 * ( epoch + 1312 ) )
        permutation = rgn_perm_epoch.permutation( np.arange(len(Itar)) )
        
        for i in range(0, len(Itar), batch_size):
            
            # Increment iteration count
            iteration_count += 1
            if n_iter is not None and iteration_count > n_iter:
                break  # Stop once we reach n_iter
                
            # Store transformation from last iteration
            transforms_.append(transform_.copy())
            
            # Select indices of i-th mini-batch
            # if loo:
            #     # In case of leave-one-out procedure, do not perform permutation ##-> change !!
            #     ind_batch = range(len(Itar))
            # else:
            ind_batch = permutation[i:min(i+batch_size, len(Itar))]
            
            # transform training explanatory variables (transform whole catalog and target sets)
            cat_X , tar_X = apply_transform(dataset_x, transform_, Icat, Itar)
            
            # Initiate nearest neighbor algorithm (anticipate a need for more than k "analogs" due to leave-one-out procedure and time-correlated data)
            nn = NearestNeighbors( algorithm = nn_algo , n_neighbors = k + int(loo==True) + 2*corr_length_train , n_jobs=n_jobs )
            nn.fit(cat_X)

            # find analogues and related distances
            dist, ind = find_analogues(nn, tar_X[ind_batch], k, loo=loo, corr_length_train=corr_length_train, ind_tar=ind_batch)        

            # Compute weights with updated distances
            p = compute_weights(dist)

            ## Compute errors

            # Error on mini-batch set
            if error_type=='CRPS':
                # Compute differences
                diff_y, diff_ana_y = compute_diffs(dataset_y, Itar, Icat, ind_batch, ind, k)
    
                # Compute MAE and MAD
                mae, mad = compute_mae_mad(p, diff_y, diff_ana_y)

                # Compute CRPS on mini-batch set
                error_batch = np.mean( mae - .5 * mad )

            elif error_type=='MSE':
                # Compute average analogue estimate
                y_av = np.sum((dataset_y[Icat][ind].T*p.T).T, axis=1) 
                # y_av = (p[..., None] * dataset_y[Icat][ind]).sum(axis=1) # MODIFIED -> CHECK OK -> IS IT EVEN USEFUL ?
                
                # Compute MSE on mini-batch set
                error_batch = np.mean( np.sum( ( y_av - dataset_y[Itar][ind_batch] )**2 , axis=1 ) )
                # error_batch = np.mean((y_av - dataset_y[Itar][ind_batch]) ** 2) # MODIFIED -> CHECK OK -> IS IT EVEN USEFUL ?
                
            # Compute error on whole train set if necessary and if not equal to target set
            if train_equals_target and batch_size==len(Itar):
                error_train = error_batch.copy()
            elif compute_error_train:
                error_train = compute_error(tar_X, dataset_y, Icat, Itar, k, nn, loo, corr_length_train, False, error_type)     
            else:
                error_train = np.array(np.nan)

            # Compute error on independent test set if provided (no LOO procedure)
            if len(Itest)>0:
                test_X = apply_transform( dataset_x , transform_ , Itest )    
                error_test = compute_error(test_X, dataset_y, Icat, Itest, k, nn, loo=False, corr_length_train=0, error_type=error_type)
            else:
                error_test = np.array(np.nan)
            
            # Store errors
            ERROR_batch.append( error_batch.copy() )
            ERROR_test.append( error_test.copy() )
            ERROR_train.append( error_train.copy() )

            # Print errors and save epoch number
            if verbose_batch:
                print( 'epoch '+str(epoch+1)+'/' + str(n_epoch) + 
                      '   |   iter. '+str(i//batch_size+1)+'/' + str(len(range(0, len(Itar), batch_size)))
                     +' : ')
                print( error_type + '(mini-batch) = ' + "{:.2E}".format( ERROR_batch[-1] ) +
                      ' ;  ' + error_type + '(train) = ' + "{:.2E}".format( ERROR_train[-1] ) +
                      ' ;  ' + error_type + '(test) = ' + "{:.2E}".format( ERROR_test[-1] ) )
            Epoch.append(epoch+1)

            # Compute analytical gradient
            if error_type=='CRPS':
                grad = compute_gradient_CRPS(dataset_x, Itar, Icat, ind_batch, ind, k, mae, mad, diff_y, diff_ana_y, p, transform_)

            elif error_type=='MSE':
                grad = compute_gradient_MSE(dataset_x, dataset_y, Itar, Icat, ind_batch, ind, k, y_av, p, transform_)
            
            # Add regularization term
            grad += compute_regularization(transform_, regul_coef, regul_type)                

            # Update the transform_matrix using gradient descent
            transform_ -= learning_rate * grad
            
        # Print epoch
        
        if verbose_epoch and not verbose_batch:
            print( 'epoch. '+str(epoch+1)+'/' + str(n_epoch) + ' :')
            print(error_type+'(train) = ' + "{:.2E}".format( ERROR_train[-1] ) +
                  ' ;  '+error_type+'(test) = ' + "{:.2E}".format( ERROR_test[-1] ) )
    
        # Stop if iteration limit is reached
        if n_iter is not None and iteration_count >= n_iter:
            break
    
    ## Compute and store last errors and transform matrix ##

    # Apply transformation
    cat_X , tar_X = apply_transform(dataset_x, transform_, Icat, Itar)

    # Initiate nearest neighbor algorithm
    nn = NearestNeighbors( algorithm = nn_algo , 
                n_neighbors = k + int(loo==True) + 2*corr_length_train, n_jobs=n_jobs ) # leave-one-out procedure + anticipating time-correlated data
    nn.fit(cat_X)

    # Compute and store error on whole training set
    if compute_error_train:
        ERROR_train.append( compute_error(tar_X, dataset_y, Icat, Itar, k, nn, loo, corr_length_train, error_type=error_type) )
    else :
        ERROR_train.append( np.array(np.nan) )

    # Compute and store CRPS on independent test set (no LOO procedure)
    if len(Itest)>0:
        test_X = apply_transform( dataset_x , transform_ , Itest )    
        error_test = compute_error(test_X, dataset_y, Icat, Itest, k, nn, loo=False, corr_length_train=0, error_type=error_type)
    else:
        error_test = np.array([np.nan])

    ERROR_test.append( error_test.copy() )
    
    # Print errors and save epoch number
    if verbose_epoch:
        print( 'epoch. '+str(epoch+1)+'/' + str(n_epoch) + ' :')
        print( error_type + '(train) = ' + "{:.2E}".format( ERROR_train[-1] ) +
              ' ;  ' + error_type + '(test) = ' + "{:.2E}".format( ERROR_test[-1] ) )
    Epoch.append(epoch+1)
        
    return ( np.array(transforms_), np.array(ERROR_batch),
             np.array(ERROR_train), np.array(ERROR_test), np.array(Epoch) )
