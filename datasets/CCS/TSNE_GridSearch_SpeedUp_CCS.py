#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

#     catsne.py

# This code implements cat-SNE, a class-aware version of t-SNE, as well as quality assessment criteria for both supervised and unsupervised dimensionality reduction.
# Cat-SNE was presented at the ESANN 2019 conference.

# Please cite as:
# - de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
# - BibTeX entry:
#@inproceedings{cdb2019catsne,
#  title={Class-aware {t-SNE}: {cat-SNE}},
#  author={de Bodt, C. and Mulders, D. and L\'opez-S\'anchez, D. and Verleysen, M. and Lee, J. A.},
#  booktitle={ESANN},
#  pages={409--414},
#  year={2019}
#}

# The most important functions of this file are:
# - catsne: enables applying cat-SNE to reduce the dimension of a data set. The documentation of the function describes its parameters.
# - eval_dr_quality: enables evaluating the quality of an embedding in an unsupervised way. It computes quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them.
# - knngain: enables evaluating the quality of an embedding in a supervised way. It computes criteria related to the accuracy of a KNN classifier in the low-dimensional space. The documentation of the function explains the meaning of the criteria and how to interpret them.
# - viz_qa: a plot function to easily visualize the quality criteria.
# At the end of the file, a demo presents how the code and the above functions can be used. Running this code will run the demo. Importing this module will not run the demo.

# Notations:
# - DR: dimensionality reduction
# - HD: high-dimensional
# - LD: low-dimensional
# - HDS: HD space
# - LDS: LD space

# References:
# [1] de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414).
# [2] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
# [3] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
# [4] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
# [5] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# [6] Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
# [7] Jacobs, R. A. (1988). Increased rates of convergence through learning rate adaptation. Neural networks, 1(4), 295-307.

# author: Cyril de Bodt (ICTEAM - UCLouvain)
# @email: cyril __dot__ debodt __at__ uclouvain.be
# Last modification date: May 15th, 2019
# Copyright (c) 2019 Universite catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

# This code was created and tested with Python 3.7.3 (Anaconda distribution, Continuum Analytics, Inc.). It uses the following modules:
# - numpy: version 1.16.3 tested
# - numba: version 0.43.1 tested
# - scipy: version 1.2.1 tested
# - matplotlib: version 3.0.3 tested
# - scikit-learn: version 0.20.3 tested

# You can use, modify and redistribute this software freely, but not for commercial purposes.
# The use of the software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

########################################################################################################
########################################################################################################
import numpy as np,pandas as pd, numba, sklearn.decomposition, scipy.spatial.distance, matplotlib.pyplot as plt, matplotlib, sklearn.datasets
from sklearn.manifold import TSNE
import seaborn as sns
from pandas import read_csv
import umap
import time
import LargeVis
import argparse

try:
    from collections import defaultdict, Mapping, namedtuple
except ImportError:
    from collections.abc import defaultdict, Mapping, namedtuple

#from collections import Mapping, defaultdict
#from collections.abc import Mapping
#from collections import defaultdict
#from Mapping import defaultdict

##############################
##############################
# General functions used by others in the code.
##############################

parser = argparse.ArgumentParser()
parser.add_argument('-fea', default = 1, type = int, help = 'whether to visualize high-dimensional feature vectors or networks')
parser.add_argument('-input', default = '', help = 'input file')
parser.add_argument('-output', default = '', help = 'output file')
parser.add_argument('-outdim', default = -1, type = int, help = 'output dimensionality')
parser.add_argument('-threads', default = -1, type = int, help = 'number of training threads') ## Numer of parallel threads for largevis algorithm to use. Defaults to output of mp.cpu_count()
parser.add_argument('-samples', default = -1, type = int, help = 'number of training mini-batches') ## Method for randomly downsampling points. If `random` is used, `sample_size` total points are sampled. if `by_year`, randomly sample `sample_size` points *from each year* are sampled. Defaults to using all points in the input array.
parser.add_argument('-prop', default = -1, type = int, help = 'number of propagations. Default 3')
parser.add_argument('-alpha', default = -1, type = float, help = 'learning rate. Default 1.0')
parser.add_argument('-trees', default = -1, type = int, help = 'number of rp-trees. Default is set according to the data size.')
parser.add_argument('-neg', default = -1, type = int, help = 'number of negative samples. Default 5')
parser.add_argument('-neigh', default = -1, type = int, help = 'number of neighbors in the NN-graph. Default 150')
parser.add_argument('-gamma', default = -1, type = float, help = 'weight assigned to negative edges')
parser.add_argument('-perp', default = -1, type = float, help = 'perplexity for the NN-grapn. Default 50')

args = parser.parse_args()

@numba.jit(nopython=True)
def close_to_zero(v):
    """
    Check whether v is close to zero or not.
    In:
    - v: a scalar or numpy array.
    Out:
    A boolean or numpy array of boolean of the same shape as v, with True when the entry is close to 0 and False otherwise.
    """
    return np.absolute(v) <= 10.0**(-8.0)

@numba.jit(nopython=True)
def arange_except_i(N, i):
    """
    Create a 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    In:
    - N: a strictly positive integer.
    - i: a positive integer which is strictly smaller than N.
    Out:
    A 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    """
    arr = np.arange(N)
    return np.hstack((arr[:i], arr[i+1:]))

@numba.jit(nopython=True)
def fill_diago(M, v):
    """
    Replace the elements on the diagonal of a square matrix M with some value v.
    In:
    - M: a 2-D numpy array storing a square matrix.
    - v: some value.
    Out:
    M, but in which the diagonal elements have been replaced with v.
    """
    for i in range(M.shape[0]):
        M[i,i] = v
    return M

@numba.jit(nopython=True)
def compute_gradn(grad):
    """
    Compute the norm of a gradient.
    In:
    - grad: numpy array of float storing a gradient.
    Out:
    Infinite norm of the gradient.
    """
    return np.absolute(grad).max()

@numba.jit(nopython=True)
def compute_rel_obj_diff(prev_obj, obj, n_eps):
    """
    Compute the relative objective function difference between two steps in a gradient descent.
    In:
    - prev_obj: objective function value at previous iteration.
    - obj: current objective function value.
    - n_eps: a small float that should be equal to np.finfo(dtype=np.float64).eps.
    Out:
    np.abs(prev_obj - obj)/max(np.abs(prev_obj), np.abs(obj))
    """
    return np.abs(prev_obj - obj)/np.maximum(n_eps, max(np.abs(prev_obj), np.abs(obj)))

##############################
##############################
# Cat-SNE [1].
# The main function which should be used is 'catsne'.
# See its documentation for details.
# The demo at the end of this file presents how to use catsne function.
##############################

@numba.jit(nopython=True)
def sne_sim(dsi, vi, i, compute_log=True):
    """
    Compute the SNE asymmetric similarities, as well as their log.
    N refers to the number of data points.
    In:
    - dsi: numpy 1-D array of floats with N squared distances with respect to data point i. Element k is the squared distance between data points k and i.
    - vi: bandwidth of the exponentials in the similarities with respect to i.
    - i: index of the data point with respect to which the similarities are computed, between 0 and N-1.
    - compute_log: boolean. If True, the logarithms of the similarities are also computed, and otherwise not.
    Out:
    A tuple with two elements:
    - A 1-D numpy array of floats with N elements. Element k is the SNE similarity between data points i and k.
    - If compute_log is True, a 1-D numpy array of floats with N element. Element k is the log of the SNE similarity between data points i and k. By convention, element i is set to 0. If compute_log is False, it is set to np.empty(shape=N, dtype=np.float64).
    """
    N = dsi.size
    si = np.empty(shape=N, dtype=np.float64)
    si[i] = 0.0
    log_si = np.empty(shape=N, dtype=np.float64)
    indj = arange_except_i(N=N, i=i)
    dsij = dsi[indj]
    log_num_sij = (dsij.min()-dsij)/vi
    si[indj] = np.exp(log_num_sij)
    den_si = si.sum()
    si /= den_si
    if compute_log:
        log_si[i] = 0.0
        log_si[indj] = log_num_sij - np.log(den_si)
    return si, log_si

@numba.jit(nopython=True)
def sne_bsf(dsi, vi, i, log_perp):
    """
    Function on which a binary search is performed to find the HD bandwidth of the i^th data point in SNE.
    In:
    - dsi, vi, i: same as in sne_sim function.
    - log_perp: logarithm of the targeted perplexity.
    Out:
    A float corresponding to the current value of the entropy of the similarities with respect to i, minus log_perp.
    """
    si, log_si = sne_sim(dsi=dsi, vi=vi, i=i, compute_log=True)
    return -np.dot(si, log_si) - log_perp

@numba.jit(nopython=True)
def sne_bs(dsi, i, log_perp, x0=1.0):
    """
    Binary search to find the root of sne_bsf over vi.
    In:
    - dsi, i, log_perp: same as in sne_bsf function.
    - x0: starting point for the binary search. Must be strictly positive.
    Out:
    A strictly positive float vi such that sne_bsf(dsi, vi, i, log_perp) is close to zero.
    """
    fx0 = sne_bsf(dsi=dsi, vi=x0, i=i, log_perp=log_perp)
    if close_to_zero(v=fx0):
        return x0
    elif not np.isfinite(fx0):
        raise ValueError("Error in function sne_bs: fx0 is nan.")
    elif fx0 > 0:
        x_up, x_low = x0, x0/2.0
        fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_low):
            return x_low
        elif not np.isfinite(fx_low):
            # WARNING: can not find a valid root!
            return x_up
        while fx_low > 0:
            x_up, x_low = x_low, x_low/2.0
            fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_low):
                return x_low
            if not np.isfinite(fx_low):
                return x_up
    else:
        x_up, x_low = x0*2.0, x0
        fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_up):
            return x_up
        elif not np.isfinite(fx_up):
            return x_low
        while fx_up < 0:
            x_up, x_low = 2.0*x_up, x_up
            fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_up):
                return x_up
    while True:
        x = (x_up+x_low)/2.0
        fx = sne_bsf(dsi=dsi, vi=x, i=i, log_perp=log_perp)
        if close_to_zero(v=fx):
            return x
        elif fx > 0:
            x_up = x
        else:
            x_low = x

@numba.jit(nopython=True)
def catsne_hd_sim(ds_hd, labels, theta, n_eps):
    """
    Compute the symmetrized HD similarities of cat-SNE, as defined in [1].
    In:
    - ds_hd: 2-D numpy array of floats with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between data points i and j.
    - labels, theta: see catsne function.
    - n_eps: should be equal to np.finfo(dtype=np.float64).eps.
    Out:
    A tuple with:
    - A 2-D numpy array of floats with shape (N, N) and in which element [i,j] is the symmetrized HD similarity between data points i and j, as defined in [1].
    - A 1-D numpy array of floats with N elements. Element i indicates the probability mass associated to data points with the same class as i in the HD Gaussian neighborhood around i.
    """
    # Number of data points
    N = ds_hd.shape[0]
    # Computing the N**2 HD similarities
    sigma_ij = np.empty(shape=(N,N), dtype=np.float64)
    L = int(round(np.log2(np.float64(N)/2.0)))
    log_perp = np.log(2.0**(np.linspace(L, 1, L).astype(np.float64)))
    max_ti = np.empty(shape=N, dtype=np.float64)
    for i in range(N):
        vi = 1.0
        h = 0
        go = True
        max_ti[i] = -1.0
        labi = labels == labels[i]
        labi[i] = False
        while go and (h < L):
            vi = sne_bs(dsi=ds_hd[i,:], i=i, log_perp=log_perp[h], x0=vi)
            si = sne_sim(dsi=ds_hd[i,:], vi=vi, i=i, compute_log=False)[0]
            h += 1
            ssi = np.sum(si[labi])
            if ssi > max_ti[i]:
                max_ti[i] = ssi
                sigma_ij[i,:] = si
                if max_ti[i] > theta:
                    go = False
    # Symmetrized version
    sigma_ij += sigma_ij.T
    # Returning the normalization of sigma_ij, and max_ti.
    return sigma_ij/np.maximum(n_eps, sigma_ij.sum()), max_ti

@numba.jit(nopython=True)
def catsne_ld_sim(ds_ld, n_eps):
    """
    Compute the LD similarities of cat-SNE, as well as their log, as defined in [1].
    In:
    - ds_ld: 2-D numpy array of floats with shape (N, N), where N is the number of data points. Element [i,j] must be the squared LD distance between data points i and j.
    - n_eps: same as in catsne_g function.
    Out:
    A tuple with three elements:
    - A 2-D numpy array of floats with shape (N, N) and in which element [i,j] is the LD similarity between data points i and j.
    - A 2-D numpy array of floats with shape (N, N) and in which element [i,j] is the log of the LD similarity between data points i and j. By convention, the log of 0 is set to 0.
    - 1.0/(1.0+ds_ld)
    """
    ds_ldp = 1.0+ds_ld
    idsld = 1.0/np.maximum(n_eps, ds_ldp)
    s_ijt = idsld.copy()
    log_s_ijt = -np.log(ds_ldp)
    s_ijt = fill_diago(M=s_ijt, v=0.0)
    log_s_ijt = fill_diago(M=log_s_ijt, v=0.0)
    den_s_ijt = s_ijt.sum()
    s_ijt /= np.maximum(n_eps, den_s_ijt)
    log_s_ijt -= np.log(den_s_ijt)
    return s_ijt, log_s_ijt, idsld

@numba.jit(nopython=True)
def catsne_obj(sigma_ijt, log_s_ijt):
    """
    Compute the cat-SNE objective function.
    In:
    - sigma_ijt: 2-D numpy array of floats, in which element [i,j] contains the HD similarity between data points i and j, as defined in [1].
    - log_s_ijt: 2-D numpy array of floats, in which element [i,j] contains the log of the LD similarity between data points i and j, as defined in [1].
    Out:
    The value of the cat-SNE objective function.
    """
    return -(sigma_ijt.ravel()).dot(log_s_ijt.ravel())

def catsne_g(X_lds, sigma_ijt, nit, eei, eef, n_eps):
    """
    Compute the gradient of the objective function of cat-SNE at some LD coordinates, as well as the current value of the objective function.
    In:
    - X_lds: 2-D numpy array of floats with N rows, where N is the number of data points. It contains one example per row and one feature per column. It stores the current LD coordinates.
    - sigma_ijt: 2-D numpy array of floats with shape (N, N), where element [i,j] contains the HD similarity between data points i and j.
    - nit: number of gradient descent steps which have already been performed.
    - eei: number of gradient steps to perform with early exageration.
    - eef: early exageration factor.
    - n_eps: a small float to avoid making divisions with a denominator close to 0.
    Out:
    A tuple with two elements:
    - grad: a 2-D numpy array of floats with the same shape as X_lds, containing the gradient at X_lds.
    - obj: objective function value at X_lds.
    """
    # Computing the LD similarities.
    s_ijt, log_s_ijt, idsld = catsne_ld_sim(ds_ld=scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_lds, metric='sqeuclidean'), force='tomatrix'), n_eps=n_eps)

    # Computing the current objective function value
    if nit < eei:
        obj = catsne_obj(sigma_ijt=sigma_ijt/eef, log_s_ijt=log_s_ijt)
    else:
        obj = catsne_obj(sigma_ijt=sigma_ijt, log_s_ijt=log_s_ijt)
    # Computing the gradient.
    c_ij = 4*(sigma_ijt-s_ijt)*idsld
    grad = (X_lds.T*c_ij.dot(np.ones(shape=X_lds.shape[0]))).T - c_ij.dot(X_lds)
    # Returning
    return grad, obj

def dbd_rule(delta_bar, grad, stepsize, kappa=0.2, phi=0.8, tdb=0.5):
    """
    Delta-bar-delta stepsize adaptation rule in a gradient descent procedure, as proposed in [7].
    In:
    - delta_bar: numpy array which stores the current value of the delta bar.
    - grad: numpy array which stores the value of the gradient at the current coordinates.
    - stepsize: numpy array which stores the current values of the step sizes associated with the variables.
    - kappa: linear stepsize increase when delta_bar and the gradient are of the same sign.
    - phi: exponential stepsize decrease when delta_bar and the gradient are of different signs.
    - tdb: parameter for the update of delta_bar.
    Out:
    A tuple with two elements:
    - A numpy array with the update of delta_bar.
    - A numpy array with the update of stepsize.
    """
    dbdp = np.sign(delta_bar) * np.sign(grad)
    stepsize[dbdp > 0] += kappa
    stepsize[dbdp < 0] *= phi
    delta_bar = (1-tdb) * grad + tdb * delta_bar
    return delta_bar, stepsize

@numba.jit(nopython=True)
def mgd_step(X, up_X, nit, mom_t, mom_init, mom_fin, stepsize, grad):
    """
    Momentum gradient descent step.
    In:
    - X: numpy array containing the current value of the variables.
    - up_X: numpy array with the same shape as X storing the update made on the variables at the previous gradient step.
    - nit: number of gradient descent iterations which have already been performed.
    - mom_t: number of gradient descent steps to perform before changing the momentum coefficient.
    - mom_init: momentum coefficient to use when nit<mom_t.
    - mom_fin: momentum coefficient to use when nit>=mom_t.
    - stepsize: step size to use in the gradient descent. Either a scalar, or a numpy array with the same shape as X.
    - grad: numpy array with the same shape as X, storing the gradient of the objective function at the current coordinates.
    Out:
    A tuple with two elements:
    - A numpy array with the updated coordinates, after having performed the momentum gradient descent step.
    - A numpy array storing the update performed on the variables.
    """
    if nit < mom_t:
        mom = mom_init
    else:
        mom = mom_fin
    up_X = mom * up_X - (1-mom) * stepsize * grad
    X += up_X
    return X, up_X

def catsne_mgd(ds_hd, labels, theta, n_eps, eei, eef, X_lds, ftol, N, dim_lds, mom_t, nit_max, gtol, mom_init, mom_fin):
    """
    Performing momentum gradient descent in cat-SNE.
    In:
    - ds_hd: 2-D numpy array of floats with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between data points i and j.
    - labels, theta, eei, eef, ftol, dim_lds, mom_t, nit_max, gtol, mom_init, mom_fin: as in catsne function.
    - n_eps: a small float to avoid making divisions with a denominator close to 0.
    - X_lds: 2-D numpy array of floats with N rows. It contains one example per row and one feature per column. It stores the initial LD coordinates.
    - N: number of data points.
    Out:
    A tuple with:
    - a 2-D numpy array of floats with shape (N, dim_lds), containing the LD representations of the data points in its rows.
    - a 1-D numpy array of floats with N elements. Element at index i indicates the probability mass around X_hds[i,:] which lies on neighbors of the same class.
    """
    # Computing the HD similarities.
    sigma_ijt, max_ti = catsne_hd_sim(ds_hd=ds_hd, labels=labels, theta=theta, n_eps=n_eps)
    # Current number of gradient descent iterations.
    nit = 0
    # Early exageration
    if eei > nit:
        sigma_ijt *= eef
    # Computing the current gradient and objective function values.
    grad, obj = catsne_g(X_lds=X_lds, sigma_ijt=sigma_ijt, nit=nit, eei=eei, eef=eef, n_eps=n_eps)
    gradn = compute_gradn(grad=grad)
    # LD coordinates achieving the smallest value of the objective function.
    best_X_lds = X_lds.copy()
    # Smallest value of the objective function.
    best_obj = obj
    # Objective function value at previous iteration.
    prev_obj = (1+100*ftol)*obj
    rel_obj_diff = compute_rel_obj_diff(prev_obj=prev_obj, obj=obj, n_eps=n_eps)
    # Step size parameters. The steps are adapted during the gradient descent as in [6], using the Delta-Bar-Delta learning rule from [7].
    epsilon, kappa, phi, tdb = 500, 0.2, 0.8, 0.5
    stepsize, delta_bar = epsilon*np.ones(shape=(N, dim_lds), dtype=np.float64), np.zeros(shape=(N, dim_lds), dtype=np.float64)
    # Update of X_lds
    up_X_lds = np.zeros(shape=(N, dim_lds), dtype=np.float64)
    # Gradient descent.
    while (nit <= eei) or (nit <= mom_t) or ((nit < nit_max) and (gradn > gtol) and (rel_obj_diff > ftol)):
        # Computing the step sizes, following the delta-bar-delta rule, from [7].
        delta_bar, stepsize = dbd_rule(delta_bar=delta_bar, grad=grad, stepsize=stepsize, kappa=kappa, phi=phi, tdb=tdb)
        # Performing the gradient descent step with momentum.
        X_lds, up_X_lds = mgd_step(X=X_lds, up_X=up_X_lds, nit=nit, mom_t=mom_t, mom_init=mom_init, mom_fin=mom_fin, stepsize=stepsize, grad=grad)
        # Centering the result
        X_lds -= X_lds.mean(axis=0)
        # Incrementing the iteration counter
        nit += 1
        # Checking whether early exageration is over
        if nit == eei:
            sigma_ijt /= eef
        # Updating the previous objective function value
        prev_obj = obj
        # Computing the gradient at the current LD coordinates and the current objective function value.
        grad, obj = catsne_g(X_lds=X_lds, sigma_ijt=sigma_ijt, nit=nit, eei=eei, eef=eef, n_eps=n_eps)
        gradn = compute_gradn(grad=grad)
        rel_obj_diff = compute_rel_obj_diff(prev_obj=prev_obj, obj=obj, n_eps=n_eps)
        # Updating best_obj and best_X_lds
        if best_obj > obj:
            best_obj, best_X_lds = obj, X_lds.copy()
    # Returning
    return best_X_lds, max_ti

def catsne(X_hds, labels, theta=0.9, init='ran', dim_lds=2, nit_max=1000, rand_state=None, hd_metric='euclidean', D_hd_metric=None, gtol=10.0**(-5.0), ftol=10.0**(-8.0), eef=4, eei=100, mom_init=0.5, mom_fin=0.8, mom_t=250):
    """
    Apply cat-SNE to reduce the dimensionality of a data set by accounting for class labels.
    Euclidean distance is employed in the LDS, as in t-SNE.
    In:
    - X_hds: 2-D numpy array of floats with shape (N, M), containing the HD data set, with one row per example and one column per dimension. N is hence the number of data points and M the dimension of the HDS. It is assumed that the rows of X_hds are all distinct. If hd_metric is set to 'precomputed', then X_hds must be a 2-D numpy array of floats with shape (N,N) containing the pairwise distances between the data points. This matrix is assumed to be symmetric.
    - labels: 1-D numpy array with N elements, containing integers indicating the class labels of the data points.
    - theta: treshold on the probability mass, around each HD datum, which lies on neighbors with the same class, to fit the precisions of the HD Gaussian neighborhoods. See [1] for further details. This parameter must range in [0.5,1[.
    - init: specify the initialization of the LDS. It is either equal to 'ran', in which case the LD coordinates of the data points are initialized randomly using a Gaussian distribution centered around the origin and with a small variance, or to 'pca', in which case the LD coordinates of the data points are initialized using the PCA projection of the HD samples, or to a 2-D numpy array with N rows, in which case the initial LD coordinates of the data points are specified in the rows of init. In case hd_metric is set to 'precomputed', init can not be set to 'pca'.
    - dim_lds: dimension of the LDS. Must be an integer strictly greater than 0. In case init is a 2-D array, dim_lds must be equal to init.shape[1].
    - nit_max: integer strictly greater than 0 wich specifies the maximum number of gradient descent iterations.
    - rand_state: instance of numpy.random.RandomState. If None, set to numpy.random.
    - hd_metric: metric to compute the HD distances. It must be one of the following:
    --- a string. In this case, it must be one of the following:
    ------ a valid value for the 'metric' parameter of the scipy.spatial.distance.pdist function.
    ------ 'precomputed', in which case X_hds must be a 2-D numpy array of floats with shape (N,N) containing the symmetric pairwise distances between the data points. init must, in this case, be different from 'pca'.
    --- a callable. In this case, it must take two rows of X_hds as parameters and return the distance between the corresponding data points. The distance function is assumed to be symmetric.
    - D_hd_metric: optional dictionary to specify additional arguments to scipy.spatial.distance.pdist, depending on the employed metric.
    - gtol: tolerance on the infinite norm of the gradient during the gradient descent.
    - ftol: tolerance on the relative updates of the objective function during the gradient descent.
    - eef: early exageration factor.
    - eei: number of gradient descent steps to perform with early exageration.
    - mom_init: initial momentum factor value in the gradient descent.
    - mom_fin: final momentum factor value in the gradient descent.
    - mom_t: iteration at which the momentum factor value changes during the gradient descent.
    Out:
    A tuple with:
    - a 2-D numpy array of floats with shape (N, dim_lds), containing the LD representations of the data points in its rows.
    - a 1-D numpy array of floats with N elements. Element at index i indicates the probability mass around X_hds[i,:] which lies on neighbors of the same class.
    """
    # Number of data points
    N = X_hds.shape[0]
    # Checking theta
    if (theta < 0.5) or (theta >= 1):
        raise ValueError("Error in function catsne: theta={theta} whereas it must range in [0.5,1[.".format(theta=theta))
    # Checking rand_state
    if rand_state is None:
        rand_state = np.random
    # Checking init and initializing the LDS
    if isinstance(init, str):
        if init == 'ran':
            X_lds = (10.0**(-4))*rand_state.randn(N, dim_lds)
        elif init == 'pca':
            if isinstance(hd_metric, str) and (hd_metric == "precomputed"):
                raise ValueError("Error in function catsne: init cannot be set to 'pca' when hd_metric is set to 'precomputed'.")
            X_lds = sklearn.decomposition.PCA(n_components=dim_lds, copy=True, random_state=rand_state).fit_transform(X_hds)
        else:
            raise ValueError("Error in function catsne: init={init} whereas it must either be equal to 'ran' or to 'pca'.".format(init=init))
    else:
        # init must be a 2-D numpy array with N rows and dim_lds columns
        if init.ndim != 2:
            raise ValueError("Error in function catsne: init.ndim={v} whereas init must be a 2-D numpy array.".format(v=init.ndim))
        if init.shape[0] != N:
            raise ValueError("Error in function catsne: init.shape[0]={v} whereas it must equal N={N}.".format(v=init.shape[0], N=N))
        if init.shape[1] != dim_lds:
            raise ValueError("Error in function catsne: init.shape[1]={v} whereas it must equal dim_lds={dim_lds}.".format(v=init.shape[1], dim_lds=dim_lds))
        X_lds = init
    # Computing the squared HD distances
    if isinstance(hd_metric, str):
        if hd_metric == "precomputed":
            ds_hd = X_hds**2.0
        else:
            if D_hd_metric is None:
                D_hd_metric = {}
            ds_hd = scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_hds, metric=hd_metric, **D_hd_metric), force='tomatrix')**2.0
    else:
        # hd_metric is a callable
        ds_hd = np.empty(shape=(N,N), dtype=np.float64)
        for i in range(N):
            ds_hd[i,i] = 0.0
            for j in range(i):
                ds_hd[i,j] = hd_metric(X_hds[i,:], X_hds[j,:])**2.0
                ds_hd[j,i] = ds_hd[i,j]
    # Small float
    n_eps = np.finfo(dtype=np.float64).eps
    # Performing momentum gradient descent and returning
    return catsne_mgd(ds_hd=ds_hd, labels=labels, theta=theta, n_eps=n_eps, eei=eei, eef=eef, X_lds=X_lds, ftol=ftol, N=N, dim_lds=dim_lds, mom_t=mom_t, nit_max=nit_max, gtol=gtol, mom_init=mom_init, mom_fin=mom_fin)

##############################
##############################
# Unsupervised DR quality assessment: rank-based criteria measuring the HD neighborhood preservation in the LDS [2, 3].
# The main function which should be used is 'eval_dr_quality'.
# See its documentation for details. It explains the meaning of the quality criteria and how to interpret them.
# The demo at the end of this file presents how to use eval_dr_quality function.
##############################

def coranking(d_hd, d_ld):
    """
    Computation of the co-ranking matrix, as described in [3].
    The time complexity of this function is O(N**2 log(N)), where N is the number of data points.
    In:
    - d_hd: 2-D numpy array representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array representing the redundant matrix of pairwise distances in the LDS.
    Out:
    The (N-1)x(N-1) co-ranking matrix, where N = d_hd.shape[0].
    """
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS.
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')

    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)
    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j,i],j] = i
    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [5].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [5], with a log scale for K=1 to arr.size.
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.float64(arr.dot(i_all_k))/(i_all_k.sum())

@numba.jit(nopython=True)
def eval_rnx(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [4]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding.
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def eval_dr_quality(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2, 3, 4, 5].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS.
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed.
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K, where N refers to the number of data points and \cap to the set intersection operator.
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K.
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1.
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail.
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random.
    In:
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [5].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2).
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)

##############################
##############################
# Supervised DR quality assessment: accuracy of a KNN classifier in the LDS.
# The main function which should be used is 'knngain'.
# See its documentation for details. It explains the meaning of the quality criteria and how to interpret them.
# The demo at the end of this file presents how to use knngain function.
##############################

@numba.jit(nopython=True)
def knngain(d_hd, d_ld, labels):
    """
    Compute the KNN gain curve and its AUC, as defined in [1].
    If c_i refers to the class label of data point i, v_i^K (resp. n_i^K) to the set of the K nearest neighbors of data point i in the HDS (resp. LDS), and N to the number of data points, the KNN gain develops as G_{NN}(K) = (1/N) * \sum_{i=1}^{N} (|{j \in n_i^K such that c_i=c_j}|-|{j \in v_i^K such that c_i=c_j}|)/K.
    It averages the gain (or loss, if negative) of neighbors of the same class around each point, after DR.
    Hence, a positive value correlates with likely improved KNN classification performances.
    As the R_{NX}(K) curve from the unsupervised DR quality assessment, the KNN gain G_{NN}(K) can be displayed with respect to K, with a log scale for K.
    A global score summarizing the resulting curve is provided by its area (AUC).
    In:
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    - labels: 1-D numpy array with N elements, containing integers indicating the class labels of the data points.
    Out:
    A tuple with:
    - a 1-D numpy array of floats with N-1 elements, storing the KNN gain for K=1 to N-1.
    - the AUC of the KNN gain curve, with a log scale for K.
    """
    # Number of data points
    N = d_hd.shape[0]
    N_1 = N-1
    k_hd = np.zeros(shape=N_1, dtype=np.int64)
    k_ld = np.zeros(shape=N_1, dtype=np.int64)
    # For each data point
    for i in range(N):
        c_i = labels[i]
        di_hd = d_hd[i,:].argsort(kind='mergesort')
        di_ld = d_ld[i,:].argsort(kind='mergesort')
        # Making sure that i is first in di_hd and di_ld
        for arr in [di_hd, di_ld]:
            for idj, j in enumerate(arr):
                if j == i:
                    idi = idj
                    break
            if idi != 0:
                arr[idi] = arr[0]
            arr = arr[1:]
        for k in range(N_1):
            if c_i == labels[di_hd[k]]:
                k_hd[k] += 1
            if c_i == labels[di_ld[k]]:
                k_ld[k] += 1
    # Computing the KNN gain
    gn = (k_ld.cumsum() - k_hd.cumsum()).astype(np.float64)/((1.0+np.arange(N_1))*N)
    # Returning the KNN gain and its AUC
    return gn, eval_auc(gn)

##############################
##############################
# Plot functions.
# Their documentations detail their parameters.
# The demo at the end of this file presents how to use these functions.
##############################

def viz_digits(X, lab, tit='', cmap='gnuplot2', stit=30, slab=15, wlab='bold', max_ti=None):
    """
    Visualize a 2-D embedding of digits data set.
    In:
    - X: a numpy array with shape (N, 2), where N is the number of data points in the data set.
    - lab: a 1-D numpy array with N elements indicating the digits.
    - tit: title of the figure.
    - cmap: colormap for the digits.
    - stit: fontsize of the title of the figure.
    - slab: fontsize of the digits.
    - wlab: weight to plot the digits.
    - max_ti: 2nd element in the tuple returned by catsne. It changes the size of the data points
    proportionally to the probability mass of their neighbors with the same class in the HDS.
    If None, it is set to np.ones(shape=N, dtype=np.int64), meaning that all data points have equal size.
    Out:
    A figure is shown.
    """
    # Checking X
    if len(X.shape) != 2:
        raise ValueError("Error in function viz_digits: X must be a numpy array with shape (N, 2), where N is the number of data points.")
    if X.shape[1] != 2:
        raise ValueError("Error in function viz_digits: X must have 2 columns.")

    N = X.shape[0]
    if max_ti is None:
        max_ti = np.ones(shape=N, dtype=np.int64)

    # Computing the limits of the axes
    xmin = X[:,0].min()
    xmax = X[:,0].max()
    expand_value = (xmax-xmin)*0.05
    x_lim = np.asarray([xmin-expand_value, xmax+expand_value])

    ymin = X[:,1].min()
    ymax = X[:,1].max()
    expand_value = (ymax-ymin)*0.05
    y_lim = np.asarray([ymin-expand_value, ymax+expand_value])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Setting the limits of the axes
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Visualizing the digits
    cmap_obj = matplotlib.cm.get_cmap(cmap)
    normalizer = matplotlib.colors.Normalize(vmin=lab.min(), vmax=lab.max())
    for i in range(N):
        ax.text(x=X[i,0], y=X[i,1], s=str(lab[i]), fontsize=slab-10.0*(1.0-max_ti[i]), color=cmap_obj(lab[i]), fontdict={'weight': wlab, 'size': slab-10.0*(1.0-max_ti[i])}, horizontalalignment='center', verticalalignment='center')
## color=cmap_obj(normalizer(lab[i]))
    # Removing the ticks on the x- and y-axes
    ax.set_xticks([], minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([], minor=False)
    ax.set_yticks([], minor=False)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=False)

    ax.set_title(tit, fontsize=stit)
    plt.tight_layout()

    # Showing the figure
    plt.show()
    plt.close()

####  test

def viz_test(X, lab, tit=''):

    #plt.scatter(X_lds[:,0], X_lds[:,1], c=labels, cmap='Set1')
    #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    #plt.title("Umap", fontsize=40)

##test
    plt.scatter(X[:, 0], X[:, 1], c=[sns.color_palette()[x] for x in lab], edgecolor='none',alpha=0.80,
    s=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('LargeVis projection of the Digit dataset', fontsize=24)

    # Showing the figure
    plt.show()
    plt.close()

def open_dataset_ccs():
    # Load dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    names = [
        'id',
        'refractive-index',
        'Sodium',
        'Magnesium',
        'Aluminum',
        'Silicon',
        'Potassium',
        'Calcium',
        'Barium',
        'Iron',
        'class'
    ]
    dataset = pd.read_excel(io = url, sheetname=0, header=0)
    # print(dataset.head())
    return dataset
####  test

def viz_qa(path, name, Ly, ymin=None, ymax=None, Lmarkers=None, Lcols=None, Lleg=None, Lls=None, Lmedw=None, Lsdots=None, lw=2, markevery=0.1, tit='', xlabel='', ylabel='', alpha_plot=0.9, alpha_leg=0.8, stit=25, sax=20, sleg=15, zleg=1, loc_leg='best', ncol_leg=1, lMticks=10, lmticks=5, wMticks=2, wmticks=1, nyMticks=11, mymticks=4, grid=True, grid_ls='solid', grid_col='lightgrey', grid_alpha=0.7, xlog=True):
    """
    Plot the DR quality criteria curves.
    In:
    - Ly: list of 1-D numpy arrays. The i^th array gathers the y-axis values of a curve from x=1 to x=Ly[i].size, with steps of 1.
    - ymin, ymax: minimum and maximum values of the y-axis. If None, ymin (resp. ymax) is set to the smallest (resp. greatest) value among [y.min() for y in Ly] (resp. [y.max() for y in Ly]).
    - Lmarkers: list with the markers for each curve. If None, some pre-defined markers are used.
    - Lcols: list with the colors of the curves. If None, some pre-defined colors are used.
    - Lleg: list of strings, containing the legend entries for each curve. If None, no legend is shown.
    - Lls: list of the linestyles ('solid', 'dashed', ...) of the curves. If None, 'solid' style is employed for all curves.
    - Lmedw: list with the markeredgewidths of the curves. If None, some pre-defined value is employed.
    - Lsdots: list with the sizes of the markers. If None, some pre-defined value is employed.
    - lw: linewidth for all the curves.
    - markevery: approximately 1/markevery markers are displayed for each curve. Set to None to mark every dot.
    - tit: title of the plot.
    - xlabel, ylabel: labels for the x- and y-axes.
    - alpha_plot: alpha for the curves.
    - alpha_leg: alpha for the legend.
    - stit: fontsize for the title.
    - sax: fontsize for the labels of the axes.
    - sleg: fontsize for the legend.
    - zleg: zorder for the legend. Set to 1 to plot the legend behind the data, and to None to keep the default value.
    - loc_leg: location of the legend ('best', 'upper left', ...).
    - ncol_leg: number of columns to use in the legend.
    - lMticks: length of the major ticks on the axes.
    - lmticks: length of the minor ticks on the axes.
    - wMticks: width of the major ticks on the axes.
    - wmticks: width of the minor ticks on the axes.
    - nyMticks: number of major ticks on the y-axis (counting ymin and ymax).
    - mymticks: there are 1+mymticks*(nyMticks-1) minor ticks on the y axis.
    - grid: True to add a grid, False otherwise.
    - grid_ls: linestyle of the grid.
    - grid_col: color of the grid.
    - grid_alpha: alpha of the grid.
    - xlog: True to produce a semilogx plot and False to produce a plot.
    Out:
    A figure is shown.
    """
    # Number of curves
    nc = len(Ly)
    #print("this is nc",nc)
    # Checking the parameters
    if ymin is None:
        ymin = np.min(np.asarray([arr.min() for arr in Ly]))
    if ymax is None:
        ymax = np.max(np.asarray([arr.max() for arr in Ly]))
    if Lmarkers is None:
        Lmarkers = ['x']*nc
    if Lcols is None:
        Lcols = ['blue']*nc
    if Lleg is None:
        Lleg = [None]*nc
        add_leg = False
    else:
        add_leg = True
    if Lls is None:
        Lls = ['solid']*nc
    if Lmedw is None:
        Lmedw = [float(lw)/2.0]*nc
    if Lsdots is None:
        Lsdots = [12]*nc

    # Setting the limits of the y-axis
    y_lim = [ymin, ymax]

    # Defining the ticks on the y-axis
    yMticks = np.linspace(start=ymin, stop=ymax, num=nyMticks, endpoint=True, retstep=False)
    ymticks = np.linspace(start=ymin, stop=ymax, num=1+mymticks*(nyMticks-1), endpoint=True, retstep=False)
    yMticksLab = [int(round(v*100.0))/100.0 for v in yMticks]

    # Initial values for xmin and xmax
    xmin, xmax = 1, -np.inf

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if xlog:
        fplot = ax.semilogx
    else:
        fplot = ax.plot

    # Plotting the data
    for id, y in enumerate(Ly):
        x = np.arange(start=1, step=1, stop=y.size+0.5, dtype=np.int64)
        xmax = max(xmax, x[-1])
        fplot(x, y, label=Lleg[id], alpha=alpha_plot, color=Lcols[id], linestyle=Lls[id], lw=lw, marker=Lmarkers[id], markeredgecolor=Lcols[id], markeredgewidth=Lmedw[id], markersize=Lsdots[id], dash_capstyle='round', solid_capstyle='round', dash_joinstyle='round', solid_joinstyle='round', markerfacecolor=Lcols[id], markevery=markevery)

    # Setting the limits of the axes
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(y_lim)

    # Setting the major and minor ticks on the y-axis
    ax.set_yticks(yMticks, minor=False)
    ax.set_yticks(ymticks, minor=True)
    ax.set_yticklabels(yMticksLab, minor=False, fontsize=sax)

    # Defining the legend
    if add_leg:
        leg = ax.legend(loc=loc_leg, fontsize=sleg, markerfirst=True, fancybox=True, framealpha=alpha_leg, ncol=ncol_leg)
        if zleg is not None:
            leg.set_zorder(zleg)

    # Setting the size of the ticks labels on the x axis
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(sax)

    # Setting ticks length and width
    ax.tick_params(axis='both', length=lMticks, width=wMticks, which='major')
    ax.tick_params(axis='both', length=lmticks, width=wmticks, which='minor')

    # Setting the positions of the labels
    ax.xaxis.set_tick_params(labelright=False, labelleft=True)
    ax.yaxis.set_tick_params(labelright=False, labelleft=True)

    # Adding the grids
    if grid:
        ax.xaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
        ax.yaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
    ax.set_axisbelow(True)

    ax.set_title(tit, fontsize=stit)
    ax.set_xlabel(xlabel, fontsize=sax)
    ax.set_ylabel(ylabel, fontsize=sax)
    plt.tight_layout()

    # Showing the figure
    #plt.show()
    #plt.close()

    save_results_to = path + str(name)
    plt.savefig(save_results_to + '_image.png')

##############################
##############################
# Demo presenting how to use the main functions of this module.
##############################

##换dataset，与kaggle或者其他例子一样，验证labels的顺序是否与highDim data一一对应
if __name__ == '__main__':

    print("========================================================================")
    print("===== Starting the demo of UMAP and of the DR quality criteria. =====")
    print("========================================================================")

############   UMAP test   #####################
    ## prepare the datasets
    print("Loading the test data set.")
    #### load ccs dataset  ###
    dataset = open_dataset_ccs()

    labels =dataset['Concrete compressive strength(MPa, megapascals) ']
    # dataset_glass = dataset.drop('id', 1)
    dataset_glass = dataset.drop('Concrete compressive strength(MPa, megapascals) ', 1)

    # 0724
    # dataset = open_dataset_

    X_hds = dataset_glass.to_numpy()
    labels = labels.to_numpy()

    ### subsample
    ## 0723_test
    # Subsampling the data set, to accelerate the demo.

    try:
        print("the size of dataset", X_hds.shape)
        rand_state = np.random.RandomState(0)
        id_subs = rand_state.choice(a=X_hds.shape[0], size=1000, replace=False)
        X_hds, labels = X_hds[id_subs,:], labels[id_subs]
        print("size after subsampling", X_hds.shape)
    except:
        print("the size of the dataset is less than 1000")
        print("the size is:", X_hds.shape)


    # print("shape of X_hds", X_hds.shape)
    ## 0723_test

    #### load glass dataset  ###
    ## digit dataset
    ## X_hds, labels = sklearn.datasets.load_digits(n_class=10, return_X_y=True)
    # name of the txt form dataset created
    output_txt_file_name = "test_digit_HD.txt"
    #  Use the previously defined variable "output_txt_file_name" as the txt file name here
    np.savetxt(output_txt_file_name, X_hds, delimiter=" ", fmt='%i')
    # Add the LargeVis format header row to the txt file so that the LargeVis algorithm can execute properly
    with open(output_txt_file_name,'w') as out:
        out.write("{} {}\n".format(*X_hds.shape))
        for row in X_hds:
            out.write(' '.join(row.astype(str))+'\n')

    ## load dataset
    ##LargeVis.loadfile("C:/Users/15754/Desktop/LDATE2990 - Master Thesis/LargeVis/LargeVis_Github/LargeVis-master/Code/mnist_vec784D.txt")
    #Y = LargeVis.run(args.outdim, args.threads, args.samples, args.prop, args.alpha, args.trees, args.neg, args.neigh, args.gamma, args.perp)
    #print("******** size of the output **********",np.asarray(Y).shape)
############   largevis test  #####################

    # Metric to compute the HD distances in cat-SNE.
    hd_metric = 'euclidean'
    # Maximum number of iterations for cat-SNE.
    nit_max = 1000
    # Dimension of the LDS.
    dim_lds = 2
    # Random initialization of the LDS.
    init = 'ran'
    # number of threads
    threads_list = [1, 2, 4]
    path = 'C:/Users/15754/Desktop/LDATE2990 - Master Thesis/Project/results/LargeVis/glass/graphs_iteration/0722/'
    # List of the theta thresholds to employ with cat-SNE.
    Ltheta = [0.7, 0.8, 0.9]
    #round = 1
    # Lists to provide as parameters to viz_qa, to visualize the DR quality criteria and the KNN gain.
    L_rnx, L_kg = [], []
    Lmarkers = ['x', 'o', 's']
    Lcols = ['green', 'red', 'blue']
    Lleg_rnx, Lleg_kg = [], []
    list_L_rnx,list_Lleg_rnx = [], []
    Ltime = []
    Lls = []
    Lmedw = [1.5, 1.0, 1.0]
    Lsdots = [14, 12, 12]
    #perplexity_gourp = [13,25,30]
    n_neighbors_group = [10,20,30]
    min_dist_group = [0.1, 0.25, 0.5]
    neigh_group = [100, 150, 200]
    gamma_group = [4,7,10]
    iterations = [1,2,3,4]
    iterations_1 = [1]
    #neg_group = [3,5,7]
    tot_perp_gourp = [15, 50, 70, 100,200]
    tot_neg_group = [3,5,10,20]
    tot_gamma_group = [7]
    #tot_k_group = [[38,75,90],[105,120,135],[165,180,195],[210,225,263]]
    tot_neigh_group = [100,150, 200]
    # tot_k_group_default = [[50,50,50],[50,50,50]]
    outermost_iterations = [1,2,3] # the most outside iteration

    tot_perp_gourp_test = [50]
    tot_neg_group_test = [5]
    tot_gamma_group_test = [7]
    tot_neigh_group_test = [150,200]

    tot_tsne_perp_group = [10, 30, 50, 70, 100,200]
    tot_tsne_min_grad_norm_group = [1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]
    tot_tsne_n_iter_group = [500, 700, 1000, 1500]
    tot_tsne_early_exaggeration_group = [10, 12, 20, 40]

    # tot_tsne_perp_group_test = [30, 50]
    # tot_tsne_min_grad_norm_group_test = [1e-07]
    # tot_tsne_n_iter_group_test = [1000, 1500]
    # tot_tsne_early_exaggeration_group_test = [12]

    tot_umap_n_neighbors_group = [2, 5, 10, 15, 30, 50, 100, 200]# default = 15
    tot_umap_min_dist_group = [0.01, 0.1, 0.25, 0.5, 0.8, 0.99]# default = 0.1
    tot_umap_spread_group = [0.1,0.5,1.0,3.0,5.0,10.0] # default = 1.0
    tot_umap_negative_sample_rate_group = [3,5,10,30] # default = 5

    # tot_umap_n_neighbors_group_test = [2]# default = 15
    # tot_umap_min_dist_group_test = [0.0, 0.1]# default = 0.1
    # tot_umap_spread_group_test = [0.1,0.5] # default = 1.0
    # tot_umap_negative_sample_rate_group_test = [3]

    #

    #print("Loading the test data set.")
    ######   test    #######
    #train_df = read_csv("C:/Users/15754/Desktop/code_master_thesis/dataset/heart.csv", index_col=0)
    #train_df = train_df.fillna(0)
    #X_hds = train_df.values## np.array from now on
    #y =train_df['target']
    #labels = y.values## np.array from now on
    #X_hds=np.delete(X_hds, -1, axis=1)

    #X_hds, labels = sklearn.datasets.load_digits(n_class=10, return_X_y=True)

    ####  before the changing, labels with mean value -0.811733 which means there are around 90% of instances with ablel -1
    ##### replace all the -1 labels with 0 for better vasualization
    #y = y.replace(-1, 0)


    # delete unnecessary columns
    #X_hds=np.delete(X_hds, 0, axis=1)
    ## delete the last column of the dataset


    ######   test    #######
    # Subsampling the data set, to accelerate the demo.
    #rand_state = np.random.RandomState(0)
    #id_subs = rand_state.choice(a=X_hds.shape[0], size=1700, replace=False)
    #X_hds, labels = X_hds[id_subs,:], labels[id_subs]
    # Computing the pairwise HD distances for the quality assessment.
    d_hd = scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_hds, metric=hd_metric), force='tomatrix')



    # for outermost_iteration in outermost_iterations:

        # for iteration in iterations:
    best_auc_score = 0.0
    for perp_value in tot_tsne_perp_group:
        for min_grad_norm_value in tot_tsne_min_grad_norm_group:
            for n_iter_value in tot_tsne_n_iter_group:
                for early_exaggeration_value in tot_tsne_early_exaggeration_group:

                    # 0722
                    # give value for each parameter
                    # round_mark = int(0)
                    # # if iteration == 1:
                    # #     tot_para_group = tot_perplexity_gourp
                    # #     mark = "perplexity"
                    # # elif iteration == 2:
                    # #     tot_para_group = tot_neg_group
                    # #     mark = "neg"
                    # # elif iteration == 3:
                    # #     tot_para_group = tot_gamma_group
                    # #     mark = "gamma"
                    # # else:
                    # #     tot_para_group = tot_k_group
                    # #     mark = "neigh"
                    # for para_group in tot_para_group:
                    #     round_mark += 1
                    #     print("outermost_iteration",outermost_iteration)
                    #     print("parameter_",mark)
                    #     print("round_",round_mark)
                        # For each theta
                        # for para_value in para_group:

                    #n_neighbors = 10
                      ####
                    start_time = time.time()
                    #print("this is mark",mark)
                    print("Applying T-SNE with perp = {perp}, min_grad_norm = {min_grad_norm}, n_iter = {n_iter}, early_exaggeration = {early_exaggeration}".format(perp = perp_value, min_grad_norm = min_grad_norm_value, n_iter = n_iter_value, early_exaggeration = early_exaggeration_value))
                    #####  test  #####
                    #X_test_lds, max_ti = catsne(X_hds=X_hds, labels=labels, theta=theta, init=init, dim_lds=dim_lds, nit_max=nit_max, rand_state=np.random.RandomState(0), hd_metric=hd_metric)
                    #print("**************",labels.shape)
                    #print(type(max_ti))
                    #print(max_ti.shape)

                    ##TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0, random_state=None, method=’barnes_hut’, angle=0.5)
                    #tsne = TSNE(n_components=2, random_state=0, perplexity= perplexity, learning_rate= learning_rate)
                    #embedding = LargeVis.LargeVis(threads = threads)
                    #X_lds = embedding.fit_transform(X_hds)

                    # LargeVis.loadfile(output_txt_file_name)
                    tsne = TSNE(n_components=2, random_state=1, perplexity= perp_value, min_grad_norm = min_grad_norm_value, n_iter = n_iter_value, early_exaggeration = early_exaggeration_value)
                    X_lds = tsne.fit_transform(X_hds)
                    #samples = -1

                    #args.perp = 20
                    #args.neigh = neigh_size
                    #print("***********neigh_size***********",args.neigh)

                    #args.gamma = 7
                    #args.neg = neg_value

                    # 0722
                    # iteration for all values
                    # if iteration == 1:
                    #     args.perp = para_value
                    #     print("perplexity =", para_value)
                    # elif iteration == 2:
                    #     args.neg = para_value
                    #     print("neg = ",para_value)
                    # elif iteration == 3:
                    #     args.gamma = para_value
                    #     print("gamma =", para_value)
                    # else:

                    # args.perp = perp_value
                    # # print("check perp =", perp_value)
                    # args.neg = neg_value
                    # # print("checkneg =", neg_value)
                    # args.gamma = gamma_value
                    # # print("check gamma =", gamma_value)
                    # args.neigh = neigh_value
                    # print("check neigh =", neigh_value)
                        # print("neigh =", para_value)


                    # X_lds = Y = LargeVis.run(2, args.threads, args.samples, args.prop, args.alpha, args.trees, args.neg, args.neigh, args.gamma, args.perp)

                    elapsed_time = time.time() - start_time
                    print("**************Finished in : " + str(elapsed_time) + " seconds****************")

                    print("Computing the DR quality of the result")

                    #####  test  #####

                    # Computing the pairwise distances in the LDS for the quality assessment.
                    d_ld = scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X_lds, metric='euclidean'), force='tomatrix')
                    # transform the dataset from list to array
                    X_lds = np.array(X_lds)


                    # Displaying the LD embedding
                    #viz_test(X=X_lds, lab=labels, tit='LargeVis ($neigh_size={neigh_size}$)'.format(neigh_size=neigh_size))

                    rnxk, auc_rnx = eval_dr_quality(d_hd=d_hd, d_ld=d_ld)
                    auc_value = int(round(auc_rnx*1000))/1000.0
                    print("AUC score", auc_value)
                    print("*")
                    #print("Computing the KNN gain of the result of t-SNE with threshold perplexity = {perplexity}".format(perplexity=perplexity))
                    #kg, auc_kg = knngain(d_hd=d_hd, d_ld=d_ld, labels=labels)
                    # Updating the lists for viz_qa
                    L_rnx.append(rnxk)
                    #L_kg.append(kg)
                    Ltime.append("{running_time} with perp = {perp}, min_grad_norm = {min_grad_norm}, n_iter = {n_iter}, early_exaggeration = {early_exaggeration}".format(running_time = elapsed_time, perp = perp_value, min_grad_norm = min_grad_norm_value, n_iter = n_iter_value, early_exaggeration = early_exaggeration_value))
                    Lleg_rnx.append("AUC_score {AUC_score} with perp = {perp}, min_grad_norm = {min_grad_norm}, n_iter = {n_iter}, early_exaggeration = {early_exaggeration}".format(AUC_score = auc_value, perp = perp_value, min_grad_norm = min_grad_norm_value, n_iter = n_iter_value, early_exaggeration = early_exaggeration_value))
                    #Lleg_kg.append("{a} t-SNE ($n_neighbors= {n_neighbors}$)".format(a=int(round(auc_kg*1000))/1000.0, n_neighbors= n_neighbors))
                    Lls.append('solid')

                    if auc_value > best_auc_score:#找到表现最好的参数
                        best_auc_score = auc_value
                        best_parameters = {'perp':perp_value,'min_grad_norm':min_grad_norm_value, 'n_iter': n_iter_value, 'early_exaggeration':early_exaggeration_value}


                # 0722
                # print the image and save the result value and images automically
                # list_L_rnx = L_rnx[-3:]
                # list_Lleg_rnx = Lleg_rnx[-3:]
                # try:
                #     name = str(mark) + '_iteration_' + str(iteration) + '_round_' + str(round_mark)
                #     #print("name", name)
                #     viz_qa(path=path, name=name, Ly=list_L_rnx, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=list_Lleg_rnx, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='DR quality', xlabel=str(mark) + '_size $K$', ylabel='$R_{NX}(K)$')
                #
                # except:
                #     name = str(mark) + '_iteration_' + str(iteration) + '_time_' + str(start_time)
                #     #print("name", name)
                #     viz_qa(path=path, name=name, Ly=list_L_rnx, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=list_Lleg_rnx, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='DR quality', xlabel=str(mark) + '_size $K$', ylabel='$R_{NX}(K)$')

            #viz_test(X=X_lds, lab=labels, tit='LargeVis ($perplexity={perplexity}$)'.format(perplexity=perplexity))
        #print("Lleg_rnx",Lleg_rnx)
        #print("Rtime", Ltime)
    #print("L_rnx",L_rnx)
    print("final Lleg_rnx",Lleg_rnx)
    print("********************* ")
    print("final Rtime", Ltime)
    print("********************* ")
    print("Best score:{:.3f}".format(best_auc_score))
    print("Best parameters:{}".format(best_parameters))

    #print(Lleg_kg)
    # Displaying the DR quality criteria
    #for i in L_rnx:
        #print("values in L_rnx", i)
    #viz_qa(Ly=L_rnx, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=Lleg_rnx, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='DR quality', xlabel='perplexity size $K$', ylabel='$R_{NX}(K)$')
    # Displaying the KNN gain
    #viz_qa(Ly=L_kg, Lmarkers=Lmarkers, Lcols=Lcols, Lleg=Lleg_kg, Lls=Lls, Lmedw=Lmedw, Lsdots=Lsdots, tit='KNN gain', xlabel='Neighborhood size $K$', ylabel='$G_{NN}(K)$')
