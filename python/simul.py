#!/usr/bin/env python

import os, sys, re
import logging, time, traceback
from time import perf_counter
import argparse
import pandas as pd
import numpy as np
import numpy.matlib as matlib
import random
import math
from functools import reduce

import itertools

import scipy.sparse
from scipy.sparse import linalg, coo_matrix
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.linalg import qr, cho_factor, cho_solve
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.linalg import eig_banded, eigvals_banded
from scipy.optimize import minimize, Bounds, LinearConstraint
from sksparse.cholmod import cholesky
from pandas_plink import read_plink1_bin, read_grm, read_rel

import matplotlib
import matplotlib.pyplot as plt

import faulthandler; faulthandler.enable()

def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

###### FORMATTING FUNCTIONS AND I/O HELPERS #######
def align_annot_l2_ldgm(ldgm_edgelist_fp, ldgm_snplist_fp, annot_fp, ldscore_fp = None, no_empty_annot = False, use_annot = False):
    '''
    Utility function for loading the LDGM precision matrices and reading the annotation matrices. 

    Optional arguments:
    -- ldscore_fp: if this is not None, the LD scores are read. Both .l2.ldscore and .annot files will be updated. 
    -- no_empty_annot: if this is True, the empty annotations will be removed (for ldsc). CAREFUL with this handling when aggregating data from multiple blocks.

    Output:
    -- P_ldgm: an LDGM class object
    -- annot_mat: the annotation matrix
    -- P_index: index of the precision matrix with corresponding annotation values (candidate of causal effects); equivalent to whichIndicesAnnot in the Matlab implementation

    '''

    # Load the precision matrix from files
    P_ldgm = ldgmMatrix(edge_list_path = ldgm_edgelist_fp, snp_list_path = ldgm_snplist_fp)
    p = np.sum(P_ldgm.nz_index)
    logging.info("Number of non-zero SNPs in the precision matrix in this block: {}".format(p))

    if use_annot:
        # Read in the annotation files
        df_annot = pd.read_csv(annot_fp + ".annot", index_col = None, delim_whitespace = True)
        use_annots = list(df_annot.columns.values)[4:] # this includes the baseline

        # Check that the baseline annot indeed has 1 for every SNP
        # logging.info("Column names of the annotation matrix: {}".format(use_annots))
        assert np.sum(df_annot['base']) == df_annot.shape[0], "Baseline annotation is not correct -- not all SNPs have value 1."

        # Merge annot with the ldgm (snplist)
        df_annot_overlap = pd.merge(P_ldgm.snps[['index', 'site_ids', 'anc_alleles', 'deriv_alleles']], df_annot[['CHR', 'BP', 'SNP', 'CM'] + use_annots], left_on = 'site_ids', right_on = 'SNP', how = 'left', indicator = True)
        df_annot_overlap.reset_index(drop = True, inplace = True)

        # logging.info("Unique number of indexes in snplist: {}".format(np.unique(P_ldgm.snps[['index']]).shape))
        # logging.info("SNPs in LDGM with available annot info: {}".format(df_annot_overlap.shape[0]))
        # logging.info(df_annot_overlap[['index', 'site_ids', 'SNP']])
        # logging.info("Maximum value in the index column of the snplist: {}".format(np.max(df_annot_overlap['index'])))
        # logging.info("Columns of the overlapped df: {}".format(df_annot_overlap.columns))

        # Drop duplicated index for unique site_ids
        # df_annot_overlap.drop_duplicates(subset = ['index'], keep = 'first', inplace = True)
        df_annot_overlap.rename(columns = {'index': 'snp_index'}, inplace = True)

        # Collect the indexes of P_ldgm that have available annot info
        P_index = np.asarray(df_annot_overlap.loc[df_annot_overlap['_merge'] == "both", 'snp_index'])
        logging.info("Checking: Min P_index: {}; Max P_index: {}".format(min(P_index), max(P_index)))
        logging.info("# of SNPs indexed on the precision matrix that has annot info: {}".format((P_index.shape[0])))

        # Turn P_index -- which are numerical indexes -- into a vector of boolean values
        P_index_bool = np.full((P_ldgm.num_edges, ), False)
        P_index_bool[P_index] = True

        # Select the annot info of the overlapped SNPs with the LDGM precision matrix
        df_annot_overlap = df_annot_overlap.loc[df_annot_overlap['_merge'] == "both"]
        logging.info("Number of SNPs with annot info {}".format(df_annot_overlap.shape[0]))

        # generate the annotation matrix (based on the OVERLAPPED snps)
        df_annot = df_annot_overlap.loc[:, use_annots]
        annot_mat = df_annot.to_numpy()
        annot_mat[np.isnan(annot_mat)] = 0

        # format the summary statistics (based on the OVERLAPPED snps)
        df_sumstats = df_annot_overlap.loc[:, ['SNP', 'anc_alleles', 'deriv_alleles']]
        df_sumstats.rename(columns = {'anc_alleles': 'A1', 'deriv_alleles': 'A2'}, inplace = True)

        # drop columns from the annot file (for LDSC)
        df_annot_overlap = df_annot_overlap.drop(['snp_index', 'anc_alleles', 'deriv_alleles', '_merge'], axis = 1)

        if no_empty_annot:
            # remove annotations with all zeros
            logging.info("Updating the annotation file to exclude non-zero annotations")
            logging.info("Number of non-zero annotation values:")
            logging.info(np.sum(annot_mat != 0, axis=0))
            idx_drop_annots = np.where(np.sum(annot_mat != 0, axis=0) == 0)[0]
            empty_annot = df_annot.columns[list(idx_drop_annots)]
            logging.info("Names of the empty annotations: {}".format(empty_annot))

            # drop empty annotations from the annot matrix (for simul of a single block, etc.)
            logging.info("Dropping annotations with no non-zero values")
            df_annot_nonzero = df_annot.drop(df_annot.columns[list(idx_drop_annots)], axis = 1)
            annot_mat = df_annot_nonzero.to_numpy()
            annot_mat[np.isnan(annot_mat)] = 0

            logging.info("CHECK: Number of non-zero annotation values:")
            logging.info(np.sum(annot_mat != 0, axis=0))
            noAnnot = annot_mat.shape[1]

            # drop empty annotations from the annot file (for ldsc)
            logging.info("Updating the annotation file to exclude non-zero annotations")
            df_annot_overlap = df_annot_overlap.drop(list(empty_annot), axis = 1)
            df_annot_overlap.to_csv(annot_fp + "_no_empty_annot.annot", index = False, header = True, sep = '\t')

            # NOTE: 1) annot; 2) l2; 3) M, should all 
            # 1) have an equal number of annotations, and 2) include the baseline annotation.

        if ldscore_fp is not None:
            # process the LD score files
            df_ldscore = pd.read_csv(ldscore_fp + '.l2.ldscore', index_col = None, delim_whitespace = True)

            if no_empty_annot:
                ldscore_fp = ldscore_fp + "_no_empty_annot"
                logging.info("Updating the LD score file to exclude non-zero annotations")
                df_ldscore = df_ldscore.drop([str(x) + "L2" for x in empty_annot], axis = 1)
                df_ldscore.to_csv(ldscore_fp + '.l2.ldscore', index = False, header = True, sep = '\t')

            # # save the counts for diff annotations as text-delimited file
            # with open(ldscore_fp + ".l2.M", 'w') as file:
            #     file.write('\t'.join([str(x) for x in np.nansum(annot_mat, axis = 0)]))
        else:
            df_ldscore = None

        # # pad the SNPs with n/a annotation values with zeros
        # df_annot = pd.DataFrame(0, columns = use_annots, index = np.arange(p))
        # df_edge_P = pd.DataFrame(data = {'P_index': np.arange(P_ldgm.num_edges)[P_ldgm.nz_index]})
        # idx_P_in_annot = df_edge_P['P_index'].isin(df_annot_overlap['index']) # index of P LDGM: whether or not has annot 
        # idx_annot_in_P = df_annot_overlap['index'].isin(df_edge_P['P_index']) # index of annot: whether or not is in P
        
        # assert idx_P_in_annot.shape[0] == p
        # assert idx_annot_in_P.shape[0] == df_annot_overlap.shape[0]

        # # fill in the annotation matrix (pad the missing annot with 0s)
        # df_annot.loc[idx_P_in_annot, use_annots] = df_annot_overlap.loc[idx_annot_in_P, use_annots]
        # annot_mat = df_annot.to_numpy()
        # annot_mat[np.isnan(annot_mat)] = 0
    else:
        P_index_bool = P_ldgm.nz_index
        annot_mat = np.ones((p, 1))
        df_annot_overlap = None
        df_ldscore = None
        df_sumstats = None

    return P_ldgm, annot_mat, P_index_bool, df_annot_overlap, df_ldscore, df_sumstats


###### PARTITIONED H2 MODEL FUNCTIONS #######
def linkFn(annot_mat, tau_vec, link_type, floor_val = 1e-100):
    '''
    Link function, mapping the annotation (vector) of each SNP to its relative per-SNP heritability. 

    The number of columns of annot_mat should match the length of tau_vec.

    Output the vector per-SNP variance, of length p.

    '''
    (p, A) = annot_mat.shape

    assert annot_mat.shape[1] == len(tau_vec), "Dimension mismatch between annot and tau!"

    if link_type == "softmax":
        per_snp_h2 = np.log(1 + np.exp(annot_mat.dot(tau_vec)))

    elif link_type == "max":
        per_snp_h2 = np.clip(annot_mat.dot(tau_vec), a_min = floor_val, a_max = None)
        per_snp_h2 = np.square(per_snp_h2)

    elif link_type == "sigmoid":
        exp_factor = np.exp(-annot_mat.dot(tau_vec))
        per_snp_h2 = 1 / (1 + exp_factor)

    elif link_type == "exp":
        per_snp_h2 = np.exp(annot_mat.dot(tau_vec))

    elif link_type == "constant":
        per_snp_h2 = np.ones((p, ))

    return per_snp_h2

def linkFnGrad(annot_mat, tau_vec, link_type, floor_val = 1e-100):
    '''
    Gradient of the link function

    Output the Jacobinan matrix for each of the parameter, size p x A.

    '''
    assert annot_mat.shape[1] == len(tau_vec), "Dimension mismatch between annot and tau!"
    (p, A) = annot_mat.shape

    if link_type == "softmax":
        expit_factor = np.exp(annot_mat.dot(tau_vec)) / (1 + np.exp(annot_mat.dot(tau_vec)))
        jacob = np.einsum('i,ij->ij', expit_factor, annot_mat)
    
    elif link_type == "max":
        pos_indicator = annot_mat.dot(tau_vec) >= floor_val
        jacob = 2*np.einsum('i,ij->ij', pos_indicator, annot_mat)
        # jacob = annot_mat

    elif link_type == "sigmoid":
        exp_factor = np.exp(-annot_mat.dot(tau_vec))
        logit_factor = exp_factor / np.square(1 + exp_factor)
        jacob = np.einsum('i,ij->ij', logit_factor, annot_mat)

    elif link_type == "exp":
        exp_factor = np.exp(annot_mat.dot(tau_vec))
        jacob = np.einsum('i,ij->ij', exp_factor, annot_mat)

    elif link_type == "constant":
        jacob = np.zeros((p, A))
    
    return jacob 

def linkFnHess(annot_mat, tau_vec, link_type, floor_val = 1e-100):
    '''
    Hessian of the link function. Used for computing the exact Hessian of the likelihood function.

    Output the Hessian tensor for each of the parameter, of size p x A x A.

    '''
    assert annot_mat.shape[1] == len(tau_vec), "Dimension mismatch between annot and tau!"
    (p, A) = annot_mat.shape

    if link_type == "softmax":
        expit_sec_factor = np.exp(annot_mat.dot(tau_vec)) / ((1 + np.exp(annot_mat.dot(tau_vec)))**2)
        ycdi = np.einsum('i,ij->ij', expit_sec_factor, annot_mat)
        hess = np.einsum('ij,ik->ijk', ycdi, annot_mat)
    
    elif link_type == "max":
        hess = np.zeros((p, A, A))

    elif link_type == "sigmoid":
        exp_factor = np.exp(-annot_mat.dot(tau_vec))
        logit_sec_factor = (exp_factor - 1) * exp_factor / np.power(1+exp_factor, 3)
        ycdi = np.einsum('i,ij->ij', logit_sec_factor, annot_mat)
        hess = np.einsum('ij,ik->ijk', ycdi, annot_mat)

    elif link_type == "exp":
        exp_factor = np.exp(annot_mat.dot(tau_vec)) # px1 vector
        exp_sq_factor = np.square(exp_factor) # px1 vector
        ycdi = np.einsum('ij,i->ij', annot_mat, exp_sq_factor)  # pxA matrix
        hess = np.einsum('ij,ik->ijk', ycdi, annot_mat) # i-SNP; j,k-annotations
    
    elif link_type == "constant":
        hess = np.zeros((p, A, A))

    hess_pSum = np.sum(hess, axis = 0)

    return hess, hess_pSum

def params_to_estimates(annot_mat, params, link_type, per_snp_param, block_agg=False):
    '''
    Heritability enrichment model. 

    Convert the parameter estimates (objects of direct optimization)
    to enrichment and total h2, using the specified link and annotation matrix

    Inputs:
    -- annot_mat: includes the baseline annotation column depending on the mode of parametrization
    -- params: the vector of estimated or true parameters
    -- link_type: link function used for simulation or estimation
    -- per_snp_param: specifies the mode of the mapping from parameters to per-SNP h2
    -- block_agg: controls whether or not to conduct optimization w.r.t. multiple blocks

    Returns:
    -- phi: estimate of the total h2
    -- h2_annot: h2 from different categories
    -- enrich: enrichment of different categories

    '''

    if block_agg:
        annot_mat = np.concatenate(annot_mat, axis=0)

    p, noAnnot = annot_mat.shape
    logging.info("Parameter values:")

    # compute h2 -- depending on the mode of the parametrization
    if per_snp_param != "link_only":
        phi = params[0]
        tau_vec = params[1:] if len(params) > 1 else np.asarray([0])
        logging.info("Phi: {}".format(phi))
        logging.info("Tau: {}".format(tau_vec))

        per_snp_factor = linkFn(annot_mat, tau_vec, link_type)

        if per_snp_param == "normalized":
            per_snp_h2 = phi * per_snp_factor / np.sum(per_snp_factor)
        elif per_snp_param == "scaled":
            per_snp_h2 = phi * per_snp_factor
            phi = np.sum(per_snp_h2)

    elif per_snp_param == "link_only":
        tau_vec = params
        logging.info("Tau: {}".format(tau_vec))

        per_snp_factor = linkFn(annot_mat, tau_vec, link_type)
        per_snp_h2 = per_snp_factor
        phi = np.sum(per_snp_h2)

    # compute enrichment
    if per_snp_param != "link_only":
        annot_mat = np.hstack((np.ones((p, 1)), annot_mat))

    h2_annot = annot_mat.T.dot(per_snp_h2)
    base_h2 = h2_annot[0]
    # logging.info("Baseline h2 is: {}".format(base_h2))
    p_annot = np.nanmean(annot_mat, axis = 0)
    enrich = h2_annot[1:] / base_h2 / p_annot[1:]

    return phi, h2_annot[1:] / base_h2, enrich


def simul_beta_mixture(p, total_h2, annot_mat, tau_vec, gen_link_type, componentWeight, componentVariance, scaled = True):
    '''
    Simulate the joint true causal effect sizes from the mixture model, sampled from a mixture of normals. 

    Notes:
    -- scaled: controls whether the causal effects are scaled such that the total sum of squares equal to the specified total h2. Turn this to false for block-specific simulation, as the scaling needs to be w.r.t. the aggregated scale.

    Returns: 
    A vector of causal effect sizes simulated from the normal mixture

    '''
    noCpts = len(componentWeight)

    # mixture component assignments
    g_cpt = np.zeros((p, ))
    whichCpt = np.random.choice(np.arange(noCpts), size = p, replace = True, p = componentWeight)

    for cpt in range(noCpts):
        g_cpt[whichCpt == cpt, ] = np.random.normal(scale = np.sqrt(componentVariance[cpt]), size = np.sum(whichCpt == cpt))

    # scale effect sizes using linkFn
    assert annot_mat.shape[1] == len(tau_vec), "Mismatch of annot_mat and tau_vec when simulating BETA"
    beta_perSD = np.multiply(g_cpt, np.sqrt(linkFn(annot_mat, tau_vec, gen_link_type)))

    if scaled:
        logging.info("Scaling block-wise effect variance -- this should NOT be printed when simulating multi-block Z.")
        total_genVar = np.nansum(np.square(beta_perSD))
        beta_perSD = beta_perSD * np.sqrt(total_h2 / total_genVar)

    return beta_perSD

def simul_Z_ldgm_annot(n, total_h2, P_index, P_ldgm, annot_mat, tau_vec, gen_link_type, componentWeight, componentVariance):
    
    '''
    Simulate marginal association statistics directly from LDGM precision matrix,
    True causal effect sizes are generated based on mixture model, incorporating h2 partitioning.

    Input:
    -- n: GWAS sample size
    -- total_h2: total genetic variance (used to scale all the per-SNP h2)
    -- P_ldgm: LDGM object loaded from files
    -- P_index: index of the precision matrix to map with the beta's
    -- annot_mat: annotation matrix with dimension pxA
    -- tau_vec: linear effect of annotation (before the link function is applied)
    -- gen_link_type: type of link function used for generating the effect sizes.
    -- componentWeight: mixture weight for each h2 component. If sum is smaller than 1, a null component is added (var=0).
    -- componentVariance: per std-SD effect size variance for each mixture component

    '''
    p = annot_mat.shape[0]
    assert annot_mat.shape[1] == len(tau_vec), "Wrong dimension: annot_mat does not have the same number of columns as the length of tau_vec."

    noAnnot = annot_mat.shape[1]
    logging.info("Number of annotations: {}".format(noAnnot))
    logging.info("True tau values: {}". format(tau_vec))

    # mixture component related checks and setup
    assert len(componentVariance) == len(componentWeight), "Mismatched dimension between componentWeight and componentVariance."
    assert np.logical_and(np.all(np.asarray(componentWeight) <= 1), np.all(np.asarray(componentWeight) >= 0)), "Entries of all component weights need to be between 0 and 1"
    assert np.sum(componentWeight) <= 1, "Sum of component weight needs to be no grater than 1."

    if np.sum(componentWeight) < 1:
        logging.info("Mixture weights do not sum up to 1 -- adding a residual null component")
        componentWeight.append(1-np.sum(componentWeight))
        componentVariance.append(0)

    logging.info("Weights of the mixture components: ")
    logging.info(componentWeight)
    logging.info("Variances of the mixture components: ")
    logging.info(componentVariance)

    # Simulate true causal effects
    beta = simul_beta_mixture(p, total_h2, annot_mat, tau_vec, gen_link_type, componentWeight, componentVariance)

    # DEBUG
    logging.info("Diagnosis statistics of the causal effect sizes")
    # logging.info("Simulated causal beta ^ 2: {}".format(np.square(beta)))
    logging.info("Sum of beta ^ 2: {}".format(np.nansum(np.square(beta))))
    logging.info("Mean of beta: {}".format(np.nanmean(beta)))
    logging.info("Var of beta: {}".format(np.nanvar(beta)))
    logging.info("Size of the index for the beta in LDGM: {}".format(P_index.shape[0]))

    # # Back out the true covariance matrix
    # D = np.diag(np.square(beta)) 

    # Simulate conditional mean: Z|beta
    mean_z = math.sqrt(n) * P_ldgm.divide(beta, P_index)

    # Simulate correlated errors using LD as the covariance
    gauss_vec = np.random.normal(size = (p, ))
    eps = P_ldgm.root_divide(gauss_vec, P_index)

    Z_direct = mean_z + eps

    return Z_direct

def simul_Z_ldgm_annot_block(n, total_h2, P_index_list, P_ldgm_list, annot_mat_list, tau_vec, gen_link_type, componentWeight, componentVariance):

    '''
    Simulate marginal association statistics directly from LDGM precision matrices, where the beta's for multiple blocks are scaled altogether.

    Input:
    -- n: GWAS sample size
    -- P_index_list: a list of the indexes for the LDGM matrices
    -- P_ldgm_list: a list of the LDGM precision matrices
    -- annot_mat_list: a list of annotation matrices with the same number of columns
    -- tau_vec: linear effect of annotation (before the link function is applied)
    -- gen_link_type: type of link function used for generating the effect sizes.
    -- componentWeight: mixture weight for each h2 component. If sum is smaller than 1, a null component is added (var=0).
    -- componentVariance: per std-SD effect size variance for each mixture component

    Returns:
    -- A list of the block-specific Z statistics.
    '''

    # dimension checkers
    assert len(P_ldgm_list) == len(annot_mat_list), "Mismatched number of blocks between P_ldgm and annot_mat!"
    assert np.all(len(tau_vec) == annot_mat_list[i].shape[1] for i in range(len(annot_mat_list))), "Some annotation matrices do not have the correct number of columns, which should equal to the number of coefficients in tau."
    p_list = [annot_mat_list[i].shape[0] for i in range(len(annot_mat_list))]

    noAnnot = len(tau_vec)
    logging.info("Number of annotations: {}".format(noAnnot))
    logging.info("True tau values: {}". format(tau_vec))

    # mixture component related checks and setup
    assert len(componentVariance) == len(componentWeight), "Mismatched dimension between componentWeight and componentVariance."
    assert np.logical_and(np.all(np.asarray(componentWeight) <= 1), np.all(np.asarray(componentWeight) >= 0)), "Entries of all component weights need to be between 0 and 1"
    assert np.sum(componentWeight) <= 1, "Sum of component weight needs to be no grater than 1."

    if np.sum(componentWeight) < 1:
        logging.info("Mixture weights do not sum up to 1 -- adding a residual null component")
        componentWeight.append(1-np.sum(componentWeight))
        componentVariance.append(0)

    logging.info("Weights of the mixture components: ")
    logging.info(componentWeight)
    logging.info("Variances of the mixture components: ")
    logging.info(componentVariance)

    # Simulate true causal effects
    beta_raw_list = [simul_beta_mixture(p_list[i], total_h2, annot_mat_list[i], tau_vec, gen_link_type, componentWeight, componentVariance, scaled = False) for i in range(len(P_ldgm_list))]
    total_genVar = np.sum([np.nansum(np.square(beta_raw_list[i])) for i in range(len(P_ldgm_list))])
    beta_perSD_list = [beta_raw_list[i] * np.sqrt(total_h2 / total_genVar) for i in range(len(P_ldgm_list))]

    # DEBUG
    beta = np.concatenate(beta_perSD_list, axis=0)
    logging.info("DEBUG: checking the total h2 based on the simulated betas: {}".format(np.nansum(np.square(beta))))

    # Side by side comparison with matlab
    logging.info("Diagnosis statistics of the causal effect sizes")
    # logging.info("Simulated causal beta ^ 2: {}".format(np.square(beta)))
    logging.info("Sum of beta ^ 2: {}".format(np.nansum(np.square(beta))))
    logging.info("Mean of beta: {}".format(np.nanmean(beta)))
    logging.info("Var of beta: {}".format(np.nanvar(beta)))
    # logging.info("Size of the index for the beta in LDGM: {}".format(np.sum(P_index)))

    # # Back out the true covariance matrix
    # D = np.diag(np.square(beta))

    # Simulate conditional mean: Z|beta
    mean_z_list = [math.sqrt(n) * P_ldgm_list[i].divide(beta_perSD_list[i], P_index_list[i]) for i in range(len(P_ldgm_list))]

    # Simulate correlated errors using LD as the covariance
    gauss_vec_list = [np.random.normal(size = (p_list[i], )) for i in range(len(P_ldgm_list))]
    eps_list = [P_ldgm_list[i].root_divide(gauss_vec_list[i], P_index_list[i]) for i in range(len(P_ldgm_list))]

    Z_direct_list = [mean_z_list[i] + eps_list[i] for i in range(len(P_ldgm_list))]

    return Z_direct_list

# Annotation enabled likelihood functions
def neg_logL_Fn_ldgm_annot(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, kernel = "Euc"):
    
    '''
    Negative of the log-likelihood function,
    with partitioned h2 model enabled.

    Assuming data from a single data is inputted. 

    Input:
    -- params: vector of parameters to be optimized. linear effect of annotation (before the link function is applied). [phi, tau]
    -- Z: vector of marginal association statistics
    -- P_ldgm: LDGM precision matrix class object
    -- P_index: index of the precision matrix for sumstats 
    -- n: GWAS sample size
    -- annot_mat: annotation matrix with dimension pxA
    -- link_type: type of link function, either "softmax" for multiplicative or "max" for additive effects
    -- per_snp_param: mode of parametrization, mapping from coef to per-SNP h2
    -- kernel: kernel used for estimation

    Output:
    -- Returns a scalar of likelihood.

    '''
    # read in the current parameter values
    if per_snp_param != "link_only":
        phi = params[0]
        tau_vec = params[1:] if len(params) > 1 else np.asarray([0])
        link_val = linkFn(annot_mat, tau_vec, link_type)
    else:
        link_val = linkFn(annot_mat, params, link_type)

    G = np.sum(link_val) # 1x1 scalar

    # compute M
    p = np.sum(P_index)
    P = sliceMatrix(P_ldgm.matrix, P_index, P_index)

    # DEBUG + warning
    if per_snp_param != "link_only":
        assert phi > 0, "Negative phi value: {}".format(phi)

    if np.sum(link_val <= 0) > 0:
        logging.info("There exist {} non-positive values on diag(K).".format(np.sum(link_val <= 0)))
        neg_idx = np.where(link_val <= 0)[0]
        invalid_sigma = (link_val)[neg_idx]
        logging.info("Values of g are: {}".format(invalid_sigma))

    if per_snp_param == "normalized":
        K = scipy.sparse.diags(link_val * phi / G, format = "csc")
    elif per_snp_param == "scaled":
        K = scipy.sparse.diags(link_val * phi, format = "csc")
    elif per_snp_param == "link_only":
        K = scipy.sparse.diags(link_val, format = "csc")

    if kernel == "Mah":
        K = P.dot(K)

    M = n * K + P

    # neg of log-likelihood formula
    factor = cholesky(M)
    Minv_Z = factor(Z)
    neg_logL = 0.5*(p * np.log(2*math.pi) + factor.logdet() + np.dot(Z.T, Minv_Z)[0][0])

    return neg_logL

def grad_Fn_ldgm_annot(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, trace_method = "MC", kernel = "Euc", stoch_samples = 20):

    '''
    First derivative of the log-likelihood function,
    with partitioned h2 model enabled.

    Assuming data from a single data is inputted. 

    Input:
    -- params: vector of parameters to be optimized. linear effect of annotation (before the link function is applied). Depending on the 
    -- Z: vector of marginal association statistics
    -- P_ldgm: LDGM precision matrix class object
    -- P_index: index of the precision matrix for sumstats 
    -- n: GWAS sample size
    -- annot_mat: annotation matrix with dimension pxA
    -- link_type: type of link function, either "softmax" for multiplicative or "max" for additive effects
    -- per_snp_param: mode of parametrization, mapping from coef to per-SNP h2
    -- kernel: kernel used for estimation
    -- stoch_samples: number of samples to use for stochastic method
    -- trace_method: method used to evaluate the trace term. Options include "exact", "MC" and "Hutchinson"

    Output:
    -- grad: vector of length (noAnnot+1), where 1 is for the scale factor phi

    '''
    logging.info("eval grad...")

    if per_snp_param != "link_only":
        phi = params[0]
        tau_vec = params[1:] if len(params) > 1 else np.asarray([0])
        link_val = linkFn(annot_mat, tau_vec, link_type)
        link_jacob = linkFnGrad(annot_mat, tau_vec, link_type)
    else:
        link_val = linkFn(annot_mat, params, link_type)
        link_jacob = linkFnGrad(annot_mat, params, link_type)

    # transform the param estimates into sigmas
    G = np.sum(link_val) # 1x1 scalar
    J = np.sum(link_jacob, axis = 0) # Ax1 or (A+1)x1 vector
    noAnnot = link_jacob.shape[1]
    noParams = noAnnot if per_snp_param == "link_only" else noAnnot+1

    # compute M
    p = np.sum(P_index)
    P = sliceMatrix(P_ldgm.matrix, P_index, P_index)

    if per_snp_param == "normalized":
        K = scipy.sparse.diags(link_val * phi / G, format = "csc")
    elif per_snp_param == "scaled":
        K = scipy.sparse.diags(link_val * phi, format = "csc")
    elif per_snp_param == "link_only":
        K = scipy.sparse.diags(link_val, format = "csc")

    if kernel == "Mah":
        K = P.dot(K)

    M = n * K + P
    factor = cholesky(M, ordering_method = "natural")
    Minv_Z = factor(Z).reshape(-1, )

    if per_snp_param == "normalized":
        dMdtau = phi / (G**2) * (G*link_jacob - np.outer(link_val, J)) # pxA matrix
        dMdtau = np.hstack((link_val.reshape(-1, 1) / G, dMdtau)) # px(A+1) matrix
    elif per_snp_param == "scaled":
        dMdtau = phi * link_jacob # pxA matrix
        dMdtau = np.hstack((link_val.reshape(-1, 1), dMdtau)) # px(A+1) matrix
    elif per_snp_param == "link_only":
        dMdtau = link_jacob # px(A+1) matrix

    # Quadratic term
    dsdA_Minv_Z = np.einsum('i,ij->ij', Minv_Z, dMdtau)
    assert dsdA_Minv_Z.shape[1] == len(params), "Wrong dimension of dsdA_Minv_Z -- should be noAnnot + 1"

    if kernel == "Euc":
        quad_term = Minv_Z.T.dot(dsdA_Minv_Z)
    elif kernel == "Mah":
        P_dsdA_Minv_Z = P_ldgm.multiply(dsdA_Minv_Z, P_index)
        quad_term = Minv_Z.T.dot(P_dsdA_Minv_Z)

    # logging.info("Quadratic term: {}".format(quad_term))

    # Trace trick implementation
    if trace_method == "exact":
        # Direct inverse of M
        Minv = factor(np.eye(p))
        if kernel == "Euc":
            trace_term = dMdtau.T.dot(np.diag(Minv))
        elif kernel == "Mah":
            trace_term = dMdtau.T.dot(P_ldgm.multiply(Minv, P_index))

        logging.info("Trace from direct inverse: ")
        # logging.info(trace_term)

    elif trace_method == "MC":    
        # Monte Carlo trick to compute the gradient
        vWv = np.zeros((noParams, stoch_samples))
        for j in range(stoch_samples):
            # sample from M^-1
            gauss_vec = np.random.normal(size = (p, ))
            yi = factor.solve_Lt(gauss_vec, use_LDLt_decomposition = False)

            # stochastic approx of the trace
            dsdA_yi = np.einsum('i,ij->ij', yi, dMdtau)
            if kernel == "Euc":
                vWv[:, j] = yi.T.dot(dsdA_yi)
            elif kernel == "Mah":
                P_dsdA_yi = P_ldgm.multiply(dsdA_yi, P_index)
                vWv[:, j] = yi.T.dot(P_dsdA_yi)

        trace_term = np.mean(vWv, axis=1)
        logging.info("Trace from using the MC approx: ")
        # logging.info(trace_term)

        # DEBUG
        if math.isnan(trace_term[0]):
            logging.info("Phi value: {}".format(phi))
            logging.info("link_val: {}".format(link_val))
            logging.info("dMdtau: {}".format(dMdtau))
            logging.info("Quad term: {}".format(quad_term))
            logging.info("Trace term: {}".format(trace_term))
            logging.info("Minv_Z: {}".format(Minv_Z))

    elif trace_method == "Hutchinson":
        # Hutchinson's estimator (Rademacher rv)
        Ber = np.random.binomial(n=1, p=0.5, size = (p, stoch_samples))
        V = np.power(-1, Ber)
        vWv = np.zeros((noParams, stoch_samples))
        for j in range(stoch_samples):
            yi = factor(V[:,j])
            dsdA_yi = np.einsum('i,ij->ij', yi, dMdtau)
            if kernel == "Euc":
                vWv[:, j] = np.dot(V[:,j].T, dsdA_yi)
            elif kernel == "Mah":
                P_dsdA_yi = P_ldgm.multiply(dsdA_yi, P_index)
                vWv[:, j] = np.dot(V[:,j].T, P_dsdA_yi)

        trace_term = np.mean(vWv, axis = 1)
        logging.info("Trace from the Hutchinson: ")
        logging.info(trace_term)

    grad = 0.5 * n * (trace_term - quad_term)

    return grad

def hess_Fn_approx_ldgm_annot(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, kernel = "Euc"):
    '''
    Second derivative of the negative log-likelihood function
    Approximated using the AI trick, using observed information instead of expected.
    The most simplied Hessian calculation. 
    May not be a good approximation, depending on the choice of the link function, and annotation values.

    Input:
    -- params: vector of parameters to be optimized. linear effect of annotation (before the link function is applied). [phi, tau]
    -- Z: vector of marginal association statistics
    -- P_index: index of the precision matrix for sumstats 
    -- P_ldgm: LDGM precision matrix class object
    -- n: GWAS sample size
    -- annot_mat: annotation matrix with dimension pxA
    -- link_type: type of link function, either "softmax" for multiplicative or "max" for additive effects
    -- per_snp_param: mode of parametrization, mapping from coef to per-SNP h2
    -- kernel: kernel used for estimation

    Output:
    -- hess: returns a matrix of dim (noAnnot+1) x (noAnnot+1).

    '''

    logging.info("eval hess...")

    if per_snp_param != "link_only":
        phi = params[0]
        tau_vec = params[1:] if len(params) > 1 else np.asarray([0])
        link_val = linkFn(annot_mat, tau_vec, link_type)
        link_jacob = linkFnGrad(annot_mat, tau_vec, link_type)
    else:
        link_val = linkFn(annot_mat, params, link_type)
        link_jacob = linkFnGrad(annot_mat, params, link_type)

    # transform the param estimates into sigmas
    G = np.sum(link_val) # 1x1 scalar
    J = np.sum(link_jacob, axis = 0) # Ax1 vector
    noAnnot = link_jacob.shape[1]

    # compute M and its factorization
    p = np.sum(P_index)
    P = sliceMatrix(P_ldgm.matrix, P_index, P_index)

    if per_snp_param == "normalized":
        K = scipy.sparse.diags(link_val * phi / G, format = "csc")
    elif per_snp_param == "scaled":
        K = scipy.sparse.diags(link_val * phi, format = "csc")
    elif per_snp_param == "link_only":
        K = scipy.sparse.diags(link_val, format = "csc")

    if kernel == "Mah":
        K = P.dot(K)

    M = n * K + P
    factor = cholesky(M, ordering_method = "natural")
    Minv_Z = factor(Z).reshape(-1, )

    if kernel == "Euc":
        dKdsigma = Minv_Z
    elif kernel == "Mah":
        dKdsigma = P_ldgm.multiply(Minv_Z, P_index)

    if per_snp_param == "normalized":
        dMdtau = phi / (G**2) * (G*link_jacob - np.outer(link_val, J)) # pxA matrix
        dMdtau = np.hstack((link_val.reshape(-1, 1) / G, dMdtau)) # px(A+1) matrix
    elif per_snp_param == "scaled":
        dMdtau = phi * link_jacob # pxA matrix
        dMdtau = np.hstack((link_val.reshape(-1, 1), dMdtau)) # px(A+1) matrix
    elif per_snp_param == "link_only":
        dMdtau = link_jacob # px(A+1) matrix

    # quadratic term
    dsdA_Minv_Z = np.einsum('i,ij->ij', dKdsigma, dMdtau) # bread
    dsdA_Minv_Z_mid = factor(dsdA_Minv_Z) # bread + cheese
    quad_term = np.dot(dsdA_Minv_Z.T, dsdA_Minv_Z_mid) # bread

    hess = 0.5 * (n**2) * quad_term

    return hess

def hess_Fn_ldgm_annot(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, kernel = "Euc"):
    '''
    Second derivative of the negative log-likelihood function
    Approximated by omitting the second order derivative of the covariance matrix. 
    May not be a good approximation, depending on the choice of the link function, and annotation values.

    Input:
    -- params: vector of parameters to be optimized. linear effect of annotation (before the link function is applied). [phi, tau]
    -- Z: vector of marginal association statistics
    -- P_ldgm: LDGM precision matrix class object
    -- P_index: index of the precision matrix for sumstats 
    -- n: GWAS sample size
    -- annot_mat: annotation matrix with dimension pxA
    -- link_type: type of link function, either "softmax" for multiplicative or "max" for additive effects
    -- per_snp_param: mode of parametrization, mapping from coef to per-SNP h2
    -- kernel: kernel used for estimation

    Output:
    -- hess: a matrix of dim (noAnnot+1) x (noAnnot+1).

    '''

    if per_snp_param != "link_only":
        phi = params[0]
        tau_vec = params[1:] if len(params) > 1 else np.asarray([0])
        link_val = linkFn(annot_mat, tau_vec, link_type)
        link_jacob = linkFnGrad(annot_mat, tau_vec, link_type)
    else:
        link_val = linkFn(annot_mat, params, link_type)
        link_jacob = linkFnGrad(annot_mat, params, link_type)

    # transform the param estimates into sigmas
    G = np.sum(link_val) # 1x1 scalar
    J = np.sum(link_jacob, axis = 0) # Ax1 vector
    noAnnot = link_jacob.shape[1]

    # compute M and its factorization
    p = np.sum(P_index)
    P = sliceMatrix(P_ldgm.matrix, P_index, P_index)

    # logging.info("Sum of the diagonal of P: {}".format(np.sum(P.diagonal())))
    
    if per_snp_param == "normalized":
        K = scipy.sparse.diags(link_val * phi / G, format = "csc")
    elif per_snp_param == "scaled":
        K = scipy.sparse.diags(link_val * phi, format = "csc")
    elif per_snp_param == "link_only":
        K = scipy.sparse.diags(link_val, format = "csc")

    if kernel == "Mah":
        K = P.dot(K)

    M = n * K + P
    factor = cholesky(M, ordering_method = "natural")

    if per_snp_param == "normalized":
        dMdtau = phi / (G**2) * (G*link_jacob - np.outer(link_val, J)) # pxA matrix
        dMdtau = np.hstack((link_val.reshape(-1, 1) / G, dMdtau)) # px(A+1) matrix
    elif per_snp_param == "scaled":
        dMdtau = phi * link_jacob # pxA matrix
        dMdtau = np.hstack((link_val.reshape(-1, 1), dMdtau)) # px(A+1) matrix
    elif per_snp_param == "link_only":
        dMdtau = link_jacob # px(A+1) matrix

    Minv = factor(np.eye(p))

    if kernel == "Euc":
        dKdsigma = Minv
    elif kernel == "Mah":
        dKdsigma = P_ldgm.multiply(Minv, P_index)

    # trace term
    dsdA_Minv = np.einsum('ij,ik->ijk', dKdsigma, dMdtau) # pxpx(A+1) tensor
    trace_term = np.einsum('ijk,jih->kh', dsdA_Minv, dsdA_Minv) # (A+1)x(A+1) matrix

    # quadratic term
    Minv_Z = factor(Z).reshape(-1, )
    if kernel == "Euc":
        dKdsigma = Minv_Z
    elif kernel == "Mah":
        dKdsigma = P_ldgm.multiply(Minv_Z, P_index)

    dsdA_Minv_Z = np.einsum('i,ij->ij', dKdsigma, dMdtau) # bread
    dsdA_Minv_Z_mid = factor(dsdA_Minv_Z) # bread + cheese
    quad_term = np.dot(dsdA_Minv_Z.T, dsdA_Minv_Z_mid) # bread

    # DEBUG:
    logging.info("If the AI approximation works well, the following quantities should match:")
    logging.info("Quadratic term: {}".format(quad_term[0,0]))
    logging.info("Trace term: {}".format(trace_term[0,0]))

    hess = 0.5 * (n**2) * (2*quad_term - trace_term)

    return hess

def hess_Fn_exact_ldgm_annot(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, kernel = "Euc"):
    '''
    Second derivative of the negative log-likelihood function
    Exact calculation of the Hessian matrix without any approximation. 

    NOTE: currently the parametrization without the normalization has not been enabled yet.

    Input:
    -- params: vector of parameters to be optimized. linear effect of annotation (before the link function is applied).
    -- Z: vector of marginal association statistics
    -- P_ldgm: LDGM precision matrix class object
    -- P_index: index of the precision matrix for sumstats 
    -- n: GWAS sample size
    -- annot_mat: annotation matrix with dimension pxA
    -- link_type: type of link function, either "softmax" for multiplicative or "max" for additive effects
    -- per_snp_param: mode of parametrization, mapping from coef to per-SNP h2
    -- kernel: kernel used for estimation

    Output:
    -- hess: a matrix of dim (A+1) x (A+1).

    '''

    # read in the current parameter values
    phi = params[0]

    # transform the param estimates into sigmas
    link_val = linkFn(annot_mat, params[1:], link_type)
    link_jacob = linkFnGrad(annot_mat, params[1:], link_type) # pxA
    link_hess, link_hess_pSum = linkFnHess(annot_mat, params[1:], link_type) # pXAxA

    G = np.sum(link_val) # scalar
    J = np.sum(link_jacob, axis = 0) # Ax1 vector

    noAnnot = link_jacob.shape[1]

    # compute M and its factorization
    p = np.sum(P_index)
    P = sliceMatrix(P_ldgm.matrix, P_index, P_index)
    K = scipy.sparse.diags(link_val * phi / G, format = "csc")
    
    if kernel == "Mah":
        K = P.dot(K)
    M = n * K + P
    factor = cholesky(M, ordering_method = "natural")

    Minv = factor(np.eye(p))

    if kernel == "Euc":
        dKdsigma = Minv
    elif kernel == "Mah":
        dKdsigma = P_ldgm.multiply(Minv, P_index)
    # dKdsigma: pxp matrix

    dMdtau = (phi / (G**2)) * (G*link_jacob - np.outer(link_val, J)) # pxA matrix
    dMdtau = np.hstack((link_val.reshape(-1, 1) / G, dMdtau)) # px(A+1) matrix

    # trace terms
    dsdA_Minv = np.einsum('ij,ik->ijk', dKdsigma, dMdtau) # pxpx(A+1) tensor
    trace_fi_ord = np.einsum('ijk,jil->kl', dsdA_Minv, dsdA_Minv) # (A+1)x(A+1) matrix

    d2M_d2tau = (phi / (G**2)) * (G*link_hess - np.einsum('k,ij->ikj', J, link_jacob) - np.einsum('i,kj->ikj', link_val, link_hess_pSum)) # pxAxA tensor

    H1 = np.concatenate((np.zeros((p,1,1)), np.expand_dims(link_jacob, axis=1)), axis = 2) # px1x(A+1) tensor
    H2 = np.concatenate((np.expand_dims(link_jacob, axis = 2), d2M_d2tau), axis = 2) # pxAx(A+1) tensor
    d2M_d2params =  np.concatenate((H1, H2), axis = 1) # px(A+1)x(A+1) tensor

    trace_sec_ord = np.einsum('ijk,i->jk', d2M_d2params, np.diag(dKdsigma)) # (A+1)x(A+1) matrix
    # trace_sec_ord = np.trace(np.einsum('ijk,il->iljk', d2M_d2tau, Minv)) # this is equiv to the last line, but uses so much more memory

    # quadratic term
    Minv_Z = factor(Z).reshape(-1, )
    if kernel == "Euc":
        dKdsigma = Minv_Z
    elif kernel == "Mah":
        dKdsigma = P_ldgm.multiply(Minv_Z, P_index)

    dsdA_Minv_Z = np.einsum('i,ij->ij', dKdsigma, dMdtau) # bread
    dsdA_Minv_Z_mid = factor(dsdA_Minv_Z) # bread + cheese
    quad_fi_ord = np.dot(dsdA_Minv_Z.T, dsdA_Minv_Z_mid) # bread

    quad_sec_ord = np.einsum('i,ijk->jk', dKdsigma, np.einsum('ijk,i->ijk', d2M_d2params, dKdsigma))

    # logging.info("Missed second-order quad in approx Hess: {}".format(quad_sec_ord))

    hess = 0.5 * n * (trace_sec_ord - n*trace_fi_ord + 2*n*quad_fi_ord - quad_sec_ord)

    return hess

def MLE_optim_annot(args, llfun, gradfun, hessfun, Z, P_index, P_ldgm, n, annot_mat, params_0, kernel="Euc", tol=1e-4, optim_maxiter=20, optim_method="trust-constr", block_agg=False):

    '''
    Use the built-in minimization function to perform GD
    Automatic tuning of step size, etc. hyperparameters.

    Input:
    -- llfun, gradfun, hessfun: block-specific likelihood functions (block-wise function)
    -- Z: Z-statistics
    -- P_index: index of the precision matrix for the sumstats
    -- P_ldgm: LDGM precision matrices
    -- n: sample size
    -- annot_mat: annotation matrix with dimension pxA or px(A+1) depending on the parametrization
    -- params_0: initial values of: heritability parameters ()
    -- tol: stopping criterion for checking the change of likelihood
    -- optim_maxiter: argument passed to the minimize() function for optimization; max number of iterations for each update
    -- use_hess: if set to False, GD with 1st order updates; if set to True, second order method is used
    -- optim_method: argument passed to the minimize() function; method of optimization
    -- block_agg: controls whether or not to conduct optimization w.r.t. multiple blocks

    Output:
    -- estimate of params from MLE
    -- number of steps (optimizations taken)
    -- a history of the likelihoods across iterations
    -- a history of the time elapsed for each optimization
    '''

    ll_list = [1e8, 1e7]
    delta_ll_list = []
    steps = 0
    timer_list = []
    params = params_0.copy()

    # modify functions to accommodate multi-block optim 
    # (everything is added across blocks)
    if block_agg:
        num_blocks = len(Z)
        def ll_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param):
            ll = np.sum([llfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param=per_snp_param) for i in range(num_blocks)])
            # logging.info("Likelihood: {}".format(ll))
            return ll

        def grad_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, trace_method):
            grad = np.sum([gradfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param=per_snp_param, trace_method=trace_method) for i in range(num_blocks)], axis = 0)
            # logging.info("Gradient: {}".format(grad))
            return grad

        def hess_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param):
            hess = np.sum([hessfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param=per_snp_param) for i in range(num_blocks)], axis = 0)
            # logging.info("Hessian: {}".format(hess))
            return hess
        
    else:
        ll_fun = llfun
        grad_fun = gradfun
        hess_fun = hessfun


    # Descent using the optim wrapper function
    while (abs(ll_list[-2] - ll_list[-1]) > tol and (steps < args.maxIter)):
        logging.info("Current parameter estimate: {}".format(params))
        steps += 1
        logging.info("Iteration {}".format(steps))   
        t1_start = perf_counter()

        # set up bounds for total h2
        # bnds_params = [(-math.inf, math.inf) for i in range(len(params))]
        lbs = -1*np.inf*np.ones(len(params))
        ubs = np.inf*np.ones(len(params))

        if args.per_snp_param != "link_only":
            # bnds_params[0] = (1e-200, math.inf)
            # bnds_params[0] = (0,1)
            lbs[0] = 0
            ubs[0] = 1 
        
        boundData = Bounds(lbs,ubs, keep_feasible = True)           

        # diff maxiter names for diff optim functions
        maxarg_name = 'maxfun' if optim_method == "TNC" else 'maxiter'

        opt = minimize(ll_fun, x0 = params, 
                  args = (Z, P_index, P_ldgm, n, annot_mat, args.est_link_type, args.per_snp_param, args.trace_method),  
                  jac = grad_fun,
                  hess = hess_fun, 
                  method = optim_method,
                  bounds = boundData, 
                  # constraints = [nonlinearConstraints],
                  options = {'disp': True, maxarg_name: optim_maxiter})

        params = opt.x
        logging.info("Updated parameter value are {}".format(params))
        t1_stop = perf_counter()
 
        # check that the likelihood indeed increases with the update
        ll = ll_fun(params, Z, P_index, P_ldgm, n, annot_mat, args.est_link_type, args.per_snp_param)
        logging.info("Current neg-LogL: {}".format(ll))
        delta_ll = ll - ll_list[-1]

        if delta_ll > 0:
            logging.info("Warning: neg-logL just increased. The update may be wrong.")

        ll_list.append(ll)
        timer_list.append(t1_stop - t1_start)
        logging.info("Time elapsed for the current iteration: {}".format(t1_stop - t1_start))

    return params, steps, ll_list[2:], timer_list

def MLE_NR_annot(args, llfun, gradfun, hessfun, Z, P_index, P_ldgm, n, annot_mat, params_0, kernel="Euc", tol=0.1, TR_param=1e-3, block_agg=False):

    '''
    Use Newton-Raphson to perform GD. Requires some manual tuning of step size, etc. hyperparameters.

    Input:
    -- llfun, gradfun, hessfun: block-specific likelihood functions (block-wise function)
    -- Z: Z-statistics
    -- P_index: index of the precision matrix for the sumstats
    -- P_ldgm: LDGM precision matrices
    -- n: sample size
    -- annot_mat: annotation matrix with dimension pxA or px(A+1) depending on the parametrization
    -- params_0: initial values of: heritability parameters ()
    -- tol: stopping criterion for checking the change of likelihood
    -- block_agg: controls whether or not to conduct optimization w.r.t. multiple blocks

    Output:
    -- estimate of params from MLE
    -- number of steps (optimizations taken)
    -- a history of the likelihoods across iterations
    -- a history of the time elapsed for each optimization
    '''

    ll_list = [1e8, 1e7]
    delta_ll_list = []
    steps = 0
    timer_list = []
    params = params_0.copy()

    # modify functions to accommodate multi-block optim 
    # (everything is added across blocks)
    if block_agg:
        num_blocks = len(Z)
        def ll_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param):
            ll = np.sum([llfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param=per_snp_param) for i in range(num_blocks)])
            # logging.info("Likelihood: {}".format(ll))
            return ll

        def grad_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param, trace_method):
            grad = np.sum([gradfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param=per_snp_param, trace_method=trace_method) for i in range(num_blocks)], axis = 0)
            # logging.info("Gradient: {}".format(grad))
            return grad

        def hess_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param):
            hess = np.sum([hessfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param=per_snp_param) for i in range(num_blocks)], axis = 0)
            # logging.info("Hessian: {}".format(hess))
            return hess
        
    else:
        ll_fun = llfun
        grad_fun = gradfun
        hess_fun = hessfun


    # Descent using the optim wrapper function
    while (abs(ll_list[-2] - ll_list[-1]) > tol and (steps < args.maxIter)):
        logging.info("Current parameter estimate: {}".format(params))
        steps += 1
        logging.info("Iteration {}".format(steps))   
        t1_start = perf_counter()
        
        # evaluate grad and hess at the current iteration
        grad = grad_fun(params, Z, P_index, P_ldgm, n, annot_mat, args.est_link_type, args.per_snp_param, args.trace_method)
        hess = hess_fun(params, Z, P_index, P_ldgm, n, annot_mat, args.est_link_type, args.per_snp_param)

        # update parameters using NR
        hess_adj = hess + TR_param * np.diag(np.diag(hess)) + 0.01 * TR_param * np.mean(np.diag(hess)) * np.eye(hess.shape[0])
        params_propose = params - grad.dot(np.linalg.inv(hess_adj))
        ll = ll_fun(params_propose, Z, P_index, P_ldgm, n, annot_mat, args.est_link_type, args.per_snp_param)
        logging.info("Current likelihood: {}".format(ll))
        logging.info("Current gradient: {}".format(grad))

        # adapt step size if likelihood increases
        if steps > 1:
            while ll - ll_list[-1] > 1e-6:
                logging.info("Warning: Objective function increases at iteration {}".format(steps))
                logging.info("Doubling the step size...")
                TR_param = 2*TR_param
                hess_adj = hess + TR_param * np.diag(np.diag(hess)) + 0.01 * TR_param * np.mean(np.diag(hess)) * np.eye(hess.shape[0])
                params_propose = params - grad.dot(np.linalg.inv(hess_adj))
                ll = ll_fun(params_propose, Z, P_index, P_ldgm, n, annot_mat, args.est_link_type, args.per_snp_param)

        params = params_propose.copy()
        logging.info("Updated parameter value are {}".format(params))
        ll_list.append(ll)
        t1_stop = perf_counter()

        timer_list.append(t1_stop - t1_start)
        logging.info("Time elapsed for the current iteration: {}".format(t1_stop - t1_start))

    return params, steps, ll_list[2:], timer_list


#### Compute variance or the SE
def calc_params_SE(params, Z, P_index, P_ldgm, n, annot_mat, link_type, hessfun, per_snp_param, block_agg=False):

    '''
    Input:
    -- params: vector of parameters to be optimized. linear effect of annotation (before the link function is applied). [phi, tau]
    -- Z: vector of marginal association statistics
    -- P_ldgm: LDGM precision matrix class object
    -- P_index: index of the precision matrix for sumstats 
    -- n: GWAS sample size
    -- annot_mat: annotation matrix with dimension pxA
    -- link_type: type of link function, either "softmax" for multiplicative or "max" for additive effects
    -- hessfun: Hessian function (single block!) 
    -- block_agg: controls whether or not to conduct optimization w.r.t. multiple blocks

    Output:
    -- Estimated standard errors of the parameters

    '''

    if block_agg:
        num_blocks = len(Z)
        def hess_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param):
            hess = np.sum([hessfun(params, Z[i], P_index[i], P_ldgm[i], n, annot_mat[i], link_type, per_snp_param) for i in range(num_blocks)], axis = 0)
            return hess
    else:
        hess_fun = hessfun

    hess_mat = hess_fun(params, Z, P_index, P_ldgm, n, annot_mat, link_type, per_snp_param)
    hess_fail = True
    diag_hess_scalar = 0.1
    attempts = 0

    while hess_fail and attempts <= 10:
        try:
            params_inv = np.linalg.inv(hess_mat)
            hess_fail = False
        except Exception as e:
            logging.error(e, exc_info=True)
            logging.info("Warning: Hessian at the estimated values of the parameters is not PSD -- Add pos number to its diag to boost its PSD.")
            # hess_mat = hess_mat + np.diag(np.diag(hess_mat)) * diag_hess_scalar
            hess_mat = hess_mat + np.eye(params.shape[0]) * diag_hess_scalar
            diag_hess_scalar  = diag_hess_scalar * 10
            attempts += 1

    if hess_fail:
        logging.info("After 10 attempts to adjust for the non-PSD of Hessian, the Hessian matrix is still not PSD. Assign NaN values to the SE and move on...")
        params_SE = np.empty((params.shape[0], )) * np.nan

    else:
        if block_agg:
            annot_mat = np.concatenate(annot_mat, axis=0)

        # compute the SE of the enrichment
        if per_snp_param != "link_only":
            phi = params[0]
            link_val = linkFn(annot_mat, params[1:], link_type)
            link_jacob = linkFnGrad(annot_mat, params[1:], link_type)
        else:
            link_val = linkFn(annot_mat, params, link_type)
            link_jacob = linkFnGrad(annot_mat, params, link_type)

        # transform the param estimates into sigmas
        G = np.sum(link_val) # 1x1 scalar
        J = np.sum(link_jacob, axis = 0) # Ax1 or (A+1)x1 vector
        noAnnot = link_jacob.shape[1]
        noParams = noAnnot if per_snp_param == "link_only" else noAnnot+1

        # derivative w.r.t. tau and annotation values
        dMdtau = 1 / (G**2) * (G*link_jacob - np.outer(link_val, J))
        dMdtau_A = dMdtau.T.dot(annot_mat)
        p_annot = np.nanmean(annot_mat, axis = 0)

        if per_snp_param != "link_only":
            SE_h2 = np.sqrt(np.diag(dMdtau_A.T.dot(params_inv[1:, 1:].dot(dMdtau_A))))
            enrich_SE = SE_h2 / p_annot
        else:
            SE_h2 = np.sqrt(np.diag(dMdtau_A.T.dot(params_inv.dot(dMdtau_A))))
            enrich_SE = SE_h2[1:] / p_annot[1:]

        # append the SE of the total h2
        if per_snp_param == "normalized":
            h2_SE = math.sqrt(params_inv[0,0])
        elif per_snp_param == "scaled":
            grad_h2 = np.insert(params[0]*J, 0, G)
            h2_SE = math.sqrt(grad_h2.T.dot(params_inv.dot(grad_h2)))
        elif per_snp_param == "link_only":
            h2_SE = math.sqrt(J.dot(params_inv.dot(J)))

        params_SE = np.insert(enrich_SE, 0, h2_SE)

    return params_SE

def run_MLE_optim_annot(args, Z_tilde, P_index, P_ldgm, annot_mat, n, params_init, block_agg=False, keep_init_0=False):

    '''
    Wrapper function for running the MLE to estimate the partitioned heritability.
    MLE is found using the minimize wrapper function.
    Main goal of this wrapper is to reinitialize the starting values of tau if the random start is bad.

    Input: 
    -- Z_tilde: Z-statistics after the precision matrix has been applied
    -- P_index: index of the precision matrix which map to the sumstats
    -- P_ldgm: LDGM precision matrix
    -- annot_mat: annotation matrix
    -- n: sample size
    -- params_init: initial parameter values
    -- block_agg: if True, assume that Z_tilde, P_index, P_ldgm and annot_mat are all lists
    -- keep_init_0: if True, will only randomize tau and keep the initial value of phi unchanged

    Output:
    -- estimate of params from MLE
    -- number of steps (optimizations taken)
    -- a history of the likelihoods across iterations
    -- a history of the time elapsed for each optimization

    '''

    h2_valid = False
    random_start = params_init.copy()
    it_fail = 1

    # if optimization fails, change the initial values of tau

    while not h2_valid and it_fail <= 5:
        try:
            params_est, steps, ll_list, time_list = MLE_optim_annot(args, neg_logL_Fn_ldgm_annot, grad_Fn_ldgm_annot, hess_Fn_approx_ldgm_annot, Z_tilde, P_index, P_ldgm, n, annot_mat, params_0=random_start, optim_method=args.optim_method, block_agg=block_agg)

            logging.info("The path of neg-ll is: {}".format(ll_list))

            h2_valid = True

        except Exception as e:
            logging.error(e, exc_info=True)
            it_fail += 1
            if keep_init_0:
                random_start = np.random.uniform(low=1, high=10, size=(params_init.shape[0]-1, ))
                random_start = np.insert(random_start, 0, params_init[0])
            else:
                random_start = np.random.uniform(low=1, high=10, size=(params_init.shape[0], ))

            logging.info("Unsuccessful MLE -- Changing the initial value")
            logging.info("New starting value: {}".format(random_start))
            
    if not h2_valid:
        logging.info("Max of 5 attempts has reached to change the initial parameters. \nMLE still cannot be estimated.")
    else:
        logging.info("Number of updates taken: {}".format(steps))
        logging.info("{} attempts to change h2_0 before MLE converges".format(it_fail))

    return params_est, steps, ll_list, time_list

def run_MLE_NR_annot(args, Z_tilde, P_index, P_ldgm, annot_mat, n, params_init, block_agg=False, keep_init_0=False):

    '''
    Wrapper function for running the MLE to estimate the partitioned heritability.
    MLE is found using the minimize wrapper function.
    Main goal of this wrapper is to reinitialize the starting values of tau if the random start is bad.

    Input: 
    -- Z_tilde: Z-statistics after the precision matrix has been applied
    -- P_index: index of the precision matrix which map to the sumstats
    -- P_ldgm: LDGM precision matrix
    -- annot_mat: annotation matrix
    -- n: sample size
    -- params_init: initial parameter values
    -- block_agg: if True, assume that Z_tilde, P_index, P_ldgm and annot_mat are all lists

    Output:
    -- estimate of params from MLE
    -- number of steps (optimizations taken)
    -- a history of the likelihoods across iterations
    -- a history of the time elapsed for each optimization

    '''

    h2_valid = False
    random_start = params_init.copy()
    it_fail = 1

    # if optimization fails, change the initial values of tau

    while not h2_valid and it_fail <= 5:
        try:
            params_est, steps, ll_list, time_list = MLE_NR_annot(args, neg_logL_Fn_ldgm_annot, grad_Fn_ldgm_annot, hess_Fn_approx_ldgm_annot, Z_tilde, P_index, P_ldgm, n, annot_mat, params_0=random_start, block_agg=block_agg)

            logging.info("The path of neg-ll is: {}".format(ll_list))

            h2_valid = True

        except Exception as e:
            logging.error(e, exc_info=True)
            it_fail += 1
            if keep_init_0:
                random_start = np.random.uniform(low=1, high=10, size=(params_init.shape[0]-1, ))
                random_start = np.insert(random_start, 0, params_init[0])
            else:
                random_start = np.random.uniform(low=1, high=10, size=(params_init.shape[0], ))
            logging.info("Unsuccessful MLE -- Changing the initial value")
            logging.info("New starting value: {}".format(random_start))

    if not h2_valid:
        logging.info("Max of 5 attempts has reached to change the initial parameters. \nMLE still cannot be estimated.")
    else:
        logging.info("Number of updates taken: {}".format(steps))
        logging.info("{} attempts to change h2_0 before MLE converges".format(it_fail))

    return params_est, steps, ll_list, time_list


def MoM_naive(Z, P, n, p): 
    h2 = np.mean(np.square(Z)) / n * p - np.trace(P) / n
    return h2

def Mah_naive(Z, P, n, p):
    h2 = (np.dot(Z, np.einsum('i,ij->j', Z, P)) - p) / (n*p)
    return h2

#### LDGM codes for testing
def sliceMatrix(A, l1, l2):
    A1 = A[l1,:]
    A12= A1[:,l2]
    return A12

class ldgmMatrix:

    def __init__(self, edge_list_path = None, snp_list_path = None):
        
        self.matrix = None
        self.name = None
        self.snps = None
        self.nz_index = None
        
        if edge_list_path is not None:
            
            # load edge_list and snp_list - make sure they are for the same set of SNPs
            edge_list = pd.read_csv(edge_list_path, header =  None)
            
            assert(all(edge_list[0] <= edge_list[1]))
            # make matrix symmetric by adding (j,i,entry) for every edge (i,j,entry) where i<j
            edge_list2 = edge_list[[1,0,2]]
            edge_list2 = edge_list2[edge_list[0] < edge_list[1]]
            edge_list2.rename(columns = {1:0, 0:1}, inplace = True)
            edge_list = pd.concat((edge_list,edge_list2),axis=0)
            
            # compressed sparse column matrix
            self.matrix = csc_matrix( (edge_list[2], (edge_list[0], edge_list[1])) )
                    
            self.name = edge_list_path
            
            Anz = self.matrix!=0
            self.nz_index = Anz * np.ones(self.matrix.shape[0]) != 0
            self.num_edges = Anz.shape[0]

            # DEBUG
            logging.info("Number of edges (incl. those with zero on the diag): {}".format(self.num_edges))
            logging.info("Number of edges (excl. those with zero on the diag): {}".format(np.sum(self.nz_index)))

        if snp_list_path is not None:
            # SNP list
            snp_list = pd.read_csv(snp_list_path, sep = ',')

            # DEBUG:
            logging.info("Number of SNPs in the snp list: {}".format(snp_list.shape[0]))

            assert "index" in snp_list.columns, "SNP list should have a column named 'index'"
            self.snps = snp_list
        
    def multiply(self, y, whichIndices):
        if np.ndim(y)==1:
            y = np.reshape(y,(y.shape[0],1))
        
        assert all(self.nz_index[whichIndices]), "Matrix should have nonzero diagonal entries for all nonmissing SNPs"
        # handle indices not in precision matrix (i.e. all-zeros columns)

        # # DEBUG
        # logging.info(whichIndices.shape)
        # logging.info(self.nz_index.shape)

        otherIndices = np.logical_and (np.logical_not(whichIndices), self.nz_index)
        
        # submatrices of A == self.matrix
        A_00 = sliceMatrix(self.matrix, otherIndices, otherIndices)
        factor = cholesky(A_00)
        A_11 = sliceMatrix(self.matrix, whichIndices, whichIndices) 
        A_01 = sliceMatrix(self.matrix, otherIndices, whichIndices)
        
        # x == (A/A_00) * y
        z = factor(A_01 * y)
        x = A_11 * y  - np.transpose(A_01) * z
        return x
  
    def divide(self, y, whichIndices):
        assert all(self.nz_index[whichIndices]), "Matrix should have nonzero diagonal entries for all nonmissing SNPs"
        
        if np.ndim(y)==1:
            y = np.reshape(y,(y.shape[0],1))
         
        #diagonal elements should not be zero
        assert np.all(self.nz_index[whichIndices])
        
        # # DEBUG
        # logging.info(whichIndices.shape)
        # logging.info(self.nz_index.shape)

        #yp is y augmented with zeros           
        yp = np.zeros((self.matrix.shape[0],y.shape[1]), dtype=float, order='C')
        yp[whichIndices, :] = y

        #xp is x augmented with entries that can be ignored
        xp = np.zeros_like(yp)
        A_11 = sliceMatrix(self.matrix, self.nz_index, self.nz_index)
        factor = cholesky(A_11)

        xp[self.nz_index, :] = factor(yp[self.nz_index, :])
        x = xp[whichIndices, :]
        return x

    def root_divide(self, y, whichIndices):
        '''
        Compute A_11^{-1/2} y.
        '''
        assert all(self.nz_index[whichIndices]), "Matrix should have nonzero diagonal entries for all nonmissing SNPs"
        
        if np.ndim(y)==1:
            y = np.reshape(y,(y.shape[0],1))
         
        #diagonal elements should not be zero
        assert np.all(self.nz_index[whichIndices])
        
        # # DEBUG
        # logging.info(whichIndices.shape)
        # logging.info(self.nz_index.shape)

        #yp is y augmented with zeros           
        yp = np.zeros((self.matrix.shape[0],y.shape[1]), dtype=float, order='C')
        yp[whichIndices, :] = y

        #xp is x augmented with entries that can be ignored
        xp = np.zeros_like(yp)
        A_11 = sliceMatrix(self.matrix, self.nz_index, self.nz_index)
        factor = cholesky(A_11, ordering_method = "natural")

        xp[self.nz_index, :] = factor.solve_Lt(yp[self.nz_index, :], use_LDLt_decomposition = False)

        # # check whether permutation is accounted for in the factorization
        # factor_check = cholesky(A_11)
        # aa = factor_check.solve_Lt(yp[self.nz_index, :], use_LDLt_decomposition = False)
        # logging.info(aa)

        x = xp[whichIndices, :]

        return x


## Argument parsers
parser = argparse.ArgumentParser(description="\n Summary statistics and LD based variance component analyses")

## input and output file paths
IOfile = parser.add_argument_group(title="Input and output options")
IOfile.add_argument('--LDGM_edgelist_dir', default=None, type=str, 
    help='File path to the edgelist of the precision matrix.')
IOfile.add_argument('--LDGM_snplist_dir', default=None, type=str, 
    help='File path to the snplist of the precision matrix.')
IOfile.add_argument('--annot_dir', default=None, type=str, 
    help='File path to the annotation matrix. Will be merged with the LDGM snplist.')
IOfile.add_argument('--ldscore_dir', default=None, type=str, 
    help='File path to the LD score file (by default calc from matlab codes). Will be merged with the non-empty annotations')
IOfile.add_argument('--sumstats_fp', default=None, type=str, 
    help='File path to the pre-calculated summary statistics.')
IOfile.add_argument('--output_fp', default=None, type=str, 
    help='File path to save the results files')

IOfile.add_argument('--N', default=100, type=int, help='GWAS sample size.')
IOfile.add_argument('--N_ref', default=1000, type=int, help='LD reference panel sample size.')
# IOfile.add_argument('--M', default=1000, type=int, help='Number of markers.')
IOfile.add_argument('--stream-stdout', default=False, action="store_true", help='Stream log information on console in addition to writing to log file.')

## Simulating summary statistics
simul = parser.add_argument_group(title="Simulation options")
# simul.add_argument('--h2_grid_size', default=100, type=int, 
#     help='Number of h2 values to try and compute likelihood.')
simul.add_argument('--num_exp', default=10, type=int, 
    help='Number of experiments to calculate operating statistics.')
simul.add_argument('--pheno_index', default=None, type=int, 
    help='Index of the summary statistics. Used for phenotype indexing in the sumstats.')
simul.add_argument('--true_h2', default=0.25, type=float, 
    help='True value of heritability.')
simul.add_argument('--sparsity', default=1, type=float, 
    help='Propotion of non-zero SNPs (used when D is GRE).')
simul.add_argument('--seed', default=7621, type=int, help='Random seed to replicate simulated data.')
simul.add_argument('--D_model', default="GCTA", type=str, 
    help='Structure of the true covariance matrix. Options include GCTA, SpikeSlab, LDAK and other.')
simul.add_argument('--componentWeight', default="1", type=str, 
    help='The propotion of the normal mixtures used to generate the causal effect sizes.')
simul.add_argument('--componentVariance', default="1", type=str, 
    help='The variance of the mixture components used to generate the causal effect sizes.')
simul.add_argument('--kernel', default="Euc", type=str, 
    help='What kernel is used for estimation. The generative model does not involve the kernel. Specify the true genetic architecture via the mixture weights and component variances.')
simul.add_argument('--gen_link_type', default=None, type=str, 
    help='Link function used in the generative model to map annotation to per-SNP heritability. Current options include: max, exp, softmax and sigmoid.')

## Estimating the MLE
est = parser.add_argument_group(title="Estimation options")
est.add_argument('--est_link_type', default=None, type=str, 
    help='Link function used in the estimation model to map annotation to per-SNP heritability. Current options include: max, exp, softmax and sigmoid.')
est.add_argument('--no_baseline', default=False, action="store_true", help='Whether or not to keep the baseline annotation, either for simulating the sumstats, or for estimation, in which case the coefficient vector should not have an intercept term.')
est.add_argument('--per_snp_param', default=False, type=str, help='Specifies how the per-SNP h2 is parametrized. Options include: normalized, scaled and link_only.')
est.add_argument('--mle_method', default="manual", type=str, help='Whether to use the optim() wrapper in Python for finding the MLE or to manually update the parameters via NR.')

est.add_argument('--trace_method', default="MC", type=str, 
    help='Whether to use approximation for computing the trace, and what method to use for approximating the trace.')
est.add_argument('--stoch_samples', default=20, type=int, 
    help='Number of samples to draw in approximating the trace. This is used when trace_method is not exact and is approximate.')
est.add_argument('--hessian_method', default="exact", type=str, 
    help='Method to use for computing the Hessian matrix.')
est.add_argument('--optim_method', default="TNC", type=str, 
    help='Optimization method used in the wrapper function')
est.add_argument('--maxIter', default=20, type=int, 
    help='Maximum number of iterations for each estimation. Default value is 20.')
# est.add_argument('--eigval_range', default="0,10", type=str, 
#     help='Min and max value of eigenvalues. Used to select eigenvectors of the LD.')


## Operators
if __name__ == '__main__':
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d %I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler()) # logging.infos to console

    start_time = time.time()

    try:
        # Parse parameters
        n = int(args.N)
        total_h2 = float(args.true_h2)
        # D_model = args.D_model
        # h2_len_grid = int(args.h2_grid_size)
        num_exp = int(args.num_exp)
        # sparsity= float(args.sparsity)
        componentWeight = [float(x) for x in args.componentWeight.split(",")]
        componentVariance = [float(x) for x in args.componentVariance.split(",")]

        # only used when simulating I 
        n_ref = int(args.N_ref)

        #### read and align multiple blocks
        logging.info("Adopting the file prefixes in the directory: {}".format(args.LDGM_edgelist_dir))
        block_names = os.listdir(args.LDGM_edgelist_dir)
        block_names.sort()
        block_names = [block_names[i][:-9] for i in range(len(block_names))] # remove ".edgelist"
        logging.info("Block names: {}".format(block_names))

        P_ldgm_list = []
        annot_mat_list = []
        P_index_list = []
        df_sumstats_list = []
        df_annot_list = []
        df_ldscore_list = []

        for b in range(len(block_names)):
            logging.info("Processing block {}...".format(b))
            P_ldgm, annot_mat, P_index, df_annot, df_ldscore, df_sumstats = align_annot_l2_ldgm(args.LDGM_edgelist_dir + "/" + block_names[b] + ".edgelist", args.LDGM_snplist_dir + "/" + block_names[b] + ".snplist", args.annot_dir + "/" + block_names[b], args.ldscore_dir + "/" + block_names[b], use_annot = True)

            P_ldgm_list.append(P_ldgm)
            annot_mat_list.append(annot_mat)
            P_index_list.append(P_index)
            df_ldscore_list.append(df_ldscore)
            df_annot_list.append(df_annot)
            df_sumstats_list.append(df_sumstats)

        num_blocks = len(annot_mat_list)

        # modify the annotation matrix for simul or estimation
        if args.no_baseline:            
            annot_mat_list = [annot_mat_list[i][:, 1:] for i in range(len(P_ldgm_list))]
        # elif args.per_snp_param != "link_only":
        #     raise IOError("Incompatible flags specified -- if the linear coef does not include an intercept term, need to set --no_baseline.")

        # If we simulate the Z stats without including a baseline annot, but still normalize the effect sizes such that the total h2 is correct, is there an easy way to back out the implied true tau coefficient for the baseline annotation? 

        ### DEBUG begins: 
        # noAnnot = 1
        # annot_names = ['base']
        ### DEBUG ends: 

        p_list = [annot_mat_list[i].shape[0] for i in range(num_blocks)]
        p = np.sum(p_list)
        logging.info("Total number of SNPs after aligning annot and ldgm files: {}".format(p))
        noAnnot = annot_mat_list[0].shape[1]
        annot_names = df_annot_list[0].columns[-noAnnot:]
        logging.info("Annotation names involved in the analysis: {}".format(annot_names))

        # simulate Z statistics or run estimation
        if args.sumstats_fp is None:
            annot_mat = np.concatenate(annot_mat_list, axis=0)

            # set up the true coefficients
            # true_tau_vec = np.random.uniform(0, 20, size=noAnnot)
            # true_tau_vec = np.asarray([1])
            true_tau_vec = np.asarray([1,7,0,5,0,1,1,0,2,-1,2,1,0,3,0,4,0,2])
            logging.info("True coefficient values: {}".format(true_tau_vec))
            logging.info("Simulating Z statistics using the {} link".format(args.gen_link_type))

            # save the true coefficients to file, but convert to enrich at the estimation step
            np.savetxt(args.output_fp + "_tau.txt", true_tau_vec)

            # with open(args.output_fp + "_tau.txt", 'w') as file:
            #     file.write('\t'.join([str(x) for x in list(tau_vec)]))

            for i in np.arange(100,110):
                # simulate Z stat block-wise and then combine together
                Z_direct_list = simul_Z_ldgm_annot_block(n, total_h2, P_index_list, P_ldgm_list, annot_mat_list, true_tau_vec, args.gen_link_type, componentWeight, componentVariance)
                Z_direct = np.concatenate(Z_direct_list, axis=0)
                logging.info("Successfully simulated annotated Z-statistics!")

                # format sumstats 
                df_sumstats = pd.concat(df_sumstats_list)
                df_sumstats['Z'] = Z_direct
                df_sumstats['N'] = n

                # DEBUG
                logging.info("Mean chi^2: {}".format(np.mean(np.square(Z_direct))))
                df_ldscore = pd.concat(df_ldscore_list)
                logging.info("Mean LD score: {}".format(np.mean(df_ldscore['baseL2'])))

                # save sumstats to files (for LDSC)
                df_sumstats.to_csv(args.output_fp + "_sumstats_{}.txt".format(str(i)), index = False, header = True, sep = '\t')

        else:
            # set up the dataframe to record estimates
            if args.per_snp_param != "link_only":
                df_est = pd.DataFrame(columns = ['exp_ID'] + ["total_h2"] + list(annot_names))
                df_se = pd.DataFrame(columns = ['exp_ID'] + ["total_h2"] + list(annot_names))
            else:
                df_est = pd.DataFrame(columns = ['exp_ID'] + list(annot_names))
                df_se = pd.DataFrame(columns = ['exp_ID'] + list(annot_names))

            # record the true estimates
            true_tau_vec = np.loadtxt(args.sumstats_fp + "_tau.txt", dtype=float)
            # true_tau_params = np.insert(true_tau_vec, 0, total_h2)
            true_tau_params = true_tau_vec.copy()
            # true_tau_params = np.asarray([math.log(math.exp(total_h2 / p)-1)])
            true_total_h2, true_h2_annot, true_enrich = params_to_estimates(annot_mat_list, true_tau_params, args.est_link_type, per_snp_param = args.per_snp_param, block_agg = True)

            df_est.loc[len(df_est)] = ['True'] + [total_h2] + list(true_enrich)
            logging.info("True enrichment: {}".format(true_enrich))

            # for i in np.arange(100):

            exp_id = args.pheno_index
            # # read in sumstats and true tau from files
            # df_ss = pd.read_csv(args.sumstats_fp + '_sumstats_{}.txt'.format(str(exp_id)), index_col = None, delim_whitespace = True)
            # Z_direct = df_ss['Z'].values

            ### DEBUG code begins: read in sumstats from Luke's codes
            df_ss = pd.read_csv('/n/holystore01/LABS/xlin/Lab/huili/h2_ldgm/simul_output/LDGM/sumstats_LDSC/100000_softmax_inf/matlab/simul_{}_ldgm.txt'.format(exp_id), index_col = None, delim_whitespace = True)
            df_ss_block = [df_ss.loc[df_ss['block']==b] for b in range(num_blocks)]

            # subset to the SNPs with available annot (WARNING: needed due to different SNPs in the simulated sumstats and annot matrix)
            # for b in range(num_blocks):
            #     df_ss_block[b] = pd.merge(df_annot_list[b][['site_ids']], df_ss_block[b], left_on = 'site_ids', right_on = 'SNP')

            Z_direct_list = [df_ss_block[b]['Z_deriv_allele'].values for b in range(num_blocks)]
            logging.info("Checking Z stat reading, first block: {}".format(Z_direct_list[0]))
            P_index_nn_l = [df_ss_block[b]['index'].values for b in range(num_blocks)]
            P_index_list = [np.full((P_ldgm_list[i].num_edges, ), False) for i in range(num_blocks)]
            for b in range(num_blocks):
                P_index_list[b][P_index_nn_l[b]] = True

            logging.info("Checking the P indexes, first block: {}".format(P_index_nn_l[0]))
            logging.info("No. of snps from the annotation file: {}".format(p_list))
            p_list = [df_ss_block[b].shape[0] for b in range(num_blocks)]
            logging.info("No. of snps in the sumstats: {}".format(p_list))
            ### DEBUG code ends: read in sumstats from Luke's codes

            # # format the Z stats into list of block-sized stats
            # p_list = [annot_mat_list[i].shape[0] for i in range(len(P_ldgm_list))]
            # block_indexes = np.cumsum(p_list)
            # block_indexes = np.insert(block_indexes, 0, 0)
            # # logging.info("Indexes for the blocks: {}".format(block_indexes))

            # Z_direct_list = [Z_direct[block_indexes[i]:block_indexes[i+1]] for i in range(len(P_ldgm_list))]

            # transform Z into Z_tilde
            Z_tilde_list = [P_ldgm_list[b].multiply(Z_direct_list[b], P_index_list[b]) for b in range(len(P_ldgm_list))]

            #### RUN: MLE with multiple annotations
            noAnnot = annot_mat_list[0].shape[1]

            ### DEBUG code begins: manual computation of gradient
            annot_mat = np.concatenate(annot_mat_list, axis=0)
            # true_phi = total_h2 / np.sum(linkFn(annot_mat, true_tau_vec, args.est_link_type))
            # logging.info("Value of true phi: {}".format(true_phi))
            # params_true = np.insert(true_tau_vec, 0, true_phi)

            neg_logL_true = np.sum([neg_logL_Fn_ldgm_annot(true_tau_vec, Z_tilde_list[i], P_index_list[i], P_ldgm_list[i], n, annot_mat_list[i], args.est_link_type, per_snp_param=args.per_snp_param) for i in range(num_blocks)])

            grad_true = np.sum([grad_Fn_ldgm_annot(true_tau_vec, Z_tilde_list[i], P_index_list[i], P_ldgm_list[i], n, annot_mat_list[i], args.est_link_type, per_snp_param=args.per_snp_param, trace_method=args.trace_method) for i in range(num_blocks)], axis = 0)

            params_dl_above = np.insert(true_tau_vec[1:], 0, true_tau_vec[0]+0.001)
            neg_logL_above = np.sum([neg_logL_Fn_ldgm_annot(params_dl_above, Z_tilde_list[i], P_index_list[i], P_ldgm_list[i], n, annot_mat_list[i], args.est_link_type, per_snp_param=args.per_snp_param) for i in range(num_blocks)])

            params_dl_below = np.insert(true_tau_vec[1:], 0, true_tau_vec[0]-0.001)
            neg_logL_below = np.sum([neg_logL_Fn_ldgm_annot(params_dl_below, Z_tilde_list[i], P_index_list[i], P_ldgm_list[i], n, annot_mat_list[i], args.est_link_type, per_snp_param=args.per_snp_param) for i in range(num_blocks)])

            grad_manual_above = (neg_logL_above - neg_logL_true) / 0.001
            grad_manual_below = (neg_logL_true - neg_logL_below) / 0.001
            logging.info("Gradient at the true value is: {}".format(grad_true[0]))
            logging.info("Manual gradient from above: {}".format(grad_manual_above))
            logging.info("Manual gradient from below: {}".format(grad_manual_below))

            sys.exit()

            ### DEBUG code ends: manual computation of gradient

            # initialize the parameters
            if args.per_snp_param == "link_only":
                params_init = np.insert(np.zeros((noAnnot-1, )), 0, math.log(math.exp(1 / p)-1))
                # 1e-3*np.random.normal(size = (noAnnot-1,))
                # params_init = params_init + np.random.normal(size = noAnnot)
            elif args.per_snp_param == "normalized":
                tau_init = np.random.uniform(low=0, high=10, size=(noAnnot, ))
                params_init = np.insert(tau_init, 0, total_h2)

            elif args.per_snp_param == "scaled":
                tau_init = np.random.uniform(low=0, high=10, size=(noAnnot, ))
                annot_mat = np.concatenate(annot_mat_list, axis=0)
                true_phi = total_h2 / np.sum(linkFn(annot_mat, true_tau_vec, args.est_link_type))
                params_init = np.insert(tau_init, 0, true_phi)

            logging.info("Initialized parameter values: {}".format(params_init))
            logging.info("Estimating the MLE using the {} link".format(args.est_link_type))

            # run MLE, with option for different methods
            if args.mle_method == "manual":
                params_est, steps, ll_list, time_list = run_MLE_NR_annot(args, Z_tilde_list, P_index_list, P_ldgm_list, annot_mat_list, n, params_init, block_agg=True, keep_init_0=True)
            elif args.mle_method == "auto":
                params_est, steps, ll_list, time_list = run_MLE_optim_annot(args, Z_tilde_list, P_index_list, P_ldgm_list, annot_mat_list, n, params_init, block_agg=True, keep_init_0=True)
                
            # record the h2 estimates 
            est_total_h2, est_h2_annot, est_enrich = params_to_estimates(annot_mat_list, params_est, args.est_link_type, per_snp_param = args.per_snp_param, block_agg = True)

            logging.info("Estimated enrichment values: {}".format(est_enrich))
            logging.info("Estimated partitioned h2: {}".format(est_h2_annot))
            logging.info("Estimated total h2: {}".format(est_total_h2))

            logging.info("Number of iterations taken: {}".format(steps))
            df_est.loc[len(df_est)] = [exp_id] + [est_total_h2] + list(est_enrich)

            # record the analytical SE
            params_SE = calc_params_SE(params_est, Z_tilde_list, P_index_list, P_ldgm_list, n, annot_mat_list, args.est_link_type, hess_Fn_approx_ldgm_annot, per_snp_param = args.per_snp_param, block_agg = True)
            logging.info("Analytic standard errors of total h2 and enrich are: {}".format(params_SE))
            df_se.loc[len(df_se)] = [exp_id] + list(params_SE)

            # write the estimates to files
            df_est.to_csv(args.output_fp+"_ldgm.txt", index = False, header = True, sep = '\t')
            df_se.to_csv(args.output_fp + "_ldgm_se.txt", index = False, header = True, sep = '\t')

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
