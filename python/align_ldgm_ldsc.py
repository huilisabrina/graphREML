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


###### FORMATTING FUNCTIONS AND I/O HELPERS #######
def align_annot_l2_ldgm(ldgm_edgelist_fp, ldgm_snplist_fp, annot_fp, ldscore_fp = None, no_empty_annot = False):
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
    df_annot_overlap.drop_duplicates(subset = ['index'], keep = 'first', inplace = True)
    df_annot_overlap.rename(columns = {'index': 'snp_index'}, inplace = True)

    # Collect the indexes of P_ldgm that have available annot info
    P_index = np.asarray(df_annot_overlap.loc[df_annot_overlap['_merge'] == "both", 'snp_index'])
    logging.info("Checking: Min P_index: {}; Max P_index: {}".format(min(P_index), max(P_index)))
    logging.info("# of SNPs indexed on the precision matrix that has annot info: {}".format((P_index.shape[0])))

    # Turn P_index -- which are numerical indexes -- into a vector of boolean values
    P_index_b = np.full((P_ldgm.num_edges, ), False)
    P_index_b[P_index] = True

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
    df_annot_overlap = df_annot_overlap.drop(['snp_index', 'site_ids', 'anc_alleles', 'deriv_alleles'], axis = 1)


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

        # save the counts for diff annotations as text-delimited file
        with open(ldscore_fp + ".l2.M", 'w') as file:
            file.write('\t'.join([str(x) for x in np.nansum(annot_mat, axis = 0)]))
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

    return P_ldgm, annot_mat, P_index_b, df_annot_overlap, df_ldscore, df_sumstats


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

            # # DEBUG
            # logging.info("Number of edges (incl. those with zero on the diag): {}".format(self.num_edges))
            # logging.info("Number of edges (excl. those with zero on the diag): {}".format(np.sum(self.nz_index)))

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
        
        #yp is y augmented with zeros           
        yp = np.zeros((self.matrix.shape[0],y.shape[1]), dtype=float, order='C')
        yp[whichIndices, :] = y

        #xp is x augmented with entries that can be ignored
        xp = np.zeros_like(yp)
        A_11 = sliceMatrix(self.matrix, self.nz_index, self.nz_index)
        factor = cholesky(A_11, ordering_method = "natural")

        xp[self.nz_index, :] = factor.solve_Lt(yp[self.nz_index, :], use_LDLt_decomposition = False)

        # # check whether permutation is accoutned for in the 
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
IOfile.add_argument('--output_fp', default=None, type=str, 
    help='File path to save the results files')


## Operators
if __name__ == '__main__':
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d %I:%M:%S %p')

    start_time = time.time()

    try:
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
            P_ldgm, annot_mat, P_index, df_annot, df_ldscore, df_sumstats = align_annot_l2_ldgm(args.LDGM_edgelist_dir + "/" + block_names[b] + ".edgelist", args.LDGM_snplist_dir + "/" + block_names[b] + ".snplist", args.annot_dir + "/" + block_names[b], args.ldscore_dir + "/" + block_names[b])

            P_ldgm_list.append(P_ldgm)
            annot_mat_list.append(annot_mat)
            P_index_list.append(P_index)
            df_ldscore_list.append(df_ldscore)
            df_annot_list.append(df_annot)
            df_sumstats_list.append(df_sumstats)

        #### files which have been AGGREGATED across blocks
        # LD scores
        df_ldscore_agg = pd.concat(df_ldscore_list)
        df_ldscore_agg.to_csv(args.output_fp + ".l2.ldscore", index = False, header = True, sep = '\t')
        
        # LD weights
        df_w_ld = df_ldscore_agg.loc[:, ["CHR", "SNP", "BP", "baseL2"]]
        df_w_ld.rename(columns = {'baseL2': 'L2'}, inplace = True)
        df_w_ld.to_csv(args.output_fp + ".w.l2.ldscore", index = False, header = True, sep = '\t')

        # annotation files
        df_annot_agg = pd.concat(df_annot_list)
        df_annot_agg.to_csv(args.output_fp + ".annot", index = False, header = True, sep = '\t')

        # M counts 
        annot_mat = np.concatenate(annot_mat_list, axis=0)
        with open(args.output_fp + ".l2.M", 'w') as file:
            file.write('\t'.join([str(x) for x in np.nansum(annot_mat, axis = 0)]))

        # save snp info columns for sumstats
        df_sumstats = pd.concat(df_sumstats_list)
        df_sumstats.to_csv(args.output_fp + ".snpinfo", index = False, header = True, sep = '\t')

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
