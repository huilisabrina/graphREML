#!/usr/bin/python

#-------------------------------------------------------
# Score test to evaluate the significance of 
# new annotations

# Note: this function requires a suite of three output 
# files from graphREML estimation. 
# 1) pre-computed SNP-level of gradient scores and hessian (*_perSNPh2.txt)
# this file is saved when specifying nullFit as True in run_graphREML
# 2) jackknifed estimates (*_paramsJK.txt)
# 3) estimates table (*_est.txt)

#-------------------------------------------------------
import os, sys, re
import logging, time, traceback
import argparse
from functools import reduce
import pickle

import pandas as pd
import numpy as np
import numpy.matlib
from scipy import stats
import random
import math

import statsmodels.api as sm

def valid_float(y):
  try:
    return float(y)
  except ValueError:
    return np.nan

def valid_int(y):
  try:
    return int(y)
  except ValueError:
    return np.nan

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

# standardize LD (cov --> correlation)
def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


## Argument parsers
parser = argparse.ArgumentParser(description="\n Score test for hypothesis testing of new annotation or functional category")
parser.add_argument('--params_fp', default=None, type=str, 
    help='File path to the parameter estimates under the null model.')
parser.add_argument('--jackknife_fp', default=None, type=str, 
    help='File path to the jackknifed parameter estimates, stored in a table.')
parser.add_argument('--snpGrad_fp', default=None, type=str, 
    help='File path to the variant-level summary statistics of gradient scores.')
parser.add_argument('--annot_fp', default=None, type=str, 
    help='File path to the variant-level annotation matrix for a set of annotations.')
parser.add_argument('--output_fp', default=None, type=str, 
    help='Output prefix name. Unless otherwise specified, a single test output will be produced for all of the annotations specified.')
parser.add_argument('--annotations', default=None, type=str, 
    help='The set of annotations to run score test on, specified as a string, separated by comma.')
parser.add_argument('--adjust_score', default=False, action="store_true", help='Whether or not to acccount for the uncertainty in the estimates of parameters.')
parser.add_argument('--joint_test', default=False, action="store_true", help='Whether or not to joint test the new annotations, as opposed to performing marginal tests for each new annot.')
parser.add_argument('--stream-stdout', default=False, action="store_true", help='Stream log information on console in addition to writing to log file.')


if __name__ == '__main__':

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO, datefmt='%Y/%m/%d %I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler()) # prints to console
    start_time = time.time()

    try:
        logging.info("Reading parameter estimates from fitting the baseline model (Null model)")
        df_est = pd.read_csv(args.params_fp, delim_whitespace = False, sep=',', index_col = None)

        # remove large effect from the score test, if present
        if 'largeEffect' in list(df_est['annotName'].values):
            null_est = df_est['params'].values[:-1]
            annot_null_list = [x for x in list(df_est['annotName'].values) if x not in ['largeEffect']]
            jk_annot = pd.read_csv(args.jackknife_fp, delim_whitespace = False, sep=',', index_col = None).values[:-1]
        else:
            null_est = df_est['params'].values
            annot_null_list = list(df_est['annotName'].values)
            jk_annot = pd.read_csv(args.jackknife_fp, delim_whitespace = False, sep=',', index_col = None).values

        logging.info("Null annot")
        logging.info(annot_null_list)
        
        # noBlocks = df_jackknife.shape[2]
        # df_jackknife.rename(columns={"Var{}".format(i+1): "block_{}".format(i+1) for i in range(noBlocks)}, inplace=True)
        # df_jackknife['param_name'] = annot_null_list

        logging.info("Reading SNP-specific gradients from fitting the null")
        df_snp = pd.read_csv(args.snpGrad_fp, delim_whitespace = False, sep=',',index_col = None)

        if args.adjust_score:
            df_snp.columns = ['perSNPh2', 'snpGrad', 'snpHess', 'block', 'rsID']
        else:
            df_snp.columns = ['perSNPh2', 'snpGrad', 'block', 'rsID']

        logging.info(df_snp.shape)
        logging.info(df_snp.head(5))

        logging.info("Reading the merged annotation files (all annotations)")
        df_annot = pd.read_csv(args.annot_fp, delim_whitespace = False, sep=',', index_col = None)
        logging.info(df_annot.shape)
        logging.info(df_annot.head(5))

        # list the annotations to be tested
        annot_test_list = args.annotations.split(',')
        # annot_test_list = ['Conserved_LindbladToh_common', 'Conserved_LindbladToh_flanking_500_common', 'Repressed_Hoffman_common', 'Repressed_Hoffman_flanking_500_common']
        
        # extract the original annotations (null annot)
        null_annot = df_annot[annot_null_list].values
        
        # merge SNP-specific scores with annotation (may be mismatched)
        if args.adjust_score:
            df_merged = df_snp[['rsID', 'snpGrad', 'snpHess', 'block']].merge(df_annot[['rsID'] + annot_test_list], on = 'rsID', how = 'inner')
        else:
            df_merged = df_snp[['rsID', 'snpGrad', 'block']].merge(df_annot[['rsID'] + annot_test_list], on = 'rsID', how = 'inner')
        logging.info(df_merged.columns)
        logging.info(df_merged.shape)

        # project out the null annotations
        proj_out_list = annot_null_list # if x not in ['Coding_UCSC_flanking_500_common', 'Conserved_LindbladToh_flanking_500_common']
        proj_annot = df_annot[proj_out_list].values
        mlr_fit = sm.OLS(df_merged['snpGrad'].values, proj_annot).fit()
        pred = mlr_fit.predict()
        resid = mlr_fit.resid
        df_merged['snpGrad_proj'] = resid
        
        logging.info(df_merged[['snpGrad', 'snpGrad_proj']].head(5))
        logging.info(np.mean(df_merged['snpGrad_proj'].values))
        logging.info(np.median(df_merged['snpGrad_proj'].values))

        #==================================
        # Test of single annotations
        #==================================
        # Define the annotations we want to test (one by one)
        chisq_list = list()
        chisq_pval_list = list()
        noBlocks = np.max(df_merged['block'])
        logging.info("Number of blocks to jackknife from: {}".format(noBlocks))

        for k in range(len(annot_test_list)):
            # obtain annot (k) specific vectors
            snpGrad = df_merged['snpGrad_proj'].values
            annot_vec = df_merged[annot_test_list[k]].values

            # compute U
            snpScore = np.multiply(snpGrad, annot_vec)
            sumscore = np.nansum(snpScore, axis = 0)

            # to account for the uncertainty in the parameter estimates
            if args.adjust_score:
                snpHess = df_merged['snpHess'].values
                aa = np.tile(null_est, (jk_annot.shape[1],1))
                logging.info(aa.shape)
                logging.info(aa[:5, :5])
                jk_diff = jk_annot - aa.T # K_null x B
                snpScoreGrad = np.einsum('j,jk->jk', np.multiply(annot_vec, snpHess), null_annot) # p x K_null
                sumScoreGrad = np.nansum(snpScoreGrad, axis = 0) # K_null

            # jackknife U by block
            blockScore = np.zeros((noBlocks, 1))

            for i in range(noBlocks):
                snpGrad = df_merged.loc[df_merged['block'] == i+1, 'snpGrad_proj'].values
                annot_vec = df_merged.loc[df_merged['block'] == i+1, annot_test_list[k]].values

                # deduct score contributed from one LD block
                blockScore[i] = sumscore - np.nansum(np.multiply(snpGrad, annot_vec), axis = 0)

                if args.adjust_score:
                    snpHess = df_merged.loc[df_merged['block'] == i+1, 'snpHess'].values
                    jackknife_index = np.where(df_merged['block'] == i+1)[0]
                    null_annot_jk = null_annot[jackknife_index, :]
                    snpScoreGrad = np.einsum('j,jk->jk', np.multiply(annot_vec, snpHess), null_annot_jk) # p_JK x K_null
                    
                    # deduct scoreGrad contributed from one LD block
                    blockScore[i] = blockScore[i] + jk_diff[:, i].dot(sumScoreGrad - np.nansum(snpScoreGrad, axis = 0))

            # perform chisq-test on the score statistics
            jack_var = np.var(blockScore)
            jack_mean = np.mean(blockScore)
            logging.info("Z statistic: {}".format(jack_mean / np.std(blockScore) / math.sqrt(noBlocks-2)))

            score_stat = jack_mean**2 / jack_var / (noBlocks-2)
            logging.info("Score test statistic: {}".format(score_stat))

            # perform chisq test
            logging.info("Performing chi2 test for annotation: {}".format(annot_test_list[k]))
            chisq_pval = 1 - stats.chi2.cdf(score_stat, 1)
            logging.info('P-value is: {}'.format(chisq_pval))

            # record the p-values
            chisq_list.append(score_stat)
            chisq_pval_list.append(chisq_pval)

        logging.info(annot_test_list)
        logging.info(chisq_list)
        logging.info(chisq_pval_list)
        
        df_score_test = pd.DataFrame({'annot': annot_test_list, 'chisq_stat': chisq_list, 'chisq_pval': chisq_pval_list})
        logging.info(df_score_test)
        df_score_test.to_csv(args.output_fp + ".txt", index=False, header=True, sep='\t')

        exit()

        if args.joint_test:
            #====================================
            # Joint test of multiple annotations
            #====================================
            snpGrad = df_merged['snpGrad'].values
            annot_mat = df_merged[annot_test_list].values

            # compute U
            snpScore = np.einsum('j,jk->jk', snpGrad, annot_mat)
            sumscore = np.nansum(snpScore, axis = 0) # Kx1

            # jackknife U by block
            blockScore = np.zeros((noBlocks, len(annot_test_list)))

            for i in range(noBlocks):
                snpGrad = df_merged.loc[df_merged['block'] == i+1., 'snpGrad'].values
                annot_mat = df_merged.loc[df_merged['block'] == i+1, annot_test_list].values

                # deduct score contributed from one LD block
                blockScore[i, :] = sumscore - np.nansum(np.einsum('j,jk->jk', snpGrad, annot_mat), axis = 0)

            # perform chisq teston U
            logging.info("Performing chi2 test with for {} annotations: {}".format(len(annot_test_list), ','.join(annot_test_list)))
            jack_cov = np.cov(blockScore.T)
            logging.info(jack_cov.shape)
            score_stat = sumscore.dot(np.linalg.inv(jack_cov).dot(sumscore)) / (noBlocks-2)
            chisq_pval =1 - stats.chi2.cdf(score_stat, len(annot_test_list))
            logging.info('P-value is: {}'.format(chisq_pval))

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
