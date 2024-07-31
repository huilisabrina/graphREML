#!/usr/bin/python

#----------------------------------------------------------
# Perform meta-analyses using the saved jackknife results
# -- AMM applications: compute pK values and tauA

# Required input:
# -- *_est.txt from graphREML estimates
# -- *_paramsJK.txt the saved jackknife estimates
#----------------------------------------------------------

# module load python/3.12.1
# # module load python3
# source activate graphREML

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

# define the robust softmax function
def softmax_robust(x):
    y = x + np.log(1 + np.exp(-x))
    y[x < 0] = np.log(1 + np.exp(x[x < 0]))

    return y

## Argument parsers
parser = argparse.ArgumentParser(description="\n Meta-analyses for pK estimates from jackknife estimates")
parser.add_argument('--est_fp', default=None, type=str, 
    help='File prefixes to the estimate files, delimited by comma for multiple traits.')
parser.add_argument('--params_idx', default="66,67,68,69,70,71", type=str, 
    help='Indices of the AMM parameters, delimited by comma.')
parser.add_argument('--ammbinvec', default="1,1,3,5,10,30", type=str, 
    help='Sizes of the bins of genes for estimation.')
parser.add_argument('--num_blocks', default=3, type=int, 
    help='Number of jackknife estimates. By default, this is the number of LD blocks.')
parser.add_argument('--output_fp', default="test", type=str, 
    help='File path to save the results')
parser.add_argument('--stream-stdout', default=False, action="store_true", help='Stream log information on console in addition to writing to log file.')


if __name__ == '__main__':

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO, datefmt='%Y/%m/%d %I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler()) # prints to console
    start_time = time.time()

    try:
        # parse file names and paths
        file_list = args.est_fp.split(',')
        est_list = [x + '_est.txt' for x in file_list]
        paramsJK_list = [x + '_paramsJK.txt' for x in file_list]
        pheno_list = [os.path.basename(x) for x in file_list]
        annot_idx = [int(x) for x in args.params_idx.split(',')]
        
        # parse AMM annot specific params
        ammbinvec = [float(x) for x in args.ammbinvec.split(',')]
        ck = np.array(ammbinvec)

        B = int(args.num_blocks)
        T = len(file_list)
        K = len(annot_idx)

        # read in jk estimates (by trait -- AMM annot -- block)
        logging.info("\nTraits analyzed: {}".format(pheno_list))
        logging.info("Meta-analyzing across {} traits using {} jackknife replicates for {} AMM annotations".format(T, B, K))

        pk_num = np.zeros((T, K, B))
        pk_denom = np.zeros((T, B))

        for i in range(T):
            # read the enrichment estimates
            df_est = pd.read_csv(est_list[i], delim_whitespace = False, sep=',', index_col = None)
            annotNames = df_est.loc[annot_idx, 'annotName']
            param_est = df_est['params']
            h2 = df_est['h2'][0]

            # read the jackknife estimates
            mat_params_jk = pd.read_csv(paramsJK_list[i], delim_whitespace = False, sep=',', index_col = None).to_numpy()
            mat_params_jk = softmax_robust(mat_params_jk)
            mat_AMM = mat_params_jk[np.array(annot_idx), :]

            # compute num and denom separately (weighted by h2)
            pk_num[i, :, :] = np.einsum('kb,k->kb', mat_AMM, ck) * h2
            pk_denom[i, :] = np.sum(pk_num[i, :, :], axis = 0)

        # meta analyses
        logging.info("Performing meta-analyses across jackknife estimates")
        pk_meta = np.mean(pk_num, axis = 0) / np.mean(pk_denom, axis = 0) #KxB
        tau_meta = np.mean(pk_denom, axis = 0) #Bx1

        pk_mean = np.mean(pk_meta, axis = 1) # Kx1
        pk_SE = np.std(pk_meta, axis = 1) / math.sqrt(B) # Kx1
        tau_mean = np.mean(tau_meta) # 1x1

        # report results
        pd_out = pd.DataFrame({"AMM annot": annotNames, "meta_pk": pk_mean, "SE_pk": pk_SE})
        logging.info(pd_out)
        logging.info("Average tauA value: {}".format(tau_mean))
        

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Meta-analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
