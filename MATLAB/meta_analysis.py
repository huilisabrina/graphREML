#!/usr/bin/python

#----------------------------------------------------------
# Perform meta-analyses using the saved jackknife results
# -- General purpose: for enrichment values

# Required input:
# -- *_est.txt from graphREML estimates
# -- *_h2JK.txt the saved jackknife estimates
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
parser = argparse.ArgumentParser(description="\n Meta-analysis of enrichment from graphREML")
parser.add_argument('--est_fp', default=None, type=str, 
    help='File prefixes to the estimate files, delimited by comma for multiple traits.')
parser.add_argument('--params_idx', default="2,3,4,5,6,7,8,9,10", type=str, 
    help='Indices of the annotations to be meta-analyzed, delimited by comma.')
parser.add_argument('--num_blocks', default=3, type=int, 
    help='Number of jackknife estimates of LD blocks.')
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
        h2JK_list = [x + '_h2JK.txt' for x in file_list]
        pheno_list = [os.path.basename(x) for x in file_list]
        annot_idx = [int(x) for x in args.params_idx.split(',')]

        B = int(args.num_blocks)
        T = len(file_list)
        K = len(annot_idx)

        # read in h2 estimates (by trait -- annotation -- block)
        logging.info("\nTraits analyzed: {}".format(pheno_list))
        logging.info("Meta-analyzing across {} traits using {} jackknife replicates for {} enrichment".format(T, B, K))

        enrich_num = np.zeros((T, K, B))
        enrich_denom = np.zeros((T, B))

        for i in range(T):
            # read the enrichment estimates
            df_est = pd.read_csv(est_list[i], delim_whitespace = False, sep=',', index_col = None)
            annotNames = df_est.loc[annot_idx, 'annotName']
            p_annot = np.array(df_est.loc[annot_idx, 'size']) # in proportion

            # read the jackknife estimates
            mat_h2_jk = pd.read_csv(h2JK_list[i], delim_whitespace = False, sep=',', index_col = None).to_numpy()
            mat_h2 = mat_h2_jk[0, :]
            mat_h2_annot = mat_h2_jk[np.array(annot_idx), :]

            # num before meta-analysis
            enrich_num[i, :, :] = np.einsum('kb,k->kb', mat_h2_annot, 1/p_annot)
            enrich_denom[i, :] = mat_h2

        # meta analyses
        logging.info("Performing meta-analyses across jackknife estimates")

        enrich_meta = np.mean(enrich_num, axis = 0) / np.mean(enrich_denom, axis = 0) #KxB
        enrich_mean = np.mean(enrich_meta, axis = 1) #Kx1
        enrich_SE = np.std(enrich_meta, axis = 1) / math.sqrt(B) #Kx1

        pd_out = pd.DataFrame({"Annot": annotNames, "meta_enrich": enrich_mean, "SE_enrich": enrich_SE})
        logging.info(pd_out)

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Meta-analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
