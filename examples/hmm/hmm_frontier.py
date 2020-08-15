# # HMM Views IC Frontier: Oldest Country ETFs

# ## 1. Data Fetching

# ### 1.1 Model configuration


import os
import sys
import datetime as dt
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import socket
from hmmlearn import hmm
#import cvxportfolio as cp
import alphamodel as am

config = {'name': 'bl_hmm_sim',
          'universe':
              {'list': ['SPY', 'EWA', 'EWC', 'EWG', 'EWH', 'EWJ', 'EWS', 'EWU', 'EWW'],
               'ticker_col': 'Symbol',
               'risk_free_symbol': 'USDOLLAR'},
          'data':
              {'name': 'eod_returns',
               'source': 'quandl',
               'table': 'EOD',
               'api_key': "6XyApK2BBj_MraQg2TMD"},
          'model':
              {'start_date': '19970102',
               'end_date': '20051231',
               'halflife': 20,
               'min_periods': 3,
               'hidden_states': 2,
               'train_len': 750,
               'data_dir': '/mnt/research_masc/data_store/hmm/',
               'returns':
                   {'sampling_freq': 'daily'},
               'covariance':
                    {'method' : 'FF5',
                     'sampling_freq' : 'monthly',
                     'train_days': 90}
              }
         }


host = socket.gethostname()
print('Running on {}'.format(host))

# ### 1.2 Fetch return data


# Fetch returns / volumes
ss = am.SingleStockHMM(config)
ss.train(force=True)


# ## 2. HMM information coefficient frontier

# Hyperparameters:
# - hmm_mode: return prediction, 'e' (expectation), 't' (regime over probability threshold)
# - preprocess: raw data or exponential decay
# - train_len: length of training data
# - halflife: halflife of exponential decay

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

ic_vs_params = {}

# for hmm_mode in ['e', 't']:
for hmm_mode in ['t']:
    for preprocess in ['None']:
        # for train_len in range(50, 1751, 50):
        for train_len in [1700]:
            for threshold in [0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]:
                # New run key                
                key = hmm_mode + str(preprocess) + str(train_len) + str(threshold)
                logging.warning('Running for hyperparams {}'.format(key))

                # Set the parameters for the prediction
                ss.cfg['train_len'] = train_len
                ss.cfg['threshold'] = threshold

                # Predict and gather metrics
                ss.predict(mode=hmm_mode, preprocess=preprocess, threshold=threshold)
                ic = ss.prediction_quality(statistic='information_coefficient', print=False)
                jitter = ss.prediction_quality(statistic='jitter', print=False)

                # Save down metrics together with parameters
                ic_vs_params[key] = [hmm_mode, preprocess, train_len, threshold,
                                    ic.loc[5, 'mean'], ic.loc[5, 'std'],
                                    jitter.loc['mean_changes', 'mean'], jitter.loc['mean_changes', 'std'],
                                    jitter.loc['mean_change_rate', 'mean'], jitter.loc['mean_change_rate', 'std'],
                                    jitter.loc['regime_changes', 'mean'], jitter.loc['regime_changes', 'std'],
                                    jitter.loc['regime_change_rate', 'mean'], jitter.loc['regime_change_rate', 'std'],
                                    jitter.loc['periods', 'mean'], jitter.loc['periods', 'std']]

                # Save down values in .csv
                ic_df = pd.DataFrame.from_dict(ic_vs_params, orient='index')
                ic_df.columns = ['hmm_mode', 'preprocess', 'train_len', 'threshold',
                                'ic_5d_mean', 'ic_5d_std',
                                'mean_changes_mean', 'mean_changes_std',
                                'mean_change_rate_mean', 'mean_change_rate_std',
                                'regime_changes_mean', 'regime_changes_std',
                                'regime_change_rate_mean', 'regime_change_rate_std',
                                'periods_mean', 'periods_std']
                ic_df.to_csv(ss.cfg['data_dir'] + 'hmm_ic_{}.csv'.format(host), index=False)

