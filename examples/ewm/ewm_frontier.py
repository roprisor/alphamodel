# # EWMA Risk Rewards Frontier: Oldest Country ETFs

# ## 1. Data Fetching

# ### 1.1 Model configuration

import datetime as dt
import logging
import os
import pandas as pd
import numpy as np
import socket
import time
import cvxportfolio as cp
import alphamodel as am

config = {'name': 'ewm_frontier',
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
               'end_date': '20200831',
               'halflife': 20,
               'min_periods': 3,
               'train_len': 1700,
               'data_dir': '/mnt/research_masc/data_store/spo/',
               'returns':
                   {'sampling_freq': 'daily'},
               'covariance':
                   {'method': 'FF5',
                    'sampling_freq': 'monthly',
                    'train_days': 360
                    }
               }
          }

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

host = socket.gethostname()
logging.warning('Running on {}'.format(host))

# ### 1.2 Fetch return data

# Fetch returns / volumes
ss = am.SingleStockEWM(config)
ss.train(force=True)

# Realized Data for Simulation
prices = ss.get('prices', 'realized', ss.cfg['returns']['sampling_freq']).iloc[1:, :]
returns = ss.get('returns', 'realized', ss.cfg['returns']['sampling_freq'])
volumes = ss.get('volumes', 'realized', ss.cfg['returns']['sampling_freq'])
sigmas = ss.get('sigmas', 'realized', ss.cfg['returns']['sampling_freq'])

simulated_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                               market_volumes=volumes, cash_key=ss.risk_free_symbol)

# ## 2. Black Litterman HMM Risk Rewards Frontier

# Aggregate market stats for cal
market_stats = pd.DataFrame({'MarketCap/GDP': [1.25, 1, 1.25, 0.45, 3.5, 0.8, 2, 1.25, 0.3, 0],
                             'GDP': [2543500, 150000, 239000, 853000, 22500, 1037500, 10000, 422500, 164500, 0]},
                            index=ss.universe + ['USDOLLAR'])
market_stats.loc[:, 'MarketCap'] = market_stats.loc[:, 'MarketCap/GDP'] * market_stats.loc[:, 'GDP']
market_stats.loc[:, 'MarketCap Weights'] = market_stats.loc[:, 'MarketCap'] / market_stats.loc[:, 'MarketCap'].sum()


# Generate market cap weights pandas.Series
w_mktcap = pd.Series(index=market_stats.index, data=market_stats.loc[:, 'MarketCap Weights'])
w_mktcap['USDOLLAR'] = 0.

# Start and end date
start_date = dt.datetime(2005, 1, 2)
end_date = dt.datetime.strptime(config['model']['end_date'], '%Y%m%d')
logging.warning('Start Date: {sd} - End Date: {ed}'.format(sd=start_date.strftime('%Y%m%d'),
                                                           ed=end_date.strftime('%Y%m%d')))

# Hyperparameters:
# - mode: type of covariance (SS - stock return direct, FF5 - Fama French 5 factor model)
# - gamma_risk: risk aversion coefficient
# - gamma_trade: turnover aversion coefficient

# Search parameters
risk_aversion = 2.5
confidence = 0.8
scns = 1
mode = ss.cfg['covariance']['method']
trading_freq = 'week'
gamma_risk = [0.001, 0.01, 0.1, 1, 10, 100]
gamma_trade = [1, 2, 3, 4, 5]
gamma_hold = 0.
total_runs = len(gamma_risk) * len(gamma_trade)

prtf_vs_params = {}
run = 1

# Predict
start_time = time.time()
ss.predict()
end_time = time.time()
logging.warning('Prediction complete, took {s} seconds'.format(s=str(end_time - start_time)))

for grisk in gamma_risk:
    for gtrd in gamma_trade:
        # New run key
        key = 'mode_' + mode + '_grisk_' + str(grisk) + '_gtrd_' + str(gtrd)
        prtf_vs_params = {}
        logging.warning('Running for hyperparams {}'.format(key))
        start_time = time.time()

        try:
            r_pred = ss.get('returns', 'predicted')
            volumes_pred = ss.get('volumes', 'predicted')
            sigmas_pred = ss.get('sigmas', 'predicted')

            # Predicted costs
            optimization_tcost = cp.TcostModel(half_spread=0.0005 / 2., nonlin_coeff=1.,
                                               sigma=sigmas_pred, volume=volumes_pred)
            optimization_hcost = cp.HcostModel(borrow_costs=0.0001)

            # Covariance setup
            if ss.cfg['covariance']['method'] == 'SS':
                risk_model = cp.FullSigma(ss.get('covariance', 'predicted'))
            elif ss.cfg['covariance']['method'] == 'FF5':
                risk_model = cp.FactorModelSigma(ss.get('exposures', 'predicted'),
                                                 ss.get('factor_sigma', 'predicted'),
                                                 ss.get('idyos', 'predicted'))
            else:
                raise NotImplemented('The %s risk model is not implemented yet'.format(ss.cfg['covariance']))

            # Black Litterman policy
            logging.warning('Running backtest')

            # Optimization parameters
            leverage_limit = cp.LeverageLimit(1)
            fully_invested = cp.ZeroCash()
            long_only = cp.LongOnly()

            # Optimization policy
            spo_policy = cp.SinglePeriodOpt(return_forecast=r_pred,
                                            costs=[grisk * risk_model,
                                                   gtrd * optimization_tcost,
                                                   gamma_hold * optimization_hcost],
                                            constraints=[leverage_limit, fully_invested, long_only],
                                            trading_freq=trading_freq)

            # Backtest
            spo_results = simulator.run_multiple_backtest(1E6 * w_mktcap, start_time=start_date, end_time=end_date,
                                                          policies=[spo_policy],
                                                          loglevel=logging.WARNING, parallel=True)
            result = spo_results[0]
            logging.warning(result.summary())

            # Save down metrics together with parameters
            prtf_vs_params[key] = [mode, scns, trading_freq, grisk, gtrd,
                                   result.excess_returns.mean() * 100 * result.ppy,
                                   result.excess_returns.std() * 100 * np.sqrt(result.ppy),
                                   result.max_drawdown * 100,
                                   result.turnover.mean() * 100 * result.ppy]
        except Exception as e:
            logging.error('Ran into an error: {e}'.format(e=str(e)))
            prtf_vs_params[key] = [mode, scns, trading_freq, grisk, gtrd, 0, 0, 100, 0]

        # Save down values in .csv
        file_path = ss.cfg['data_dir'] + 'spo_frontier_{}.csv'.format(host)
        prtf_df = pd.DataFrame.from_dict(prtf_vs_params, orient='index')
        prtf_df.columns = ['scenario_mode', 'scenarios', 'trading_freq', 'gamma_risk', 'gamma_trade',
                           'excess_return', 'excess_risk', 'max_drawdown', 'turnover']

        # We've already written to this file, read + append + overwrite
        prtf_df.to_csv(file_path, index=False,
                       header=False if os.path.isfile(file_path) else True,
                       mode='a' if os.path.isfile(file_path) else 'w')

        # Print run stats and advance run
        end_time = time.time()
        logging.warning('Run #{run}/{runs} complete. Expected time of completion: {eta}'.format(
            run=run, runs=total_runs, eta=dt.datetime.now() +
                                          (total_runs - run) * dt.timedelta(seconds=end_time - start_time)
            )
        )
        run += 1
