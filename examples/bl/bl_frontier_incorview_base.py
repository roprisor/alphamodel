# # Black Litterman with HMM Views Risk Rewards Frontier: Oldest Country ETFs

# ## 1. Data Fetching

# ### 1.1 Model configuration

import datetime as dt
import logging
import pandas as pd
import numpy as np
import socket
import time
import cvxportfolio as cp
import alphamodel as am

config = {'name': 'bl_sim_incor',
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
               'end_date': '20091231',
               'halflife': 20,
               'min_periods': 3,
               'hidden_states': 2,
               'train_len': 1700,
               'data_dir': '/mnt/research_masc/data_store/bl/',
               'returns':
                   {'sampling_freq': 'daily'},
               'covariance':
                    {'method': 'SS',
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
ss = am.SingleStockBLEWM(config)
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
# start_date = dt.datetime.strptime(config['model']['start_date'], '%Y%m%d') + \
#                 dt.timedelta(days=config['model']['train_len']*1.75)
start_date = dt.datetime(2005, 1, 2)
end_date = dt.datetime.strptime(config['model']['end_date'], '%Y%m%d')
logging.warning('Start Date: {sd} - End Date: {ed}'.format(sd=start_date.strftime('%Y%m%d'),
                                                           ed=end_date.strftime('%Y%m%d')))

# Hyperparameters:
# - view_confidence: confidence in BL view
# - gamma_risk: risk weighting factor
# - gamma_trade: trade cost weighting factor

# Search parameters
risk_aversion = 2.5
confidence = 0.8
view_confidence = np.arange(0.0, 1.05, 0.05)
gamma_risk = [0.1]
gamma_trade = [0.1]
gamma_hold = 0.
total_runs = len(view_confidence) * len(gamma_risk) * len(gamma_trade)

prtf_vs_params = {}
run = 1

for vconf in view_confidence:
    for grisk in gamma_risk:
        for gtrd in gamma_trade:
            # New run key
            key = 'vconf' + str(vconf) + '_grisk' + str(grisk) + '_gtrd' + str(gtrd)
            logging.warning('Running for hyperparams {}'.format(key))
            start_time = time.time()

            try:
                # Predict and gather metrics
                # US outperforms Germany 4% per year - incorrect view
                ss.predict(w_market_cap_init=w_mktcap, risk_aversion=risk_aversion, c=confidence,
                           P_view=np.array([1, 0, 0, -1, 0, 0, 0, 0, 0, 0]), Q_view=np.array(0.04 / 252),
                           view_confidence=vconf, noise_mode='confidence_dynamic'
                           )

                logging.warning('Prediction complete')

                # Black Litterman output (HMM views included)
                r_pred = ss.get('returns', 'predicted')
                covariance_pred = ss.get('covariance', 'predicted')
                volumes_pred = ss.get('volumes', 'predicted')
                sigmas_pred = ss.get('sigmas', 'predicted')

                # Predicted costs
                optimization_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1.,
                                                   sigma=sigmas_pred, volume=volumes_pred)
                optimization_hcost=cp.HcostModel(borrow_costs=0.0001)

                # Covariance setup
                bl_risk_model = cp.FullSigma(covariance_pred)

                # Black Litterman policy
                logging.warning('Running backtest')

                # Optimization parameters
                leverage_limit = cp.LeverageLimit(1)
                fully_invested = cp.ZeroCash()
                long_only = cp.LongOnly()

                # Optimization policy
                bl_spo_policy = cp.SinglePeriodOpt(return_forecast=r_pred,
                                                   costs=[grisk * bl_risk_model,
                                                          gtrd * optimization_tcost,
                                                          gamma_hold * optimization_hcost],
                                                   constraints=[leverage_limit, fully_invested, long_only],
                                                   trading_freq='hour')

                # Backtest
                blu_results = simulator.run_multiple_backtest(1E6*w_mktcap, start_time=start_date,  end_time=end_date,
                                                              policies=[bl_spo_policy],
                                                              loglevel=logging.WARNING, parallel=True)
                result = blu_results[0]
                logging.warning(result.summary())

                # Save down metrics together with parameters
                prtf_vs_params[key] = [vconf, grisk, gtrd,
                                       result.excess_returns.mean() * 100 * result.ppy,
                                       result.excess_returns.std() * 100 * np.sqrt(result.ppy),
                                       result.max_drawdown * 100,
                                       result.turnover.mean() * 100 * result.ppy]
            except Exception as e:
                logging.error('Ran into an error: {e}'.format(e=str(e)))
                prtf_vs_params[key] = [vconf, grisk, gtrd, 0, 0, 100, 0]

            # Save down values in .csv
            prtf_df = pd.DataFrame.from_dict(prtf_vs_params, orient='index')
            prtf_df.columns = ['view_confidence', 'gamma)risk', 'gamma_trade',
                               'excess_return', 'excess_risk', 'max_drawdown', 'turnover']
            prtf_df.to_csv(ss.cfg['data_dir'] + 'bl_ewm_incorview_{}.csv'.format(host), index=False)

            # Print run stats and advance run
            end_time = time.time()
            logging.warning('Run #{run}/{runs} complete. Expected time of completion: {eta}'.format(
                run=run, runs=total_runs, eta=dt.datetime.now() +
                                              (total_runs - run) * dt.timedelta(seconds=end_time - start_time)
                )
            )
            run += 1
