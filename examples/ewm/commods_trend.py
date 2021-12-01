import logging
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import cvxportfolio as cp
import alphamodel as am
import seaborn as sns
import quandl
quandl.ApiConfig.api_key = "FUCTkpiUgvDWk6faUD9J"

# Fetch asset data
ss = am.SingleStockEWM('../cvxpt_ewm.yml')
ss.train(force=True)

# Realized Data for Simulation
asset_prices = ss.get('prices', 'realized', ss.cfg['returns']['sampling_freq'])
asset_returns = ss.get('returns', 'realized', ss.cfg['returns']['sampling_freq'])
print(asset_prices)

# Realized Data for Simulation
prices = ss.get('prices', 'realized', ss.cfg['returns']['sampling_freq']).iloc[1:, :]
returns = ss.get('returns', 'realized', ss.cfg['returns']['sampling_freq'])
volumes = ss.get('volumes', 'realized', ss.cfg['returns']['sampling_freq'])
sigmas = ss.get('sigmas', 'realized', ss.cfg['returns']['sampling_freq'])

simulated_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                               market_volumes=volumes, cash_key=ss.risk_free_symbol)


# Search parameters
trading_freq = 'month'
gamma_risk = 0.1
gamma_trade = 5
gamma_hold = 0.1

# Predicted values
ss.predict()
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

# Single Period Policy
logging.warning('Running backtest')

# Optimization parameters
start_date = dt.datetime(1987, 6, 1)
end_date = dt.datetime(2021, 6, 1)
leverage_limit = cp.LeverageLimit(5)
fully_invested = cp.ZeroCash()
long_only = cp.LongOnly()
w_init = pd.Series(index=ss.universe, data=[1e6/len(ss.universe)] * len(ss.universe))
w_init['USDOLLAR'] = 0.

# Optimization policy
spo_policy = cp.SinglePeriodOpt(return_forecast=r_pred,
                                costs=[gamma_risk * risk_model,
                                        gamma_trade * optimization_tcost,
                                        gamma_hold * optimization_hcost],
                                constraints=[leverage_limit, fully_invested],
                                trading_freq=trading_freq)

# Backtest
spo_results = simulator.run_multiple_backtest(w_init,
                                              start_time=start_date, end_time=end_date,
                                              policies=[spo_policy],
                                              loglevel=logging.WARNING, parallel=False)
result = spo_results[0]
print(result.summary())
