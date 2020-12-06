# alphamodel

**alphamodel** is an alpha development tool meant to package data fetching, model training and model prediction.
The base example models are designed to fetch historical data from Quandl, generate predictions (basic EWMA or 
HMM) and estimate a covariance matrix based on direct estimation or Fama-French factor models. The base model
outputs are standardized as inputs to the **cvxportfolio** library which can be used for portfolio optimization
and back testing.

A special application of **alphamodel** is its use with Black Litterman return and risk estimates where the
user can provide investment views in a linear combination based format and the model automatically incorporates
them together with an EWMA or HMM based confidence level into the output return and risk estimates.

If using this library please cite the upcoming paper:
**Multi-Period Optimization with Investor Views under Regime Switching** by Razvan G. Oprisor and Roy H. Kwon

## Config

```alpha:
  name: rebalance_sim
  universe:
    path: '../data/SP100_2010.csv'
    ticker_col: Symbol
    risk_free_symbol: USDOLLAR
  drop_threshold: 0.5
  data:
    name: eod_returns
    source: quandl
    table: EOD
    api_key: 6XyApK2BBj_MraQg2TMD
  model:
    start_date: '20100102'
    end_date: '20171231'
    data_dir: '../data/'
    halflife: 4
    horizon: 1
    min_periods: 4
    returns:
      sampling_freq: weekly
    covariance:
      method: FF5
      sampling_freq: daily
      update: monthly```

## Examples

Please review the **alphamodel/examples** folder for iPython notebooks with example simulations.

