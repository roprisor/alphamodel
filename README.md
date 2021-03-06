# alphamodel

## What is it meant for?

`alphamodel` is an alpha development tool meant to package data fetching, model training and model prediction.
The base example models are designed to:
* fetch historical data from Quandl or csvs
* generate predictions (basic EWMA or HMM)
* estimate a covariance matrix based on direct estimation or Fama-French factor models.

The base model outputs are standardized as inputs to the `cvxportfolio` library which can be used for portfolio 
optimization and back testing.

## Can it do anything special?

Funny you should ask, yes! A custom application of **alphamodel** is its use to tell investors when to invest in
their views and when to hold off. To achieve this, it uses Black Litterman return and risk estimates where:
* the user can provide investment views in a linear combination based format
and
* the model automatically incorporates them together with an EWMA or HMM based confidence level a new set of
output return and risk estimates.

Using this new set of estimates leads to a portfolio with the views incorporated (proportional with how much
the model thinks they're likely to be active at that time).

## Config

The configuration follows a simple `yml` format but can also be provided directly as a nested dictionary.
See `alphamodel/examples/` for 2 sample yml files.

```
alpha:
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
      update: monthly
```

## Examples

Please review the `alphamodel/examples` sub-folders for:
1. Jupyter notebooks with sample simulations and charts similar to the paper
2. Python scripts to rerun the full efficient frontier simulations


## Remember to cite our paper

If using this library please cite the upcoming paper:
* **Multi-Period Optimization with Investor Views under Regime Switching** by Razvan G. Oprisor and Roy H. Kwon, J. Risk Financial Manag. 2021, 14(1), 3; https://doi.org/10.3390/jrfm14010003


