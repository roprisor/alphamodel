"""
Alpha Model Base Template
"""

import pandas as pd
import numpy as np
import yaml

from abc import ABCMeta, abstractmethod
from .data_set import TimeSeriesDataSet
from datetime import datetime

__all__ = ['Model']


class Model(metaclass=ABCMeta):
    """
    Model Base Template
    """
    def __init__(self, config):
        """
        Initialization of params needed to create/use an alpha model
        :param config: config file path or dictionary
        :return: n/a
        """
        cfg = Model.parse_config(config)

        # Parse required model params
        self.name = cfg['name']
        self.cfg = cfg['model']
        self.data_dir = self.cfg['data_dir']

        # TODO: Configs should be easily accessible through vars, at least model configs

        try:
            if 'list' in cfg['universe']:
                self.universe = cfg['universe']['list']
            elif 'path' in cfg['universe']:
                self.universe = pd.read_csv(cfg['universe']['path'])[cfg['universe']['ticker_col']].to_list()
            self.risk_free_symbol = cfg['universe']['risk_free_symbol']
        except ValueError:
            raise NotImplemented('Model\'s universe can only be a a dict w/ (list, risk_free_symbol) or '
                                 '(path, ticker_col, risk_free_symbol)')

        # Initialize data sources & variables
        self.__data_source = TimeSeriesDataSet.init(cfg)
        self.realized = {}
        self.predicted = {}

    @property
    def data_source(self):
        """
        Data source
        :return: ds
        """
        return self.__data_source

    @staticmethod
    def parse_config(config):
        """
        Input validation of config
        :param config: config file path or dictionary
        :return: alpha config dict
        """

        # Input validation and parsing
        cfg = {}
        if isinstance(config, str):
            with open(config, 'r') as cfg_file:
                cfg = yaml.load(cfg_file, yaml.SafeLoader)

                if 'alpha' not in cfg:
                    raise ValueError('\'alpha\'  section missing, required to initialize an alpha model.')
                cfg = cfg['alpha']
        elif isinstance(config, dict):
            if 'alpha' in config:
                cfg = config['alpha']
            else:
                cfg = config
        else:
            raise TypeError('Model configuration needs to be passed in as either yaml file path or dict.')

        return cfg

    @property
    def filename(self):
        """
        Generate save/load filename
        :return: string
        """
        return self.data_dir + 'model_' + self.name + '_' + datetime.today().strftime('%Y%m%d') + '.mdl'

    @abstractmethod
    def save(self):
        """
        Save model to h5 or pickle file
        :return:
        """
        pass

    @abstractmethod
    def load(self):
        """
        Save model to h5 or pickle file
        :return:
        """
        pass

    @abstractmethod
    def train(self, **kwargs):
        """
        Train model
        :param kwargs:
        :return: n/a
        """
        pass

    def _fetch_data(self, force=False):
        """
        Training function for model
        :return:
        """
        # If we can load past state from file, let's just do that
        if not force and self.load():
            return True

        data = {}

        # #### Download loop

        # Download asset data & construct a data dictionary: {ticker: pd.DataFrame(price/volume)}
        # If Quandl complains about the speed of requests, try adding sleep time.
        for ticker in self.universe:
            if ticker in data:
                continue
            print('downloading %s from %s to %s' % (ticker, self.cfg['start_date'], self.cfg['end_date']))
            fetched = self.data_source.get(ticker, self.cfg['start_date'], self.cfg['end_date'])
            if fetched is not None:
                data[ticker] = fetched

        # #### Computation

        keys = [el for el in self.universe if el not in (set(self.universe) - set(data.keys()))]

        def select_first_valid_column(df, columns):
            for column in columns:
                if column in df.columns:
                    return df[column]

        # extract prices
        prices = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(data[k], ["Adj. Close", "Close", "Value"])
                            for k in keys])))

        # compute sigmas
        open_price = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(data[k], ["Open"]) for k in keys])))
        close_price = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(data[k], ["Close"]) for k in keys])))
        sigmas = np.abs(np.log(open_price.astype(float)) - np.log(close_price.astype(float)))

        # extract volumes
        volumes = pd.DataFrame.from_dict(dict(zip(keys, [select_first_valid_column(data[k], ["Adj. Volume", "Volume"])
                                                         for k in keys])))

        # fix risk free
        prices[self.risk_free_symbol] = 10000 * (1 + prices[self.risk_free_symbol] / (100 * 250)).cumprod()

        # #### Filtering

        # filter NaNs - threshold at 2% missing values
        bad_assets = prices.columns[prices.isnull().sum() > len(prices) * 0.02]
        if len(bad_assets):
            print('Assets %s have too many NaNs, removing them' % bad_assets)

        prices = prices.loc[:, ~prices.columns.isin(bad_assets)]
        sigmas = sigmas.loc[:, ~sigmas.columns.isin(bad_assets)]
        volumes = volumes.loc[:, ~volumes.columns.isin(bad_assets)]

        nassets = prices.shape[1]

        # days on which many assets have missing values
        bad_days1 = sigmas.index[sigmas.isnull().sum(1) > nassets * .9]
        bad_days2 = prices.index[prices.isnull().sum(1) > nassets * .9]
        bad_days3 = volumes.index[volumes.isnull().sum(1) > nassets * .9]
        bad_days = pd.Index(set(bad_days1).union(set(bad_days2)).union(set(bad_days3))).sort_values()
        print("Removing these days from dataset:")
        print(pd.DataFrame({'nan price': prices.isnull().sum(1)[bad_days],
                            'nan volumes': volumes.isnull().sum(1)[bad_days],
                            'nan sigmas': sigmas.isnull().sum(1)[bad_days]}))

        prices = prices.loc[~prices.index.isin(bad_days)]
        sigmas = sigmas.loc[~sigmas.index.isin(bad_days)]
        volumes = volumes.loc[~volumes.index.isin(bad_days)]
        print(pd.DataFrame({'remaining nan price': prices.isnull().sum(),
                            'remaining nan volumes': volumes.isnull().sum(),
                            'remaining nan sigmas': sigmas.isnull().sum()}))

        # forward fill any gaps
        prices = prices.fillna(method='ffill')
        sigmas = sigmas.fillna(method='ffill')
        volumes = volumes.fillna(method='ffill')

        # also remove the first row just in case it had gaps since we can't forward fill it
        prices = prices.iloc[1:]
        sigmas = sigmas.iloc[1:]
        volumes = volumes.iloc[1:]
        print(pd.DataFrame({'remaining nan price': prices.isnull().sum(),
                            'remaining nan volumes': volumes.isnull().sum(),
                            'remaining nan sigmas': sigmas.isnull().sum()}))

        # #### Save

        # make volumes in dollars
        volumes = volumes * prices

        # compute returns
        returns = (prices.diff() / prices.shift(1)).fillna(method='ffill').iloc[1:]

        bad_assets = returns.columns[((-.5 > returns).sum() > 0) | ((returns > 2.).sum() > 0)]
        if len(bad_assets):
            print('Assets %s have dubious returns, removed' % bad_assets)

        prices = prices.loc[:, ~prices.columns.isin(bad_assets)]
        sigmas = sigmas.loc[:, ~sigmas.columns.isin(bad_assets)]
        volumes = volumes.loc[:, ~volumes.columns.isin(bad_assets)]
        returns = returns.loc[:, ~returns.columns.isin(bad_assets)]

        # remove USDOLLAR except from returns
        prices = prices.iloc[:, :-1]
        sigmas = sigmas.iloc[:, :-1]
        volumes = volumes.iloc[:, :-1]

        self.realized['data'] = data
        self.realized['prices'] = prices
        self.realized['returns'] = returns
        self.realized['sigmas'] = sigmas
        self.realized['volumes'] = volumes

        self.save()
        return True

    @abstractmethod
    def predict(self, **kwargs):
        """
        Predict using model
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def prediction_quality(self, statistic=None):
        """
        Output 1 statistic to judge the prediction quality, should be configurable
        :param statistic:
        :return:
        """
        pass

    @abstractmethod
    def predict_next(self, **kwargs):
        """
        Predict using model outside of data period, i.e., the future
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def show_results(self, **kwargs):
        """
        Show/plot results for out of sample prediction
        :param kwargs:
        :return:
        """
        pass
