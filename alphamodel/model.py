"""
Alpha Model Base Template
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import pickle
import yaml

from abc import ABCMeta, abstractmethod
from .data_set import TimeSeriesDataSet
from datetime import datetime
from enum import Enum
from os import path

__all__ = ['Model', 'SamplingFrequency']


class SamplingFrequency(Enum):
    DAY = 'daily'
    WEEK = 'weekly'
    MONTH = 'monthly'
    QUARTER = 'quarterly'


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
        self.__realized = {}
        self.__predicted = {}
        for freq in SamplingFrequency:
            self.__realized[freq] = {}
            self.__predicted[freq] = {}
        self.__removed_assets = set()
        self.__removed_dates = set()

    @property
    def _realized(self):
        """
        Retrieval of realized data, hidden for model use. Use get/set instead externally
        :return:
        """
        populated_freq = {}
        for freq in self.__realized:
            if self.__realized[freq]:
                populated_freq[freq] = self.__realized[freq]

        return populated_freq

    @property
    def _predicted(self):
        """
        Retrieval of realized data, hidden for model use. Use get/set instead externally
        :return:
        """
        populated_freq = {}
        for freq in self.__realized:
            if self.__predicted[freq]:
                populated_freq[freq] = self.__predicted[freq]

        return populated_freq

    def get(self, item, data_type='realized', sampling_freq='daily'):
        """
        Universal getter for data: realized or predicted
        :param item:
        :param data_type:
        :param sampling_freq:
        :return:
        """
        if isinstance(sampling_freq, SamplingFrequency):
            pass
        elif isinstance(sampling_freq, str):
            sampling_freq = SamplingFrequency(sampling_freq)
        else:
            raise ValueError('Model.get: Unable to parse sampling_freq that is not SamplingFrequency or str')

        if data_type == 'realized':
            if item in self.__realized[sampling_freq]:
                return self.__realized[sampling_freq][item]
            else:
                return None
        elif data_type == 'predicted':
            return self.__predicted[item]
        else:
            raise ValueError('Model.get: Unable to parse data_type not in [\'realized\', \'predicted\']')

    def set(self, item, value, data_type='realized', sampling_freq='daily'):
        """
        Universal setter for data: realized or predicted
        :param item:
        :param value:
        :param data_type:
        :param sampling_freq:
        :return:
        """
        if isinstance(sampling_freq, SamplingFrequency):
            pass
        elif isinstance(sampling_freq, str):
            sampling_freq = SamplingFrequency(sampling_freq)
        else:
            raise ValueError('Model.get: Unable to parse sampling_freq that is not SamplingFrequency or str')

        if data_type == 'realized':
            self.__realized[sampling_freq][item] = value
        elif data_type == 'predicted':
            self.__predicted[item] = value
        else:
            raise ValueError('Model.get: Unable to parse data_type not in [\'realized\', \'predicted\']')

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

    def save(self):
        """
        Save all data in class
        :return: n/a
        """
        f = open(self.filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self):
        """
        Load back data from file
        :return: success bool
        """
        if path.exists(self.filename):
            # Load class from file
            f = open(self.filename, 'rb')
            tmp_dict = pickle.load(f)
            f.close()

            # Save config
            cfg = self.cfg

            # Reload class from file, but keep current config
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)
            self.cfg = cfg
            return True

        return False

    @abstractmethod
    def train(self, **kwargs):
        """
        Train model
        :param kwargs:
        :return: n/a
        """
        pass

    def _fetch_base_data(self, force=False):
        """
        Base data fetching function for model
        :param force: force re-fetch
        :return: success bool
        """
        # If we can load past state from file, let's just do that
        if not force and self.load():
            return True

        success = True

        # Figure out what time frequencies we need to retrieve
        sampling_freqs = set()
        # Validate return series we're working with
        if 'sampling_freq' in self.cfg['returns']:
            sampling_freqs.add(self.cfg['returns']['sampling_freq'])
        else:
            sampling_freqs.add('daily')

        # Validate type of covariance we're constructing
        if self.cfg['covariance']['method'] == 'FF5':
            # If already loaded daily returns, don't need to do it again
            sampling_freqs.add('daily')
            success = success and self.__fetch_factor_data()
        elif 'sampling_freq' in self.cfg['covariance'] and \
                self.cfg['covariance']['sampling_freq'] != self.cfg['returns']['sampling_freq']:
            # If already loaded the same sampling frequency, don't need to do it again
            sampling_freqs.add(self.cfg['returns']['sampling_freq'])

        # Retrieve all sampling frequencies, validate & save
        for freq in sampling_freqs:
            success = success and self.__fetch_market_data(freq)
        success = success and self.__validate_and_sync_return_data()

        if success:
            self.save()
        return success

    def __fetch_market_data(self, sampling_freq='daily'):
        """
        Raw data fetch & save
        :return:
        """
        # Fetch market data & store as raw_data + individual prices / volumes
        raw_data = {}

        # #### Download loop

        # Download asset data & construct a data dictionary: {ticker: pd.DataFrame(price/volume)}
        # If Quandl complains about the speed of requests, try adding sleep time.
        for ticker in self.universe:
            if ticker in raw_data:
                continue
            print('downloading %s from %s to %s' % (ticker, self.cfg['start_date'], self.cfg['end_date']))
            fetched = self.data_source.get(ticker, self.cfg['start_date'], self.cfg['end_date'],
                                           freq=sampling_freq)
            if fetched is not None:
                raw_data[ticker] = fetched

        # #### Computation

        keys = [el for el in self.universe if el not in (set(self.universe) - set(raw_data.keys()))]

        def select_first_valid_column(df, columns):
            for column in columns:
                if column in df.columns:
                    return df[column]

        # extract prices
        prices = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(raw_data[k], ["Adj. Close", "Close", "Value"])
                            for k in keys])))

        # compute sigmas
        open_prices = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(raw_data[k], ["Open"]) for k in keys])))
        close_prices = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(raw_data[k], ["Close"]) for k in keys])))

        # extract volumes
        volumes = pd.DataFrame.from_dict(dict(zip(keys, [select_first_valid_column(raw_data[k],
                                                                                   ["Adj. Volume", "Volume"])
                                                         for k in keys])))

        # fix risk free
        prices[self.risk_free_symbol] = 10000 * (1 + prices[self.risk_free_symbol] / (100 * 250)).cumprod()

        # #### Save raw price data
        self.set('raw_data', raw_data, data_type='realized', sampling_freq=sampling_freq)
        self.set('prices', prices, data_type='realized', sampling_freq=sampling_freq)
        self.set('open_prices', open_prices, data_type='realized', sampling_freq=sampling_freq)
        self.set('close_prices', close_prices, data_type='realized', sampling_freq=sampling_freq)
        self.set('volumes', volumes, data_type='realized', sampling_freq=sampling_freq)

        return True

    def __validate_and_sync_return_data(self):
        """
        Validate and sync up return data
        :return:
        """
        # For each sampling frequency, validate the data & store problem assets/dates
        for freq in self._realized:
            # Grab stored raw market data
            prices = self.get('prices', data_type='realized', sampling_freq=freq)
            open_prices = self.get('open_prices', data_type='realized', sampling_freq=freq)
            close_prices = self.get('close_prices', data_type='realized', sampling_freq=freq)
            volumes = self.get('volumes', data_type='realized', sampling_freq=freq)

            # Validate prices
            # Filter NaNs - threshold at 2% missing values
            bad_assets = prices.columns[prices.isnull().sum() > len(prices) * 0.02]
            if len(bad_assets):
                self.__removed_assets = self.__removed_assets.union(set(bad_assets))

            # Fix dates on which many assets have missing values
            nassets = prices.shape[1]
            bad_dates_p = prices.index[prices.isnull().sum(1) > nassets * .9]
            bad_dates_o = open_prices.index[open_prices.isnull().sum(1) > nassets * .9]
            bad_dates_c = close_prices.index[close_prices.isnull().sum(1) > nassets * .9]
            bad_dates_v = volumes.index[volumes.isnull().sum(1) > nassets * .9]
            bad_dates = set(bad_dates_p).union(set(bad_dates_o)).union(set(bad_dates_c)).union(set(bad_dates_v))

            # Maintain list of removed dates across all data fetches
            if len(bad_dates):
                self.__removed_dates = self.__removed_dates.union(set(bad_dates))

            # Compute returns
            returns = (prices.diff() / prices.shift(1)).fillna(method='ffill').iloc[1:]
            bad_assets = returns.columns[((-.5 > returns).sum() > 0) | ((returns > 2.).sum() > 0)]

            # Maintain list of removed assets across all data fetches
            if len(bad_assets):
                self.__removed_assets = self.__removed_assets.union(set(bad_assets))

        # Now that all validation is complete, filter out the bad assets/dates across all sampling frequencies &
        #   run calculations
        for freq in self._realized:
            # Grab stored raw market data
            prices = self.get('prices', data_type='realized', sampling_freq=freq)
            open_prices = self.get('open_prices', data_type='realized', sampling_freq=freq)
            close_prices = self.get('close_prices', data_type='realized', sampling_freq=freq)
            volumes = self.get('volumes', data_type='realized', sampling_freq=freq)

            # Filter NaNs - threshold at 2% missing values
            if len(self.__removed_assets):
                print('%s assets %s have too many NaNs, removing them' % (str(freq), self.__removed_assets))

                prices = prices.loc[:, ~prices.columns.isin(self.__removed_assets)]
                open_prices = open_prices.loc[:, ~open_prices.columns.isin(self.__removed_assets)]
                close_prices = close_prices.loc[:, ~close_prices.columns.isin(self.__removed_assets)]
                volumes = volumes.loc[:, ~volumes.columns.isin(self.__removed_assets)]

            # Fix dates on which many assets have missing values
            if len(self.__removed_dates):
                bad_dates_idx = pd.Index(self.__removed_dates).sort_values()
                print("Removing these days from dataset:")
                print(pd.DataFrame({'nan price': prices.isnull().sum(1)[bad_dates_idx],
                                    'nan open price': open_prices.isnull().sum(1)[bad_dates_idx],
                                    'nan close price': close_prices.isnull().sum(1)[bad_dates_idx],
                                    'nan volumes': volumes.isnull().sum(1)[bad_dates_idx]}))

                prices = prices.loc[~prices.index.isin(bad_dates_idx)]
                open_prices = open_prices.loc[~open_prices.index.isin(bad_dates_idx)]
                close_prices = close_prices.loc[~close_prices.index.isin(bad_dates_idx)]
                volumes = volumes.loc[~volumes.index.isin(bad_dates_idx)]

            # Fix prices
            print(pd.DataFrame({'remaining nan price': prices.isnull().sum(),
                                'remaining nan open price': open_prices.isnull().sum(),
                                'remaining nan close price': close_prices.isnull().sum(),
                                'remaining nan volumes': volumes.isnull().sum()}))

            # ### Calculate sigmas
            sigmas = np.abs(np.log(open_prices.astype(float)) - np.log(close_prices.astype(float)))

            # Forward fill any gaps
            prices = prices.fillna(method='ffill')
            open_prices = open_prices.fillna(method='ffill')
            close_prices = close_prices.fillna(method='ffill')
            sigmas = sigmas.fillna(method='ffill')
            volumes = volumes.fillna(method='ffill')

            # Also remove the first row just in case it had gaps since we can't forward fill it
            prices = prices.iloc[1:]
            open_prices = open_prices.iloc[1:]
            close_prices = close_prices.iloc[1:]
            sigmas = sigmas.iloc[1:]
            volumes = volumes.iloc[1:]
            print(pd.DataFrame({'remaining nan price': prices.isnull().sum(),
                                'remaining nan open price': open_prices.isnull().sum(),
                                'remaining nan close price': close_prices.isnull().sum(),
                                'remaining nan volumes': volumes.isnull().sum(),
                                'remaining nan sigmas': sigmas.isnull().sum()}))

            # #### Calculate volumes & returns
            # Make volumes in dollars
            volumes = volumes * prices

            # Compute returns
            returns = (prices.diff() / prices.shift(1)).fillna(method='ffill').iloc[1:]

            # Remove USDOLLAR except from returns
            prices = prices.iloc[:, :-1]
            open_prices = open_prices.iloc[:, :-1]
            close_prices = close_prices.iloc[:, :-1]
            sigmas = sigmas.iloc[:, :-1]
            volumes = volumes.iloc[:, :-1]

            # Save all calculated data
            self.set('prices', prices, data_type='realized', sampling_freq=freq)
            self.set('open_prices', open_prices, data_type='realized', sampling_freq=freq)
            self.set('close_prices', close_prices, data_type='realized', sampling_freq=freq)
            self.set('volumes', volumes, data_type='realized', sampling_freq=freq)
            self.set('sigmas', sigmas, data_type='realized', sampling_freq=freq)
            self.set('returns', returns, data_type='realized', sampling_freq=freq)

        return True

    def __fetch_factor_data(self):
        """
        Training function for model
        :return:
        """
        ds = pdr.DataReader('North_America_5_Factors_Daily', 'famafrench',
                            start=self.cfg['start_date'], end=self.cfg['end_date'])
        ff_returns = ds[0]
        ff_returns.index = ff_returns.index.to_timestamp()
        self.set('ff_returns', ff_returns, data_type='realized', sampling_freq='daily')

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
