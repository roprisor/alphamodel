"""
Alpha Model Base Template
"""

import logging
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

__all__ = ['Model', 'ModelState', 'SamplingFrequency']


class SamplingFrequency(Enum):
    DAY = 'daily'
    WEEK = 'weekly'
    MONTH = 'monthly'
    QUARTER = 'quarterly'


class ModelState(Enum):
    INITIALIZED = 0
    TRAINED = 0
    PREDICTED = 0


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
                self._universe = cfg['universe']['list']
            elif 'path' in cfg['universe']:
                self._universe = pd.read_csv(cfg['universe']['path'])[cfg['universe']['ticker_col']].to_list()

            # Add risk_free_symbol
            if cfg['universe']['risk_free_symbol']:
                self.risk_free_symbol = cfg['universe']['risk_free_symbol']
            else:
                self.risk_free_symbol = 'USDOLLAR'

            # risk_free_symbol should be part of _universe in order to be fetched but should not be accessible otherwise
            if self.risk_free_symbol not in self._universe:
                self._universe.append(self.risk_free_symbol)
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
        self.__state = ModelState.INITIALIZED

    @property
    def state(self):
        """
        Retrieve model state
        :return: ModelState Enum
        """
        return self.__state

    @property
    def universe(self):
        """
        Retrieve universe == list of symbols we want to trade
        :return: list of strings
        """
        return [s for s in self._universe if s != self.risk_free_symbol]

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
        try:
            f = open(self.filename, 'wb')
            pickle.dump(self.__dict__, f, 2)
            f.close()
        except Exception as e:
            logging.error('save: Unable to save pickle file, {e}'.format(e=str(e)))
            return False

        return True

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
        self.__state = ModelState.TRAINED

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
        for ticker in self._universe:
            if ticker in raw_data:
                continue
            logging.info('downloading %s from %s to %s' % (ticker, self.cfg['start_date'], self.cfg['end_date']))
            fetched = self.data_source.get(ticker, self.cfg['start_date'], self.cfg['end_date'],
                                           freq=sampling_freq)
            if fetched is not None:
                raw_data[ticker] = fetched

        # #### Computation

        keys = [el for el in self._universe if el not in (set(self._universe) - set(raw_data.keys()))]

        def select_first_valid_column(df, columns):
            for column in columns:
                if column in df.columns:
                    return df[column]

        # extract prices
        prices = pd.DataFrame.from_dict(
            dict(zip(keys, [select_first_valid_column(raw_data[k], ["Adj_Close", "Close", "Value"])
                            for k in keys])))

        if 'Open' in self.data_source.columns and 'Close' in self.data_source.columns:
            # compute sigmas
            open_prices = pd.DataFrame.from_dict(
                dict(zip(keys, [select_first_valid_column(raw_data[k], ["Open"]) for k in keys])))
            close_prices = pd.DataFrame.from_dict(
                dict(zip(keys, [select_first_valid_column(raw_data[k], ["Close"]) for k in keys])))

            self.set('open_prices', open_prices, data_type='realized', sampling_freq=sampling_freq)
            self.set('close_prices', close_prices, data_type='realized', sampling_freq=sampling_freq)

        if 'Volume' in self.data_source.columns or 'Adj_Volume' in self.data_source.columns:
            # extract volumes
            volumes = pd.DataFrame.from_dict(dict(zip(keys, [select_first_valid_column(raw_data[k],
                                                                                       ["Adj_Volume", "Volume"])
                                                             for k in keys])))

            self.set('volumes', volumes, data_type='realized', sampling_freq=sampling_freq)

        # fix risk free
        prices[self.risk_free_symbol] = 10000 * (1 + prices[self.risk_free_symbol] / (100 * 250)).cumprod()

        # #### Save raw price data
        self.set('raw_data', raw_data, data_type='realized', sampling_freq=sampling_freq)
        self.set('prices', prices, data_type='realized', sampling_freq=sampling_freq)

        return True

    def __validate_and_sync_return_data(self):
        """
        Validate and sync up return data
        :return:
        """
        range_avail = 'Open' in self.data_source.columns and 'Close' in self.data_source.columns
        volume_avail = 'Volume' in self.data_source.columns or 'Adj_Volume' in self.data_source.columns

        # For each sampling frequency, validate the data & store problem assets/dates
        for freq in self._realized:
            # Grab stored raw market data
            prices = self.get('prices', data_type='realized', sampling_freq=freq)
            if range_avail:
                open_prices = self.get('open_prices', data_type='realized', sampling_freq=freq)
                close_prices = self.get('close_prices', data_type='realized', sampling_freq=freq)
            if volume_avail:
                volumes = self.get('volumes', data_type='realized', sampling_freq=freq)

            # Validate prices
            # Filter NaNs - threshold at 2% missing values
            bad_assets = prices.columns[prices.isnull().sum() > len(prices) * 0.02]
            if len(bad_assets):
                self.__removed_assets = self.__removed_assets.union(set(bad_assets))

            # Fix dates on which many assets have missing values
            nassets = prices.shape[1]
            bad_dates_p = prices.index[prices.isnull().sum(1) >= min(nassets * .9, nassets - 1)]
            if range_avail:
                bad_dates_o = open_prices.index[open_prices.isnull().sum(1) >= min(nassets * .9, nassets - 1)]
                bad_dates_c = close_prices.index[close_prices.isnull().sum(1) >= min(nassets * .9, nassets - 1)]
            if volume_avail:
                bad_dates_v = volumes.index[volumes.isnull().sum(1) >= min(nassets * .9, nassets - 1)]

            bad_dates = set(bad_dates_p)
            if range_avail:
                bad_dates = bad_dates.union(set(bad_dates_o)).union(set(bad_dates_c))
            if volume_avail:
                bad_dates = bad_dates.union(set(bad_dates_v))

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
            if range_avail:
                open_prices = self.get('open_prices', data_type='realized', sampling_freq=freq)
                close_prices = self.get('close_prices', data_type='realized', sampling_freq=freq)
            if volume_avail:
                volumes = self.get('volumes', data_type='realized', sampling_freq=freq)

            # Filter NaNs - threshold at 2% missing values
            if len(self.__removed_assets):
                logging.warning('%s assets %s have too many NaNs, removing them' % (str(freq), self.__removed_assets))

                prices = prices.loc[:, ~prices.columns.isin(self.__removed_assets)]
                if range_avail:
                    open_prices = open_prices.loc[:, ~open_prices.columns.isin(self.__removed_assets)]
                    close_prices = close_prices.loc[:, ~close_prices.columns.isin(self.__removed_assets)]
                if volume_avail:
                    volumes = volumes.loc[:, ~volumes.columns.isin(self.__removed_assets)]

            # Fix dates on which many assets have missing values
            if len(self.__removed_dates):
                bad_dates_idx = pd.Index(self.__removed_dates).sort_values()
                logging.warning("Removing these days from dataset:")
                logging.warning(pd.DataFrame({'nan price': prices.isnull().sum(1)[bad_dates_idx]}))

                prices = prices.loc[~prices.index.isin(bad_dates_idx)]
                if range_avail:
                    open_prices = open_prices.loc[~open_prices.index.isin(bad_dates_idx)]
                    close_prices = close_prices.loc[~close_prices.index.isin(bad_dates_idx)]
                if volume_avail:
                    volumes = volumes.loc[~volumes.index.isin(bad_dates_idx)]

            # Fix prices
            if sum([x.isnull().sum().sum() for x in [prices]]) != 0:
                logging.warning(pd.DataFrame({'remaining nan price': prices.isnull().sum()}))
                logging.warning('Proceeding with forward fills to remove remaining NaNs')

            if range_avail:
                # ### Calculate sigmas
                sigmas = np.abs(np.log(open_prices.astype(float)) - np.log(close_prices.astype(float)))

            if volume_avail:
                # #### Calculate volumes
                # Make volumes in dollars
                volumes = volumes * prices

            # Forward fill any gaps
            prices = prices.fillna(method='ffill')
            if range_avail:
                open_prices = open_prices.fillna(method='ffill')
                close_prices = close_prices.fillna(method='ffill')
                sigmas = sigmas.fillna(method='ffill')
            if volume_avail:
                volumes = volumes.fillna(method='ffill')

            # Also remove the first row just in case it had gaps since we can't forward fill it
            prices = prices.iloc[1:]
            if range_avail:
                open_prices = open_prices.iloc[1:]
                close_prices = close_prices.iloc[1:]
                sigmas = sigmas.iloc[1:]
            if volume_avail:
                volumes = volumes.iloc[1:]

            # At this point there should be no NaNs remaining
            if sum([x.isnull().sum().sum() for x in [prices]]) != 0:
                logging.warning(pd.DataFrame({'remaining nan price': prices.isnull().sum()}))

            # #### Compute returns
            returns = (prices.diff() / prices.shift(1)).fillna(method='ffill').iloc[1:]

            # Remove USDOLLAR except from returns
            prices = prices.iloc[:, :-1]
            if range_avail:
                open_prices = open_prices.iloc[:, :-1]
                close_prices = close_prices.iloc[:, :-1]
                sigmas = sigmas.iloc[:, :-1]
            if volume_avail:
                volumes = volumes.iloc[:, :-1]

            # Save all calculated data
            self.set('prices', prices, data_type='realized', sampling_freq=freq)
            self.set('returns', returns, data_type='realized', sampling_freq=freq)
            if range_avail:
                self.set('open_prices', open_prices, data_type='realized', sampling_freq=freq)
                self.set('close_prices', close_prices, data_type='realized', sampling_freq=freq)
                self.set('sigmas', sigmas, data_type='realized', sampling_freq=freq)
            if volume_avail:
                self.set('volumes', volumes, data_type='realized', sampling_freq=freq)

        return True

    def __fetch_factor_data(self, data_set='Developed_5_Factors_Daily'):
        """
        Training function for model
        :return:
        """
        if 'factors' in self.cfg['covariance']:
            data_set = self.cfg['covariance']['factors']

        ds = pdr.DataReader(data_set, 'famafrench', start=self.cfg['start_date'], end=self.cfg['end_date'])
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
        self.__state = ModelState.PREDICTED

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
