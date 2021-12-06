"""
Data Source - Data Input Manager
"""
import logging
import numpy as np
import pandas as pd
import quandl

from abc import ABCMeta, abstractmethod
from datetime import datetime
from enum import Enum

__all__ = ['TimeSeriesDataSetType', 'TimeSeriesDataSet', 'CsvTimeSeriesDataSet', 'QuandlTimeSeriesDataSet',
           'QuandlSamplingFrequency']


class TimeSeriesDataSetType(Enum):
    QUANDL = 0
    FAMA_FRENCH = 1
    CSV = 2


class QuandlSamplingFrequency(Enum):
    DAY = 'daily'
    WEEK = 'weekly'
    MONTH = 'monthly'
    QUARTER = 'quarterly'


class TimeSeriesDataSet(metaclass=ABCMeta):
    """
    Data Set Class
    """
    @classmethod
    def init(cls, config):
        """
        :param config: config dict
        """
        if 'data' in config:
            config = config['data']
        if isinstance(config['source'], str):
            ds_type = TimeSeriesDataSetType[config['source'].upper()]
        elif isinstance(config['source'], int):
            ds_type = TimeSeriesDataSetType(config['source'])
        else:
            raise ValueError('Data source type input can only be string or int.')

        if ds_type == TimeSeriesDataSetType.CSV:
            return CsvTimeSeriesDataSet(config)
        elif ds_type == TimeSeriesDataSetType.QUANDL:
            return QuandlTimeSeriesDataSet(config)
        else:
            raise NotImplementedError('{} data source has not been implemented yet.')

    @abstractmethod
    def get(self, financial_asset, start_date, end_date, **kwargs):
        """

        :param financial_asset:
        :param start_date:
        :param end_date:
        :return:
        """
        pass


class CsvTimeSeriesDataSet(TimeSeriesDataSet):
    """
    Data Set - Csv Type
    """
    def __init__(self, config):
        """
        Initialize from config
        :param config:
        """
        if 'name' not in config:
            raise ValueError('Quandl data source requires \'name\' to be configured.')
        elif 'path' not in config:
            raise ValueError('Csv data source requires data \'path\' to be configured.')

        # Public
        self.name = config['name']
        self.columns = []
        self.na_threshold_asset = config['na_threshold_asset'] if 'na_threshold_asset' in config else 0.05
        self.na_threshold_date = config['na_threshold_date'] if 'na_threshold_date' in config else 0.9
        self.return_check_on = config['return_check_on'] if 'return_check_on' in config else True
        self.return_min = config['return_min'] if 'return_min' in config else -0.5
        self.return_max = config['return_max'] if 'return_max' in config else 2

        # Private
        self.__path = config['path']
        self.__dt_col = config['dt_column']
        self.__dt_format = config['dt_format']

    def to_csv_ticker(self, ticker):
        """
        Converts ticker to format in CSV storage
        """
        if 'USDOLLAR' not in ticker:
            return ticker
        else:
            return 'DTB3'

    def get(self, financial_asset, start_date, end_date, cols=None, freq=None):
        """

        :param financial_asset:
        :param start_date:
        :param end_date:
        :param cols:
        :return:
        """
        # Read in data, process date, filter by date range
        loaded = pd.read_csv(self.__path + '/' + str(self.to_csv_ticker(financial_asset)) + '.csv')
        loaded.loc[:, self.__dt_col] = [datetime.strptime(dt, self.__dt_format) for dt in loaded.loc[:, self.__dt_col]]
        loaded.set_index(self.__dt_col, inplace=True)
        loaded = loaded[(loaded.index >= start_date) & (loaded.index <= end_date)]

        # Apply required data frequency
        if freq is not None:
            if freq == 'daily':
                loaded = loaded.resample('D').agg('last')
            elif freq == 'weekly':
                loaded = loaded.resample('W').agg('last')
            elif freq == 'monthly':
                loaded = loaded.resample('M').agg('last')
            elif freq == 'quarterly':
                loaded = loaded.resample('Q').agg('last')
            else:
                raise ValueError('CSV data source only supports freq: daily, weekly, monthly or quarterly.')

        # Replace all misformatted data with NaN
        loaded = loaded.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Extract columns of interest
        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            loaded = loaded[cols]
            self.columns = cols
        else:
            self.columns = loaded.columns
        return loaded


class QuandlTimeSeriesDataSet(TimeSeriesDataSet):
    """
    Data Set - Quandl Type
    """
    def __init__(self, config):
        """
        Initialize from config
        :param config:
        """
        if 'name' not in config:
            raise ValueError('Quandl data source requires \'name\' to be configured.')
        elif 'table' not in config:
            raise ValueError('Quandl data source requires \'table\' to be configured.')
        elif 'api_key' not in config:
            raise ValueError('Quandl data source requires \'api_key\' to be configured.')

        # Public
        self.name = config['name']
        self.table = config['table']
        self.columns = []
        self.na_threshold_asset = config['na_threshold_asset'] if 'na_threshold_asset' in config else 0.05
        self.na_threshold_date = config['na_threshold_date'] if 'na_threshold_date' in config else 0.9
        self.return_check_on = config['return_check_on'] if 'return_check_on' in config else True
        self.return_min = config['return_min'] if 'return_min' in config else -0.5
        self.return_max = config['return_max'] if 'return_max' in config else 2

        # Public optional
        self.freq = config['freq'] if 'freq' in config else None

        # Private
        self.__api_key = config['api_key']

    def to_quandl_ticker(self, ticker):
        """
        Converts ticker to format in Quandl dataset
        """
        if 'USDOLLAR' not in ticker:
            return self.table + '/' + ticker.replace('.', '_')
        else:
            return 'FRED/DTB3'

    def get(self, financial_asset, start_date, end_date, cols=None, freq=None):
        """

        :param financial_asset:
        :param start_date:
        :param end_date:
        :param cols:
        :param freq:
        :return:
        """
        # Input validation
        if not freq:
            if self.freq:
                freq = self.freq
            else:
                freq = QuandlSamplingFrequency.DAY

        if type(freq) == QuandlSamplingFrequency:
            freq_str = freq.value
        elif type(freq) == str and type(QuandlSamplingFrequency(freq)) == QuandlSamplingFrequency:
            freq_str = freq
        else:
            raise NotImplemented(
                'Sampling frequency of %s not implemented yet, please use QuandlSamplingFrequency' % str(freq))

        if '#' in financial_asset:
            # This is likely a commented line, let's filter it out
            return None

        try:
            if cols is None:
                result = quandl.get(self.to_quandl_ticker(financial_asset), start_date=start_date, end_date=end_date,
                                    api_key=self.__api_key, collapse=freq_str)
                if type(self.columns) != pd.Index:
                    self.columns = result.columns
                return result
            else:
                if type(cols) == str:
                    cols = [cols]
                if type(self.columns) != pd.Index:
                    self.columns = cols
                return quandl.get(self.to_quandl_ticker(financial_asset), start_date=start_date, end_date=end_date,
                                  api_key=self.__api_key, collapse=freq_str)[cols]
        except quandl.NotFoundError as e:
            logging.warning('quandl.get: %s is not valid - %s' % (self.to_quandl_ticker(financial_asset), e.__str__()))
            return None


if __name__ == '__main__':
    import yaml
    from datetime import datetime

    with open('../examples/cvxpt_ewm.yml') as cfg_file:
        yml_cfg = yaml.load(cfg_file, yaml.SafeLoader)
        yml_cfg['alpha']['data']['table'] = 'FRED'
        ds = TimeSeriesDataSet.init(yml_cfg['alpha'])

    spy_data = ds.get('SPY', datetime(2017, 1, 1), datetime(2018, 12, 31), ['Adj_Close', 'Adj_Volume'], 'monthly')
    fred_data = ds.get('USDOLLAR', datetime(1997, 1, 1), datetime(2019, 12, 31), None, 'daily')
    print(spy_data.tail())
    print(fred_data.tail())
