"""
Alpha Model Base Template
"""

import pandas as pd
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
