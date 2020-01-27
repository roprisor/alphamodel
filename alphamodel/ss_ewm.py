"""
Single stock returns - exponentially weighted moving average model
"""

import pandas as pd
import numpy as np
import pickle

from .model import Model
from os import path

__all__ = ['SingleStockEWM']


class SingleStockEWM(Model):

    def train(self, force=False):
        """
        Training function for model
        :return:
        """
        # If we can load past state from file, let's just do that
        if not force and self.load():
            return

        data = {}

        # #### Download loop

        # If Quandl complains about the speed of requests, try adding sleep time.
        # Download asset data & construct a data dictionary: {ticker: pd.DataFrame(price/volume)}
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
            f = open(self.filename, 'rb')
            tmp_dict = pickle.load(f)
            f.close()

            self.__dict__.clear()
            self.__dict__.update(tmp_dict)
            return True

        return False

    def predict(self):
        """
        Prediction function for model, for out of sample historical test set
        :return: n/a (all data stored in self.predicted)
        """
        # ## Load up model configs
        alpha = self.cfg['alpha']
        min_periods = self.cfg['min_periods']

        # ## Estimates
        realized_returns = self.realized['returns']
        print("Typical variance of returns: %g" % realized_returns.var().mean())

        self.predicted['returns'] = realized_returns.ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).dropna()
        self.predicted['volumes'] = self.realized['volumes'].ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).\
            dropna()
        self.predicted['sigmas'] = self.realized['sigmas'].shift(1).dropna()
        self.predicted['covariance'] = realized_returns.ewm(alpha=alpha, min_periods=min_periods).cov().\
            shift(realized_returns.shape[1]).dropna()

    def predict_next(self):
        pass

    def prediction_quality(self, statistic='correlation'):
        """
        Compute prediction quality
        :param statistic:
        :return:
        """
        agree_on_sign = np.sign(self.realized['returns'].iloc[60:, :-1]) ==\
            np.sign(self.predicted['returns'].iloc[:, :-1])
        print("Return predictions have the right sign %.1f%% of the times" %
              (100 * agree_on_sign.sum().sum() / (agree_on_sign.shape[0] * (agree_on_sign.shape[1] - 1))))
        pass

    def show_results(self):
        pass


if __name__ == '__main__':
    ss_ewm_model = SingleStockEWM('../examples/cvxpt_rebalance.yml')
    ss_ewm_model.train()
    ss_ewm_model.predict()
