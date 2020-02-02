"""
Single stock returns - ex-post returns randomized by a scaling factor
"""

import numpy as np
import pickle

from .model import Model
from os import path

__all__ = ['SingleStockExPost']


class SingleStockExPost(Model):

    def train(self, force=False):
        """
        Training function for model
        :return:
        """
        return self._fetch_data(force)

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

    def returns_expost(self, returns):
        """
        Construct r_ex(horizon, lambda) = lambda * r_ex(horizon) + (1-lambda) * noise, where noise = N(0, sd(r_ex))

        :param returns:
        :return:
        """
        # Input validation
        if 'horizon' not in self.cfg or 'lambda' not in self.cfg:
            raise ValueError('SingleStockExPost: Model config requires both a horizon (periods forward) and a lambda '
                             'value (how much of expost returns to keep, 0 to 1) to run.')

        # r_ex(horizon)
        r_ex = returns.rolling(self.cfg['horizon']).sum()

        # noise, where noise = N(0, sd(r_ex(horizon))) - shape(r_ex)
        noise = np.random.normal([0] * len(r_ex.std()), r_ex.std(), r_ex.shape)

        return (self.cfg['lambda'] * r_ex).add((1 - self.cfg['lambda']) * noise).dropna()

    def predict(self):
        """
        Prediction function for model, for out of sample historical test set
        :return: n/a (all data stored in self.predicted)
        """
        # Input validation
        if 'alpha' not in self.cfg or 'min_periods' not in self.cfg:
            raise ValueError('SingleStockExPost: Model config requires both min_periods (periods backwards) and an '
                             'alpha value (decay of historical values, 0 to 1) to run.')

        # ## Load up model configs
        alpha = self.cfg['alpha']
        min_periods = self.cfg['min_periods']

        # ## Estimates
        realized_returns = self.realized['returns']
        print("Typical variance of returns: %g" % realized_returns.var().mean())

        self.predicted['returns'] = self.returns_expost(realized_returns)
        self.predicted['volumes'] = self.realized['volumes'].ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).\
            dropna()
        self.predicted['sigmas'] = self.realized['sigmas'].shift(1).dropna()
        self.predicted['covariance'] = realized_returns.ewm(alpha=alpha, min_periods=min_periods).cov(). \
            shift(realized_returns.shape[1]).dropna()

    def predict_next(self):
        pass

    def prediction_quality(self, statistic='correlation'):
        """
        Compute prediction quality
        :param statistic:
        :return:
        """
        agree_on_sign = np.sign(self.realized['returns'].iloc[:, :-1]) == \
                            np.sign(self.predicted['returns'].iloc[:, :-1])
        print("Return predictions have the right sign %.1f%% of the times" %
              (100 * agree_on_sign.sum().sum() / (agree_on_sign.shape[0] * (agree_on_sign.shape[1] - 1))))

    def show_results(self):
        pass


if __name__ == '__main__':
    ss_ep_model = SingleStockExPost('../examples/cvxpt_rebalance.yml')
    ss_ep_model.train(False)
    ss_ep_model.predict()
    ss_ep_model.prediction_quality()
