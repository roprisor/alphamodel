"""
Single stock returns - exponentially weighted moving average model
"""

import numpy as np

from .model import Model

__all__ = ['SingleStockEWM']


class SingleStockEWM(Model):

    def train(self, force=False):
        """
        Training function for model
        :return:
        """
        return self._fetch_data(force)

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

        self.predicted['returns'] = realized_returns.ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).dropna()
        self.predicted['volumes'] = self.realized['volumes'].ewm(alpha=alpha, min_periods=min_periods).mean().shift(1). \
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
    ss_ewm_model = SingleStockEWM('../examples/cvxpt_rebalance.yml')
    ss_ewm_model.train()
    ss_ewm_model.predict()
