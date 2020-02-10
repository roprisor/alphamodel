"""
Single stock returns - exponentially weighted moving average model
"""

import numpy as np
import pandas as pd

from .model import Model
from sklearn import linear_model

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
        # Returns
        realized_returns = self.realized['returns']
        print("Typical variance of returns: %g" % realized_returns.var().mean())

        self.predicted['returns'] = realized_returns.ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).dropna()

        # Volumes & sigmas
        self.predicted['volumes'] = self.realized['volumes'].ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).\
            dropna()
        self.predicted['sigmas'] = self.realized['sigmas'].shift(1).dropna()

        # Covariance
        if 'covariance' not in self.cfg:
            raise NotImplemented('Covariance section needs to be defined under SS EWM model config.')
        elif self.cfg['covariance']['method'] == 'SS':
            self.predicted['covariance'] = realized_returns.ewm(alpha=alpha, min_periods=min_periods).cov(). \
                shift(realized_returns.shape[1]).dropna()
        elif self.cfg['covariance']['method'] == 'FF5':
            # Fetch data
            ff_returns = self.realized['ff_returns']
            freq = self.cfg['covariance']['freq']
            if freq == 'monthly':
                freq_str = 'M'
            elif freq == 'biweekly':
                freq_str = '2W'
            elif freq == 'weekly':
                freq_str = 'W'
            else:
                raise NotImplemented('Freq under covariance only supports: month, biweekly, weekly.')

            # Generate computation frequency
            first_days = pd.date_range(start=realized_returns.index[max(self.cfg['min_periods'] + 1, 90)],
                                       end=realized_returns.index[-1],
                                       freq=freq_str)

            # Use ML regression to obtain factor loadings. Then factor covariance and stock idiosyncratic variances
            exposures, factor_sigma, idyos = {}, {}, {}

            # Every first day in each biweekly period
            for day in first_days:
                print('Running for {}'.format(day.strftime('%Y %b %d')))

                # Grab asset returns for preceding 90 days
                used_returns = realized_returns.loc[(realized_returns.index < day) &
                                           (realized_returns.index >= day - pd.Timedelta("90 days"))]
                used_ff_returns = ff_returns.loc[ff_returns.index.isin(used_returns.index)].iloc[:, :-1]

                # Multi linear regression to extract factor loadings
                mlr = linear_model.LinearRegression()
                mlr.fit(used_ff_returns, used_returns)
                mlr.predict(used_ff_returns)
                print('predict_cov_FF5: mlr score = {s}'.format(s=mlr.score(used_ff_returns, used_returns)))

                # Factor covariance - on EWMA of FF returns
                factor_sigma[day] = used_ff_returns.cov().fillna(0)
                # Exposures - factor loadings obtained from multi linear regression coefficients of stock on FF factors
                exposures[day] = pd.DataFrame(data=mlr.coef_, index=realized_returns.columns).fillna(0)
                # Stock idiosyncratic variances - stock var minus FF var, ensure >=0
                idyos[day] = pd.Series(np.diag(used_returns.cov().values -
                                               exposures[day].values @ factor_sigma[day].values @ exposures[
                                                   day].values.T),
                                       index=realized_returns.columns).fillna(method='ffill')
                idyos[day][idyos[day] < 0] = 0

            self.predicted['factor_sigma'] = pd.concat(factor_sigma.values(), axis=0, keys=factor_sigma.keys())
            self.predicted['exposures'] = pd.concat(exposures.values(), axis=0, keys=exposures.keys())
            self.predicted['idyos'] = pd.DataFrame(idyos).T
        else:
            raise NotImplemented('Covariance section needs to be defined under SSEWM moodel config and needs definition'
                                 ' of method: SS (single stock returns) or FF5 (Fama French 5 factor returns).')

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
