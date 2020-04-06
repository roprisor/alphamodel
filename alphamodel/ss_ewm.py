"""
Single stock returns - exponentially weighted moving average model
"""

import numpy as np
import pandas as pd

from .model import Model, SamplingFrequency
from sklearn import linear_model

__all__ = ['SingleStockEWM']


class SingleStockEWM(Model):

    def train(self, force=False):
        """
        Training function for model
        :return:
        """
        return self._fetch_base_data(force)

    def predict(self):
        """
        Prediction function for model, for out of sample historical test set
        :return: n/a (all data stored in self.predicted)
        """
        # Input validation
        if 'alpha' not in self.cfg or 'min_periods' not in self.cfg:
            raise ValueError('SingleStockEWM: Model config requires both min_periods (periods backwards) and an '
                             'alpha value (decay of historical values, 0 to 1) to run.')

        # ## Load up model configs
        alpha = self.cfg['alpha']
        min_periods = self.cfg['min_periods']

        # ## Estimates
        # Returns
        realized_returns = self.get('returns', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_volumes = self.get('volumes', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_sigmas = self.get('sigmas', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        print("Typical variance of returns: %g" % realized_returns.var().mean())

        self.set('returns', realized_returns.ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).dropna(),
                 'predicted')

        # Volumes & sigmas
        self.set('volumes', realized_volumes.ewm(alpha=alpha, min_periods=min_periods).mean().shift(1).dropna(),
                 'predicted')
        self.set('sigmas', realized_sigmas.shift(1).dropna(), 'predicted')

        # Covariance
        if 'covariance' not in self.cfg:
            raise NotImplemented('Covariance section needs to be defined under SS EWM model config.')
        elif self.cfg['covariance']['method'] == 'SS':
            self.set('covariance', realized_returns.ewm(alpha=alpha, min_periods=min_periods).cov().
                     shift(realized_returns.shape[1]).dropna(), 'predicted', self.cfg['covariance']['sampling_freq'])
        elif self.cfg['covariance']['method'] == 'FF5':
            # Fetch data
            ff_returns = self.get('ff_returns', 'realized', SamplingFrequency.DAY)
            realized_returns = self.get('returns', data_type='realized', sampling_freq=SamplingFrequency.DAY)

            update = self.cfg['covariance']['sampling_freq'] if 'sampling_freq' in self.cfg['covariance'] else 'monthly'
            if update == 'quarterly':
                update_freq = '3M'
            elif update == 'monthly':
                update_freq = 'M'
            elif update == 'biweekly':
                update_freq = '2W'
            elif update == 'weekly':
                update_freq = 'W'
            else:
                raise NotImplemented('Update freq under covariance only supports: month, biweekly, weekly.')

            # Generate computation frequency
            first_days = pd.date_range(start=realized_returns.index[max(self.cfg['min_periods'] + 1, 90)],
                                       end=realized_returns.index[-1],
                                       freq=update_freq)
            days_back = self.cfg['covariance']['train_days'] if 'train_days' in self.cfg['covariance'] else 90

            # Use ML regression to obtain factor loadings. Then factor covariance and stock idiosyncratic variances
            exposures, factor_sigma, idyos = {}, {}, {}

            # Every first day in each biweekly period
            cov_rscore = []
            for day in first_days:
                print('Running for {}'.format(day.strftime('%Y %b %d')))

                # Grab asset returns for preceding train_days (90 by default)
                used_returns = realized_returns.loc[(realized_returns.index < day) &
                                           (realized_returns.index >= day - pd.Timedelta(str(days_back) + " days"))]
                used_ff_returns = ff_returns.loc[ff_returns.index.isin(used_returns.index)].iloc[:, :-1]

                # Multi linear regression to extract factor loadings
                mlr = linear_model.LinearRegression()
                mlr.fit(used_ff_returns, used_returns)
                mlr.predict(used_ff_returns)

                # Track performance of FF fit
                rscore = mlr.score(used_ff_returns, used_returns)
                cov_rscore.append(rscore)
                print('predict_cov_FF5: mlr score = {s}'.format(s=rscore))

                # Factor covariance - on FF returns
                factor_sigma[day] = used_ff_returns.cov().fillna(0)
                # Exposures - factor loadings obtained from multi linear regression coefficients of stock on FF factors
                exposures[day] = pd.DataFrame(data=mlr.coef_, index=realized_returns.columns).fillna(0)
                # Stock idiosyncratic variances - stock var minus FF var, ensure >=0
                idyos[day] = pd.Series(np.diag(used_returns.cov().values -
                                               exposures[day].values @ factor_sigma[day].values @ exposures[
                                                   day].values.T),
                                       index=realized_returns.columns).fillna(method='ffill')
                idyos[day][idyos[day] < 0] = 0

            self.set('factor_sigma', pd.concat(factor_sigma.values(), axis=0, keys=factor_sigma.keys()), 'predicted')
            self.set('exposures', pd.concat(exposures.values(), axis=0, keys=exposures.keys()), 'predicted')
            self.set('idyos', pd.DataFrame(idyos).T, 'predicted')
            self.set('cov_rscore', pd.DataFrame.from_dict({'date': first_days,
                                                           'rscore': cov_rscore,
                                                           'train_days': days_back}), 'predicted')

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
