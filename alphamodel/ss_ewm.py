"""
Single stock returns - exponentially weighted moving average model
"""

import logging
import numpy as np
import pandas as pd
import seaborn as sns

from .model import Model, SamplingFrequency
from sklearn import linear_model, metrics

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
        if 'halflife' not in self.cfg or 'min_periods' not in self.cfg:
            raise ValueError('SingleStockEWM: Model config requires both min_periods (periods backwards) and a '
                             'halflife (decay of historical values, periods to half original value) to run.')

        # ## Load up model configs
        halflife = self.cfg['halflife']
        min_periods = self.cfg['min_periods']

        # ## Estimates
        # Returns
        realized_returns = self.get('returns', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_volumes = self.get('volumes', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_sigmas = self.get('sigmas', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        logging.info("Typical variance of returns: %g" % realized_returns.var().mean())

        self.set('returns', realized_returns.ewm(halflife=halflife, min_periods=min_periods).mean().shift(1).dropna(),
                 'predicted')

        # Volumes & sigmas
        self.set('volumes', realized_volumes.ewm(halflife=halflife, min_periods=min_periods).mean().shift(1).dropna(),
                 'predicted')
        self.set('sigmas', realized_sigmas.shift(1).dropna(), 'predicted')

        # Covariance
        if 'covariance' not in self.cfg:
            raise NotImplemented('Covariance section needs to be defined under SS EWM model config.')
        elif self.cfg['covariance']['method'] == 'SS':
            self.set('covariance', realized_returns.ewm(halflife=halflife, min_periods=min_periods).cov().
                     shift(realized_returns.shape[1]).dropna(), 'predicted', self.cfg['covariance']['sampling_freq'])
        elif self.cfg['covariance']['method'] == 'FF5':
            # Fetch data
            ff_returns = self.get('ff_returns', 'realized', SamplingFrequency.DAY)
            realized_returns = self.get('returns', data_type='realized', sampling_freq=SamplingFrequency.DAY)

            update = self.cfg['covariance']['update'] if 'update' in self.cfg['covariance'] else 'monthly'
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
                logging.info('Running for {}'.format(day.strftime('%Y %b %d')))

                # Grab asset returns for preceding train_days (90 by default)
                used_returns = realized_returns.loc[(realized_returns.index < day) &
                                           (realized_returns.index >= day - pd.Timedelta(str(days_back) + " days"))]
                used_ff_returns = ff_returns.loc[ff_returns.index.isin(used_returns.index)].iloc[:, :-1]

                # Multi linear regression to extract factor loadings
                mlr = linear_model.LinearRegression()
                mlr.fit(used_ff_returns, used_returns)
                mlr.predict(used_ff_returns)

                # Track performance of FF fit
                # rscore = metrics.r2_score(used_ff_returns, used_returns)
                cov_rscore.append(0)
                #p rint('predict_cov_FF5: mlr score = {s}'.format(s=rscore))

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

    @staticmethod
    def win_rate_symbol_horizon(returns_pred, returns_real, symbol, horizon):
        """
        Compute % of alpha values in the correct direction - 1 symbol, 1 horizon
        """
        # Return estimate at horizon
        if horizon > 1:
            returns = returns_pred[[symbol]].merge(returns_real[[symbol]].rolling(horizon).mean().shift(-horizon + 1),
                                                   suffixes=('_pred', '_real'),
                                                   left_index=True, right_index=True)
        else:
            returns = returns_pred[[symbol]].merge(returns_real[[symbol]],
                                                   suffixes=('_pred', '_real'),
                                                   left_index=True, right_index=True)

        # Comparison value
        return np.sum(np.sign(returns.loc[:, symbol + '_pred']) == np.sign(returns.loc[:, symbol + '_real'])) \
               / returns.shape[0]

    @staticmethod
    def win_rate(returns_pred, returns_real, symbol=None, horizon=None):
        """
        Compute % of alpha values in the correct direction - sample horizons, all symbols
        """
        # Input processing
        if not horizon:
            horizons = [1, 3, 5, 10, 20, 40, 60, 90, 120]
        elif type(symbol) == str:
            return SingleStockEWM.win_rate_symbol_horizon(returns_pred, returns_real, symbol, horizon)
        else:
            horizons = [horizon]

        # Data frame skeleton
        win_rate_all = pd.DataFrame(index=horizons)

        # Compute win rate for each symbol
        for symbol in [col for col in returns_pred.columns if col != 'USDOLLAR']:
            win_rate = []
            for horizon in horizons:
                win_rate.append(SingleStockEWM.win_rate_symbol_horizon(returns_pred, returns_real, symbol, horizon))
            win_rate_all[symbol] = win_rate

        # Compute statistics across all symbols
        win_rate_all = win_rate_all.agg(['mean', 'std'], axis=1).merge(win_rate_all, left_index=True, right_index=True)

        # Formatting
        cm = sns.light_palette("green", as_cmap=True)
        return win_rate_all.style.background_gradient(cmap=cm).format("{:.1%}")

    def prediction_quality(self, statistic='win_rate', **kwargs):
        """
        Compute prediction quality
        :param statistic:
        :return:
        """
        if statistic == 'win_rate':
            realized_returns = self.get('returns', data_type='realized',
                                        sampling_freq=self.cfg['returns']['sampling_freq'])
            predicted_returns = self.get('returns', data_type='predicted',
                                         sampling_freq=self.cfg['returns']['sampling_freq'])

            return SingleStockEWM.win_rate(predicted_returns, realized_returns, **kwargs)

    def show_results(self):
        pass


if __name__ == '__main__':
    ss_ewm_model = SingleStockEWM('../examples/cvxpt_ewm.yml')
    ss_ewm_model.train(force=True)
    ss_ewm_model.predict()
    ss_ewm_model.prediction_quality()
