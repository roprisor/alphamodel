"""
Single stock returns - Hidden Markov Model
"""

import numpy as np
import pandas as pd
import seaborn as sns

from .model import Model, SamplingFrequency
from sklearn import linear_model
from hmmlearn import hmm

__all__ = ['SingleStockHMM']


class SingleStockHMM(Model):

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
        if 'train_len' not in self.cfg or 'hidden_states' not in self.cfg or 'alpha' not in self.cfg:
            raise ValueError('SingleStockHMM: Model config requires:\n - train_len (periods of dataset to train on)\n'
                             '- hidden_states (how many states should be fit in the model)\n'
                             '- alpha (EWM decay for time series)')

        # ## Load up model configs
        alpha = self.cfg['alpha']
        train_len = self.cfg['train_len']
        hidden_states = self.cfg['hidden_states']

        # ## Estimates - Returns
        realized_returns = self.get('returns', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_volumes = self.get('volumes', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_sigmas = self.get('sigmas', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        print("Typical variance of returns: %g" % realized_returns.var().mean())

        # Split data in train & test data sets
        start_idx_train = 0
        end_idx_train = train_len
        start_idx_test = end_idx_train
        end_idx_test = realized_returns.shape[0]

        test_set = realized_returns.iloc[start_idx_test:end_idx_test]

        returns_pred = pd.DataFrame(index=test_set.index)
        confidence_pred = pd.DataFrame(index=test_set.index)

        symbol_idx = 0
        test_idx = start_idx_test

        # For each asset
        while symbol_idx < len(self.universe):
            print('Running for ticker: {s}'.format(s=self.universe[symbol_idx]))

            sym_return_pred = []
            sym_confidence_pred = []

            # For each test date in the test_set
            while test_idx < end_idx_test:
                # Current iteration
                # print("Running for index: " + str(test_idx))

                # Grab train_set
                # Which one is better?
                train_set_idx = realized_returns.iloc[(test_idx - train_len):test_idx, symbol_idx]
                # train_set_idx = returns.iloc[0:test_idx, symbol_idx]

                # Fit the returns to an HMM model
                train_set_hmm = train_set_idx.values.reshape(-1, 1)
                regime_model = hmm.GaussianHMM(n_components=hidden_states, n_iter=200)
                regime_model.fit(train_set_hmm)
                state_proba = regime_model.predict_proba(train_set_hmm)
                regime = regime_model.predict(train_set_hmm)

                # Predict next returns, covariances & add to prediction list
                sym_return_pred.append(float(regime_model.means_.T.dot(state_proba[-1])))  # expected mean

                # Check here how frequent to switch state - toggle regimes
                # if state_proba[-1][0] > 0.7:
                #    sym_return_pred.append(float(regime_model.means_[0]))
                # elif state_proba[-1][0] > 0.7:
                #    sym_return_pred.append(float(regime_model.means_[1]))
                # else:
                #    sym_return_pred.append(0)

                # Use confidence as input into Black Litterman
                sym_confidence_pred.append(sum(state_proba[-1] ** 4))

                # Loop or no loop?
                # Break here to show results if requested to do so at a particular index
                # break
                test_idx += 1

            # Store prediction results for this symbol
            print('\n{sym} return pred length: {len}'.format(sym=self.universe[symbol_idx],
                                                             len=len(sym_return_pred)))
            print('\n{sym} return real length: {len}'.format(sym=self.universe[symbol_idx],
                                                             len=returns_pred.index.shape[0]))
            returns_pred[self.universe[symbol_idx]] = sym_return_pred
            confidence_pred[self.universe[symbol_idx]] = sym_confidence_pred

            # Loop or no loop?
            # break
            symbol_idx += 1
            test_idx = start_idx_test

        self.set('returns', returns_pred, 'predicted')

        # ## Estimates - Volumes and Sigmas
        self.set('volumes', realized_volumes.ewm(alpha=alpha, min_periods=10).mean().shift(1).dropna(), 'predicted')
        self.set('sigmas', realized_sigmas.shift(1).dropna(), 'predicted')

        # ## Estimates - Covariance
        if 'covariance' not in self.cfg:
            raise NotImplemented('Covariance section needs to be defined under SS EWM model config.')
        elif self.cfg['covariance']['method'] == 'SS':
            self.set('covariance', realized_returns.ewm(alpha=alpha, min_periods=10).cov().
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
            raise NotImplemented('Covariance section needs to be defined under ss_hmm moodel config and needs either:\n'
                                 ' - SS (single stock returns)\n'
                                 '- FF5 (Fama French 5 factor returns).')

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
            return SingleStockHMM.win_rate_symbol_horizon(returns_pred, returns_real, symbol, horizon)
        else:
            horizons = [horizon]

        # Data frame skeleton
        win_rate_all = pd.DataFrame(index=horizons)

        # Compute win rate for each symbol
        for symbol in returns_pred.columns:
            win_rate = []
            for horizon in horizons:
                win_rate.append(SingleStockHMM.win_rate_symbol_horizon(returns_pred, returns_real, symbol, horizon))
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

            return SingleStockHMM.win_rate(predicted_returns, realized_returns, **kwargs)

    def show_results(self):
        pass


if __name__ == '__main__':
    ss_hmm_model = SingleStockHMM('../examples/cvxpt_hmm.yml')
    ss_hmm_model.train(force=True)
    ss_hmm_model.predict()
    ss_hmm_model.prediction_quality()
