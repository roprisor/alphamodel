"""
Single stock returns - Hidden Markov Model
"""

import numpy as np
import pandas as pd
import seaborn as sns

from .model import Model, ModelState, SamplingFrequency
from .scenario import Scenario
from datetime import timedelta
from sklearn import linear_model
from hmmlearn import hmm

__all__ = ['SingleStockHMM']


class SingleStockHMM(Model):

    def train(self, force=False, **kwargs):
        """
        Training function for model
        :return:
        """
        success = self._fetch_base_data(force)
        if success:
            self.__state = ModelState.TRAINED

        return success

    def predict(self, mode='e', threshold=0.8, **kwargs):
        """
        Prediction function for model, for out of sample historical test set
        :param mode:    e = expected return & sigma (probability weighted)
                        t = state with probability over threshold return & sigma
        :param threshold: probability threshold for state to be fully selected

        :return: n/a (all data stored in self.predicted)
        """
        # Input validation
        if 'train_len' not in self.cfg or 'hidden_states' not in self.cfg or 'halflife' not in self.cfg:
            raise ValueError('SingleStockHMM: Model config requires:\n - train_len (periods of dataset to train on)\n'
                             '- hidden_states (how many states should be fit in the model)\n'
                             '- halflife (decay of historical values, periods to half original value)')

        # ## Load up model configs
        halflife = self.cfg['halflife']
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
        sigmas_pred = pd.DataFrame(index=test_set.index)
        state0_pred = pd.DataFrame(index=test_set.index)
        confidence_pred = pd.DataFrame(index=test_set.index)
        hmm_storage = {}

        symbol_idx = 0
        test_idx = start_idx_test

        # For each asset
        while symbol_idx < len(self._universe):
            print('Running for ticker: {s}'.format(s=self._universe[symbol_idx]))

            sym_return_pred = []
            sym_sigma_pred = []
            sym_state0_pred = []
            sym_confidence_pred = []

            # For each test date in the test_set
            while test_idx < end_idx_test:
                # Grab train_set
                # Which one is better?
                train_set_idx = realized_returns.iloc[(test_idx - train_len):test_idx, symbol_idx]
                # train_set_idx = returns.iloc[0:test_idx, symbol_idx]

                # Fit the returns to an HMM model
                train_set_hmm = train_set_idx.values.reshape(-1, 1)
                regime_model = hmm.GaussianHMM(n_components=hidden_states, n_iter=200, tol=0.01)
                regime_model.fit(train_set_hmm)
                logprob, state_proba = regime_model.score_samples(train_set_hmm)
                # regime = regime_model.predict(train_set_hmm)

                # Predict next returns, covariances & add to prediction list
                if mode == 'e':
                    # Expected return & sigma
                    sym_return_pred.append(float(regime_model.means_.T.dot(state_proba[-1])))  # proba weighted mean
                    sym_sigma_pred.append(float(regime_model.covars_.T.dot(state_proba[-1])))  # proba weighted sigma
                elif mode == 't':
                    # State return & sigma - if state passes threshold hurdle
                    if state_proba[-1][0] > threshold:
                        sym_return_pred.append(float(regime_model.means_[0]))
                        sym_sigma_pred.append(float(regime_model.covars_[0][0]))
                    elif state_proba[-1][1] > threshold:
                        sym_return_pred.append(float(regime_model.means_[1]))
                        sym_sigma_pred.append(float(regime_model.covars_[1][0]))
                    else:
                        sym_return_pred.append(float(regime_model.means_.T.dot(state_proba[-1])))  # proba weighted mean
                        sym_sigma_pred.append(
                            float(regime_model.covars_.T.dot(state_proba[-1])))  # proba weighted sigma
                else:
                    raise ValueError('mode: {} not supported, please check function def for allowed values'.
                                     format(mode))

                # Compute confidence for input into Black Litterman
                sym_confidence_pred.append(sum(state_proba[-1] ** 4))
                sym_state0_pred.append(state_proba[-1][0])

                # Store model for later use
                if realized_returns.index[test_idx] not in hmm_storage:
                    hmm_storage[realized_returns.index[test_idx]] = {}
                hmm_storage[realized_returns.index[test_idx]][self._universe[symbol_idx]] = regime_model

                # Loop or no loop?
                # Break here to show results if requested to do so at a particular index
                # break
                test_idx += 1

            # Store prediction results for this symbol
            print('\n{sym} return pred length: {len}'.format(sym=self._universe[symbol_idx],
                                                             len=len(sym_return_pred)))
            print('\n{sym} return real length: {len}'.format(sym=self._universe[symbol_idx],
                                                             len=returns_pred.index.shape[0]))
            returns_pred[self._universe[symbol_idx]] = sym_return_pred
            sigmas_pred[self._universe[symbol_idx]] = sym_sigma_pred
            state0_pred[self._universe[symbol_idx]] = sym_state0_pred
            confidence_pred[self._universe[symbol_idx]] = sym_confidence_pred

            # Loop or no loop?
            # break
            symbol_idx += 1
            test_idx = start_idx_test

        self.set('returns', returns_pred, 'predicted')
        self.set('returns_hmm', returns_pred, 'predicted')
        self.set('sigmas_hmm', sigmas_pred, 'predicted')
        self.set('state0_prob', state0_pred, 'predicted')
        self.set('confidence', confidence_pred, 'predicted')
        self.set('models_hmm', hmm_storage, 'predicted')

        # ## Estimates - Volumes and Sigmas
        self.set('volumes', realized_volumes.ewm(halflife=halflife, min_periods=10).mean().shift(1).dropna(),
                 'predicted')
        self.set('sigmas', realized_sigmas.shift(1).dropna(), 'predicted')

        # ## Estimates - Covariance
        if 'covariance' not in self.cfg:
            raise NotImplemented('Covariance section needs to be defined under SS EWM model config.')
        elif self.cfg['covariance']['method'] == 'SS':
            self.set('covariance', realized_returns.ewm(halflife=halflife, min_periods=10).cov().
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
                # TODO use HMM predicted stock var
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
            raise NotImplemented('Covariance section needs to be defined under ss_hmm model config and needs either:\n'
                                 ' - SS (single stock returns)\n'
                                 '- FF5 (Fama French 5 factor returns).')

        self.__state = ModelState.PREDICTED
        return True

    def predict_next(self):
        pass

    def generate_forward_scenario(self, dt, horizon, mode='eg', threshold=0.8):
        """
        Generate forward scenario
        :param dt: datetime to start at
        :param horizon: periods ahead to be included in the scenario
        :param mode:    eg = expected Gaussian (Gaussian of expected return & sigma)
                        lg = likely Gaussian (highest likelihood return & sigma)
                        er = expected return (constant)
                        hmm = HMM (fit parameters to historical training data)
        :param threshold: likelihood threshold for 'lg' mode to pick 1 state
        :return:
        """
        if self.state != ModelState.PREDICTED:
            raise ValueError('generate_forward_scenario: Unable to run if model is not in predicted state.')

        # Grab needed inputs
        volumes_pred = self.get('volumes', 'predicted')
        sigmas_pred = self.get('sigmas', 'predicted')
        returns_hmm = self.get('returns_hmm', 'predicted')
        sigmas_hmm = self.get('sigmas_hmm', 'predicted')
        state0_hmm = self.get('state0_prob', 'predicted')
        hmm_dt = self.get('models_hmm', 'predicted')[dt]

        # Generate indices (dates)
        dt_index = volumes_pred.index.get_loc(dt)
        avail_dates = volumes_pred.shape[0] - dt_index
        if avail_dates >= horizon:
            indices = volumes_pred.index[dt_index:(dt_index + horizon)]
        else:
            real_dates = volumes_pred.index[dt_index:(dt_index + avail_dates)]
            potential_dates = pd.date_range(start=real_dates[-1],
                                            end=real_dates[-1] + timedelta(days=horizon - avail_dates))
            indices = real_dates.union(potential_dates)

        # Generate returns
        returns = pd.DataFrame(index=indices)
        for symbol in self._universe:
            if mode == 'hmm':
                # HMM model generated samples
                returns.loc[:, symbol], _ = hmm_dt[symbol].sample(horizon)

            elif mode == 'eg':
                # Gaussian generated samples (expected return & sigmas)
                # Expected return & sigma
                sym_return = returns_hmm.loc[dt, symbol]
                sym_sigma = sigmas_hmm.loc[dt, symbol]

                # Generate a Gaussian distribution of horizon length
                rng = np.random.default_rng()
                samples = rng.normal(sym_return, sym_sigma, horizon)
                returns.loc[:, symbol] = samples

            elif mode == 'lg':
                # Gaussian generated samples (most likely Gaussian distribution)
                # Extract likelihood of state 0
                sym_state0_prob = state0_hmm.loc[dt, symbol]
                hmm_model = hmm_dt[symbol]

                # Choose return/sigma based on state likelihood
                if sym_state0_prob >= threshold:
                    sym_return = hmm_model.means_[0]
                    sym_sigma = hmm_model.covars_[0][0]
                elif sym_state0_prob <= (1 - threshold):
                    sym_return = hmm_model.means_[1]
                    sym_sigma = hmm_model.covars_[1][0]
                else:
                    sym_return = returns_hmm.loc[dt, symbol]
                    sym_sigma = sigmas_hmm.loc[dt, symbol]

                # Generate a Gaussian distribution of horizon length
                rng = np.random.default_rng()
                samples = rng.normal(sym_return, sym_sigma, horizon)
                returns.loc[:, symbol] = samples

            elif mode == 'er':
                # Constant expected return
                sym_return = returns_hmm.loc[dt, symbol]
                returns.loc[:, symbol] = sym_return

        # Generate volumes
        volumes = pd.DataFrame(index=indices)
        for i in indices:
            for symbol in self.universe:
                volumes.loc[i, symbol] = volumes_pred.loc[dt, symbol]

        # Generate sigmas
        sigmas = pd.DataFrame(index=indices)
        for i in indices:
            for symbol in self.universe:
                sigmas.loc[i, symbol] = sigmas_pred.loc[dt, symbol]

        # Create a scenario from the inputs
        return Scenario(dt, horizon, returns, volumes, sigmas)

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
