"""
Single stock returns - Hidden Markov Model
"""

import cvxportfolio as cp
import logging
import numpy as np
import pandas as pd
import seaborn as sns

from .model import Model, ModelState, SamplingFrequency
from .scenario import Scenario
from datetime import timedelta, datetime
from sklearn import linear_model, metrics
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

    def predict(self, mode='e', threshold=0.8, preprocess=None, **kwargs):
        """
        Prediction function for model, for out of sample historical test set
        :param mode:    e = expected return & sigma (probability weighted)
                        t = state with probability over threshold return & sigma
        :param threshold: probability threshold for state to be fully selected
        :param preprocess: preprocessing method
                            None: raw return data
                            'exponential': exponential decay

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

        # Save new configs for later
        self.cfg['mode'] = mode
        self.cfg['threshold'] = threshold
        self.cfg['preprocess'] = preprocess

        # ## Estimates - Returns
        realized_returns = self.get('returns', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_volumes = self.get('volumes', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        realized_sigmas = self.get('sigmas', data_type='realized', sampling_freq=self.cfg['returns']['sampling_freq'])
        logging.info("Typical variance of returns: %g" % realized_returns.var().mean())

        # Split data in train & test data sets
        start_idx_train = 0
        end_idx_train = train_len
        start_idx_test = end_idx_train
        end_idx_test = realized_returns.shape[0]

        test_set = realized_returns.iloc[start_idx_test:end_idx_test]

        returns_pred = pd.DataFrame(index=test_set.index)
        sigmas_pred = pd.DataFrame(index=test_set.index)
        state0_prob = pd.DataFrame(index=test_set.index)
        confidence_pred = pd.DataFrame(index=test_set.index)
        hmm_storage = {}

        symbol_idx = 0
        test_idx = start_idx_test

        # For each asset
        while symbol_idx < len(self._universe):
            logging.info('Running for ticker: {s}'.format(s=self._universe[symbol_idx]))

            sym_return_pred = []
            sym_sigma_pred = []
            sym_state0_prob = []
            sym_confidence_pred = []

            # For each test date in the test_set
            while test_idx < end_idx_test:
                # Grab train_set
                # Which one is better?
                train_set_idx = realized_returns.iloc[(test_idx - train_len):test_idx, symbol_idx]

                if preprocess == 'exponential':
                    train_set_idx = train_set_idx.ewm(halflife=halflife).mean()

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
                    # State return & sigma - if any current state passes threshold hurdle, use it
                    if state_proba[-1][0] > threshold:
                        sym_return_pred.append(float(regime_model.means_[0]))
                        sym_sigma_pred.append(float(regime_model.covars_[0][0]))
                    elif state_proba[-1][1] > threshold:
                        sym_return_pred.append(float(regime_model.means_[1]))
                        sym_sigma_pred.append(float(regime_model.covars_[1][0]))
                    # else default back to previous if available
                    else:
                        if sym_return_pred and sym_sigma_pred:
                            sym_return_pred.append(sym_return_pred[-1])
                            sym_sigma_pred.append(sym_sigma_pred[-1])
                        else:
                            # Expected return & sigma
                            sym_return_pred.append(
                                float(regime_model.means_.T.dot(state_proba[-1])))  # proba weighted mean
                            sym_sigma_pred.append(
                                float(regime_model.covars_.T.dot(state_proba[-1])))  # proba weighted sigma
                else:
                    raise ValueError('mode: {} not supported, please check function def for allowed values'.
                                     format(mode))

                # Compute confidence for input into Black Litterman
                sym_confidence_pred.append(sum(state_proba[-1] ** 4))
                sym_state0_prob.append(state_proba[-1][0])

                # Store model for later use
                if realized_returns.index[test_idx] not in hmm_storage:
                    hmm_storage[realized_returns.index[test_idx]] = {}
                hmm_storage[realized_returns.index[test_idx]][self._universe[symbol_idx]] = regime_model

                # Loop or no loop?
                # Break here to show results if requested to do so at a particular index
                # break
                test_idx += 1

            # Store prediction results for this symbol
            logging.debug('\n{sym} return pred length: {len}'.format(sym=self._universe[symbol_idx],
                                                                     len=len(sym_return_pred)))
            logging.debug('\n{sym} return real length: {len}'.format(sym=self._universe[symbol_idx],
                                                                     len=returns_pred.index.shape[0]))
            returns_pred[self._universe[symbol_idx]] = sym_return_pred
            sigmas_pred[self._universe[symbol_idx]] = sym_sigma_pred
            state0_prob[self._universe[symbol_idx]] = sym_state0_prob
            confidence_pred[self._universe[symbol_idx]] = sym_confidence_pred

            # Loop or no loop?
            # break
            symbol_idx += 1
            test_idx = start_idx_test

        self.set('returns', returns_pred, 'predicted')
        self.set('returns_hmm', returns_pred, 'predicted')
        self.set('sigmas_hmm', sigmas_pred, 'predicted')
        self.set('state0_prob', state0_prob, 'predicted')
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
                                                    (realized_returns.index >= day - pd.Timedelta(str(days_back)
                                                                                                  + " days"))]
                used_ff_returns = ff_returns.loc[ff_returns.index.isin(used_returns.index)].iloc[:, :-1]

                # Multi linear regression to extract factor loadings
                mlr = linear_model.LinearRegression()
                mlr.fit(used_ff_returns, used_returns)
                used_ret_pred = mlr.predict(used_ff_returns)

                # Track performance of FF fit
                rscore = metrics.r2_score(used_returns, used_ret_pred, multioutput='uniform_average')
                cov_rscore.append(rscore)
                logging.debug('predict_cov_FF5: mlr score = {s}'.format(s=rscore))

                # Factor covariance - on FF returns
                factor_sigma[day] = used_ff_returns.cov().fillna(0)
                # Exposures - factor loadings obtained from multi linear regression coefficients of stock on FF factors
                exposures[day] = pd.DataFrame(data=mlr.coef_, index=realized_returns.columns).fillna(0)
                # Stock idiosyncratic variances - HMM variance; if not avail, then stock var minus FF var, ensure >=0
                try:
                    t_sigma = cp.utils.time_locator(sigmas_pred, day)
                    idyos[day] = t_sigma
                except KeyError as e:
                    logging.debug('predict: Day index {} not found, defaulting to factor residual idyo variance'.
                                  format(str(day)))
                    idyos[day] = pd.Series(np.diag(used_returns.cov().values -
                                                   exposures[day].values @ factor_sigma[day].values @ exposures[
                                                       day].values.T),
                                           index=realized_returns.columns).fillna(method='ffill')
                idyos[day].loc[idyos[day] < 0] = 0

            self.set('factor_sigma', pd.concat(factor_sigma.values(), axis=0, keys=factor_sigma.keys()), 'predicted')
            self.set('exposures', pd.concat(exposures.values(), axis=0, keys=exposures.keys()), 'predicted')
            self.set('idyos', pd.DataFrame(idyos).T, 'predicted')
            self.set('cov_rscore', pd.DataFrame.from_dict({'date': first_days,
                                                           'rscore': cov_rscore,
                                                           'train_days': days_back}), 'predicted')

        else:
            raise NotImplemented('Covariance section needs to be defined under ss_hmm model config and needs either:\n'
                                 ' - SS (single stock returns)\n'
                                 ' - FF5 (Fama French 5 factor returns).')

        self.__state = ModelState.PREDICTED
        return True

    def predict_next(self):
        pass

    def generate_forward_scenario(self, dt, horizon, mode='eg', threshold=0.8):
        """
        Generate forward scenario
        :param dt: datetime to start at
        :param horizon: periods ahead to be included in the scenario
        :param mode:    eg = Gaussian sampled scenario (Gaussian of expected mean & sigma for values)
                        lg = likely Gaussian (highest likelihood return & sigma)
                        c = constant scenario
                        hmm = HMM sampled scenario (fit parameters to historical training data)
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

            elif mode == 'c':
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

    def win_rate(self, returns_pred, returns_real, symbol=None, horizon=None, print=True):
        """
        Compute % of alpha values in the correct direction - sample horizons, all symbols (-risk_free)
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
        for symbol in self.universe:
            win_rate = []
            for horizon in horizons:
                win_rate.append(SingleStockHMM.win_rate_symbol_horizon(returns_pred, returns_real, symbol, horizon))
            win_rate_all[symbol] = win_rate

        # Compute statistics across all symbols
        win_rate_all = win_rate_all.agg(['mean', 'std'], axis=1).merge(win_rate_all, left_index=True, right_index=True)

        # Formatting
        if print:
            cm = sns.light_palette("green", as_cmap=True)
            return win_rate_all.style.background_gradient(cmap=cm).format("{:.1%}")

        return win_rate_all

    def information_coef(self, returns_pred, returns_real, symbol=None, horizon=None, print=True):
        """
        Compute IC of alpha values - sample horizons, all symbols (-risk_free)
        """
        # 1. Process the win rate for the given inputs
        wr = self.win_rate(returns_pred, returns_real, symbol, horizon, print=False)

        # 2. Transform the win rate into information coefficient: IC = 2 * WR - 1
        wr = wr[self.universe]
        ic = 2 * wr - 1

        # Compute statistics across all symbols
        ic = ic.agg(['mean', 'std'], axis=1).merge(ic, left_index=True, right_index=True)

        # Formatting
        if print:
            cm = sns.light_palette("green", as_cmap=True)
            return ic.style.background_gradient(cmap=cm).format("{:.1f}")

        return ic

    def jitter(self, print=True):
        """
        Compute % of state jumps out of total periods
        """
        # Initialize data frame
        jitter = pd.DataFrame(index=['mean_changes', 'regime_changes',
                                     'mean_change_rate', 'regime_change_rate',
                                     'periods'])

        # Grab prediction since need to put into the context of the model
        returns_pred = self.get('returns', 'predicted')
        date_model_store = self.get('models_hmm', 'predicted')

        # For each symbol in the universe
        for symbol in self.universe:
            mean_changes = 0
            regime_changes = 0
            prev_regime = 0
            prev_mean = 0

            # For each date in the period
            for date in date_model_store:
                means = date_model_store[date][symbol].means_
                high_mean = means[0] if means[0] > means[1] else means[1]
                current_mean = returns_pred.loc[date, symbol]

                # Initialize prev_regime to the first period's regime once
                if prev_regime == 0:
                    prev_mean = current_mean
                    prev_regime = 1 if abs(high_mean - current_mean) < 1e-6 else -1

                # Was there a mean change?
                if abs(prev_mean - current_mean) > 1e-6:
                    mean_changes += 1
                    prev_mean = current_mean

                    current_regime = 1 if abs(high_mean - current_mean) < 1e-6 else -1
                    if current_regime != prev_regime:
                        regime_changes += 1
                        prev_regime = current_regime

            jitter.loc['mean_changes', symbol] = mean_changes
            jitter.loc['mean_change_rate', symbol] = mean_changes / len(date_model_store.keys())
            jitter.loc['regime_changes', symbol] = regime_changes
            jitter.loc['regime_change_rate', symbol] = regime_changes / len(date_model_store.keys())
            jitter.loc['periods', symbol] = len(date_model_store.keys())

            logging.debug('jitter: Symbol {} had {} mean and {} regime changes in {} periods'.format(
                symbol, mean_changes, regime_changes, len(date_model_store.keys())))

        # Compute statistics across all symbols
        jitter_all = jitter.agg(['mean', 'std'], axis=1).merge(jitter, left_index=True, right_index=True)

        # Formatting
        if print:
            cm = sns.light_palette("green", as_cmap=True)
            return jitter_all.style.background_gradient(cmap=cm).format("{:.1%}")

        return jitter_all

    def prediction_quality(self, statistic='win_rate', print=True, **kwargs):
        """
        Compute prediction quality
        :param statistic: type of statistic
        :param print: True to show plots, False for silent
        :return:
        """
        if self.__state != ModelState.PREDICTED:
            raise ValueError('Need to run predict before we can generate prediction quality statistics')

        if statistic == 'win_rate':
            realized_returns = self.get('returns', data_type='realized',
                                        sampling_freq=self.cfg['returns']['sampling_freq'])
            predicted_returns = self.get('returns', data_type='predicted',
                                         sampling_freq=self.cfg['returns']['sampling_freq'])

            return self.win_rate(predicted_returns, realized_returns, print=print, **kwargs)
        elif statistic == 'information_coefficient':
            realized_returns = self.get('returns', data_type='realized',
                                        sampling_freq=self.cfg['returns']['sampling_freq'])
            predicted_returns = self.get('returns', data_type='predicted',
                                         sampling_freq=self.cfg['returns']['sampling_freq'])

            return self.information_coef(predicted_returns, realized_returns, print=print, **kwargs)
        elif statistic == 'jitter':
            return self.jitter(print)

    def show_results(self):
        pass


if __name__ == '__main__':
    ss_hmm_model = SingleStockHMM('../examples/cvxpt_hmm.yml')
    ss_hmm_model.train(force=True)

    # Realized Data for Simulation
    prices = ss_hmm_model.get('prices', 'realized', ss_hmm_model.cfg['returns']['sampling_freq']).iloc[1:, :]
    returns = ss_hmm_model.get('returns', 'realized', ss_hmm_model.cfg['returns']['sampling_freq'])
    volumes = ss_hmm_model.get('volumes', 'realized', ss_hmm_model.cfg['returns']['sampling_freq'])
    sigmas = ss_hmm_model.get('sigmas', 'realized', ss_hmm_model.cfg['returns']['sampling_freq'])

    simulated_tcost = cp.TcostModel(half_spread=0.0005 / 2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
    simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
    simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                                   market_volumes=volumes, cash_key=ss_hmm_model.risk_free_symbol)

    ss_hmm_model.predict(mode='t', threshold=0.9, preprocess=None)
    ss_hmm_model.prediction_quality(statistic='information_coefficient', print=False)
    ss_hmm_model.prediction_quality(statistic='jitter', print=False)

    r_pred = ss_hmm_model.get('returns', 'predicted')
    conf_pred = ss_hmm_model.get('confidence', 'predicted')
    volumes_pred = ss_hmm_model.get('volumes', 'predicted')
    sigmas_pred = ss_hmm_model.get('sigmas', 'predicted')

    # Equilibrium results
    start_date = datetime.strptime(ss_hmm_model.cfg['start_date'], '%Y%m%d') + \
                 timedelta(days=ss_hmm_model.cfg['train_len']*1.75)
    end_date = datetime.strptime(ss_hmm_model.cfg['end_date'], '%Y%m%d')

    w_equal = pd.Series(index=r_pred.columns, data=[1] * len(r_pred.columns))
    w_equal.loc['USDOLLAR'] = 0.
    w_equal = w_equal / sum(w_equal)

    optimization_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1.,
                                       sigma=sigmas_pred,
                                       volume=volumes_pred)
    optimization_hcost=cp.HcostModel(borrow_costs=0.0001)

    if ss_hmm_model.cfg['covariance']['method'] == 'SS':
        spo_risk_model = cp.FullSigma(ss_hmm_model.get('covariance', 'predicted'))
    elif ss_hmm_model.cfg['covariance']['method'] == 'FF5':
        spo_risk_model = cp.FactorModelSigma(ss_hmm_model.get('exposures', 'predicted'),
                                             ss_hmm_model.get('factor_sigma', 'predicted'),
                                             ss_hmm_model.get('idyos', 'predicted'))
    else:
        raise NotImplemented('The %s risk model is not implemented yet'.format(ss_hmm_model.cfg['risk']))

    logging.basicConfig(level=logging.INFO)

    # Optimization parameters
    gamma_risk, gamma_trade, gamma_hold = 5., 15., 1.
    leverage_limit = cp.LeverageLimit(1)
    min_weight = cp.MinWeights(-0.5)
    max_weight = cp.MaxWeights(0.5)
    long_only = cp.LongOnly()

    # Optimization policy
    c_mpc_policy = cp.ModelPredictiveControlScenarioOpt(alphamodel=ss_hmm_model, horizon=5, scenarios=5,
                                                        scenario_mode='lg', costs=[gamma_risk*spo_risk_model,
                                                                                   gamma_trade*optimization_tcost,
                                                                                   gamma_hold*optimization_hcost],
                                                        constraints=[leverage_limit, min_weight, max_weight, long_only],
                                                        return_target=0.0015, mpc_method='c',
                                                        trading_freq='day')

    # Backtest
    c_mpc_results = simulator.run_multiple_backtest(1E6*w_equal,
                                                    start_time=start_date,  end_time=end_date,
                                                    policies=[c_mpc_policy],
                                                    loglevel=logging.INFO, parallel=True)
    c_mpc_results[0].summary()
