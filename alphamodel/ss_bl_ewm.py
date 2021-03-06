"""
Single stock returns - Black Litterman model with exponentially weighted moving average generated return/variance views
"""

import cvxportfolio as cp
import logging
import math
import numpy as np
import pandas as pd
import scipy.stats as sts

from .model import ModelState
from .scenario import Scenario
from .ss_ewm import SingleStockEWM
from .utils import is_pd, nearest_pd
from datetime import timedelta, datetime

__all__ = ['SingleStockBLEWM']


class SingleStockBLEWM(SingleStockEWM):

    def predict(self, w_market_cap_init=None, risk_aversion=2,
                P_view=np.array([]), Q_view=np.array([]),
                Omega_view=np.array([]), view_confidence=0.75,
                noise_mode='dynamic_sigmoid',
                c=0.75, **kwargs):
        """
        Prediction function for model, for out of sample historical test set

        :param w_market_cap_init: market cap weights at beginning of training period
        :param risk_aversion: delta (δ) - risk aversion parameter (scalar)
        :param P_view: KxN matrix for views (P * miu = Q + Epsilon)
        :param Q_view: K vector of view constants
        :param Omega_view: K vector of view noise
        :param view_confidence: confidence in view, will set uncertainty [0, 1)
        :param noise_mode:
                pass_through: direct pass of Omega_view as Omega
                confidence_static: Walters - alpha * P * Sigma * P.T
                confidence_dynamic: CDF of Q_view within investor predicted view distribution
        :param c: certainty weight (in investor views) (scalar)
                0: complete certainty, use only investor views
                1: complete uncertainty, ignore investor views
        :return: n/a (all data stored in self.predicted)
        """
        # Predict the underlying HMM returns/variances
        if self.state not in [ModelState.TRAINED, ModelState.PREDICTED]:
            raise ValueError('SingleStockBLHMM.predict: Can\'t predict if we haven\'t trained the model at least once.')
        super().predict(**kwargs)

        # We are not yet done predicting so reset the state for now
        self.__state = ModelState.TRAINED

        # Compute the market cap and equilibrium returns starting with the time 0 market cap weights
        # 1. Roll forward the market cap weights according to the realized returns for each asset
        if type(w_market_cap_init) not in [np.array, list, pd.Series]:
            return ValueError('SingleStockEWM.predict: w_market_cap needs to be an array or a pandas Series.')
        r_realized = self.get('returns', 'realized').shift(1)
        r_realized.iloc[:, :] = r_realized.iloc[:, :] + 1
        raw_weights = np.multiply(r_realized.cumprod().values, w_market_cap_init.values.T)
        w_market_cap = pd.DataFrame(data=(raw_weights / np.sum(raw_weights, axis=1)[:, None]), index=r_realized.index)

        # 2. Retrieve the covariance at each time step
        sigma = pd.DataFrame()
        if self.cfg['covariance']['method'] == 'SS':
            # For single stock it's computed directly
            sigma = self.get('covariance', 'predicted')
        elif self.cfg['covariance']['method'] == 'FF5':
            # For factor model we need to reconstruct it
            factor_sigma = self.get('factor_sigma', 'predicted')
            exposures = self.get('exposures', 'predicted')
            idyos = self.get('idyos', 'predicted')

            sigma = pd.DataFrame()
            for t in idyos.index:
                t_cov = cp.utils.time_locator(exposures, t, True).dot(cp.utils.time_locator(factor_sigma, t, True)).dot(
                    cp.utils.time_locator(exposures, t, True).T) + np.diag(cp.utils.time_locator(idyos, t, True))
                sigma = pd.concat([sigma, pd.DataFrame(index=pd.MultiIndex.from_product(([t], r_realized.columns)),
                                                       data=t_cov, columns=r_realized.columns)])

        # 3. Compute the equilibrium returns from the covariance and market cap weights
        r_pred = self.get('returns', 'predicted')
        r_equilibrium = pd.DataFrame()
        t_sigma = np.array([])
        for t in w_market_cap.index:
            try:
                t_sigma = cp.utils.time_locator(sigma, t, True)
            except KeyError as e:
                logging.debug('SingleStockBLEWM.predict: Unable to find a new covariance for {}, using current: {}'.
                              format(str(t), str(e)))
            if t_sigma.size > 0:
                t_r_eq = risk_aversion * t_sigma.dot(cp.utils.time_locator(w_market_cap, t, True))
                r_equilibrium = pd.concat([r_equilibrium, pd.DataFrame(index=[t], data=t_r_eq[None, :],
                                                                       columns=r_pred.columns)])

        # Compute the BL posterior returns & covariance once the HMM views are incorporated
        r_expected, sigma_expected, conf_expected = self.black_litterman_posterior_r_sigma(P_view, Q_view,
                                                                                           r_equilibrium, r_pred,
                                                                                           c, sigma,
                                                                                           O_view=Omega_view,
                                                                                           view_confidence=view_confidence,
                                                                                           noise_mode=noise_mode)

        # Save down the returns & covariances
        # Save the old into _raw then overwrite the old
        self.set('returns_ewm', r_pred, 'predicted')
        self.set('covariance_ewm', sigma, 'predicted')
        self.set('w_market_cap', w_market_cap, 'realized')
        self.set('r_equilibrium', r_equilibrium, 'predicted')

        self.set('returns', r_expected, 'predicted')
        self.set('covariance', sigma_expected, 'predicted')
        self.set('confidence', conf_expected, 'predicted')

        # We're now done so can call ourselves predicted
        self.__state = ModelState.PREDICTED

        return True

    def generate_forward_scenario(self, dt, horizon, mode='g', return_src='pred', **kwargs):
        """
        Generate forward scenario
        :param dt: datetime to start at
        :param horizon: periods ahead to be included in the scenario
        :param mode:    c = constant scenario
                        g = Gaussian sampled scenario (Gaussian of expected mean & sigma for values)
                        # hmm = HMM sampled scenario (fit parameters to historical training data)
        :param return_src:  bl = Black Litterman
                            pred = underlying quant model
        :return: Scenario() instance
        """
        if self.state != ModelState.PREDICTED:
            raise ValueError('generate_forward_scenario: Unable to run if model is not in predicted state.')

        # Grab needed inputs
        if return_src == 'bl':
            returns_mean = self.get('returns', 'predicted')
            return_sigmas_mean = self.get('covariance', 'predicted')
        elif return_src == 'pred':
            returns_mean = self.get('returns_ewm', 'predicted')
            return_sigmas_mean = self.get('covariance_ewm', 'predicted')
        else:
            raise NotImplemented('generate_forward_scenario: Only bl and pred return_src implemented.')
        volumes_mean = self.get('volumes', 'predicted')
        sigmas_mean = self.get('sigmas', 'predicted')

        # Generate indices (dates)
        dt_index = volumes_mean.index.get_loc(dt)
        avail_dates = volumes_mean.shape[0] - dt_index
        if avail_dates >= horizon:
            indices = volumes_mean.index[dt_index:(dt_index + horizon)]
        else:
            real_dates = volumes_mean.index[dt_index:(dt_index + avail_dates)]
            potential_dates = pd.date_range(start=real_dates[-1],
                                            end=real_dates[-1] + timedelta(days=horizon - avail_dates))
            indices = real_dates.union(potential_dates)

        # Generate samples based on configuration
        returns = pd.DataFrame(index=indices)
        volumes = pd.DataFrame(index=indices)
        sigmas = pd.DataFrame(index=indices)

        for symbol in self._universe:
            if mode == 'hmm':
                raise NotImplemented('generate_forward_scenario: HMM mode not implemented.')
                # HMM model generated samples
                # returns.loc[:, symbol], _ = hmm_dt[symbol].sample(horizon)

            elif mode == 'g':
                # Gaussian generated samples

                # Generate returns - expected return & sigma
                sym_return_mean = returns_mean.loc[dt, symbol]
                sym_return_sigma = np.sqrt(return_sigmas_mean.loc[(dt, symbol), symbol])

                # Gaussian distribution of horizon length
                rng = np.random.default_rng()
                samples = rng.normal(sym_return_mean, sym_return_sigma, horizon)
                returns.loc[:, symbol] = samples

                if symbol != self.risk_free_symbol:
                    # Generate volumes
                    sym_volume_mean = volumes_mean.loc[dt, symbol]
                    sym_volume_sigma = volumes_mean.loc[:dt, symbol].std()

                    # Gaussian distribution of horizon length
                    rng = np.random.default_rng()
                    samples = rng.normal(sym_volume_mean, sym_volume_sigma, horizon)
                    volumes.loc[:, symbol] = samples

                    # Generate sigmas
                    sym_sigma_mean = sigmas_mean.loc[dt, symbol]
                    sym_sigma_sigma = sigmas_mean.loc[:dt, symbol].std()

                    # Gaussian distribution of horizon length
                    rng = np.random.default_rng()
                    samples = rng.normal(sym_sigma_mean, sym_sigma_sigma, horizon)
                    sigmas.loc[:, symbol] = samples

            elif mode == 'c':
                # Constant expected return
                sym_return = returns_mean.loc[dt, symbol]
                returns.loc[:, symbol] = sym_return

                if symbol != self.risk_free_symbol:
                    # Constant expected volume
                    sym_volume = volumes_mean.loc[dt, symbol]
                    volumes.loc[:, symbol] = sym_volume

                    # Constant expected sigma
                    sym_sigma = sigmas_mean.loc[dt, symbol]
                    sigmas.loc[:, symbol] = sym_sigma

        # Create a scenario from the inputs
        return Scenario(dt, horizon, returns, volumes, sigmas)

    @staticmethod
    def black_litterman_posterior_r_sigma(P_view, Q_view, r_eq, r_predicted, c, Sigma,
                                          noise_mode='dynamic_sigmoid',
                                          eta=1,
                                          view_confidence=0.75,
                                          O_view=np.array([])):
        """
        Incorporates view return into equilibrium returns for the Black Litterman model

        Quick example: https://thefjg.blogspot.com/2017/12/black-litterman-portfolio-optimization.html

        :param P_view: KxN matrix for views (P * miu = Q + Epsilon)
        :param Q_view: K vector of view constants
        :param r_eq: equilibrium returns (priors)
        :param r_predicted: predicted direct asset returns
        :param O_view: noise covariance matrix
        :param view_confidence: confidence in view, will set uncertainty [0, 1)
        :param noise_mode:
                pass_through: direct pass of Omega_view as Omega
                static: Walters - alpha * P * Sigma * P.T
                dynamic_cdf: CDF of Q_view within investor predicted view distribution
                dynamic_sigmoid: sigmoid of Q_view within investor predicted view distribution
        :param eta: sigmoid slope parameter
        :param view_confidence: noise in predicted returns (investor views)
        :param noise_mode:
                pass_through: direct
                diag: diagonalize
        :param c: certainty weight (in investor views) (scalar)
               1: complete certainty, use only investor views
               0: complete uncertainty, ignore investor views
        :param Sigma: covariance
        :return:
        """
        r_posterior = {}
        sigma_posterior = {}
        confidence_posterior = {}

        first_index = r_predicted.index[0]
        first_found = False
        t_r_eq = np.array([])
        t_r_pred = np.array([])
        t_sigma = np.array([])

        tau = (1 / (1 - c)) - 1
        P = P_view
        Q = Q_view

        while not first_found:
            try:
                t_r_eq = cp.utils.time_locator(r_eq, first_index, True)
                t_r_pred = cp.utils.time_locator(r_predicted, first_index, True)
                t_sigma = cp.utils.time_locator(Sigma, first_index, True)
                first_found = True
            except KeyError as e:
                logging.debug('black_litterman_posterior_r_sigma: First index {} didn\'t work, going back 1 more day'.
                              format(str(first_index)))
                first_index = first_index - timedelta(days=1)

        # Generate posterior returns and sigma based on predictions and confidences
        for index, row in r_predicted.iterrows():
            # Gather variables
            try:
                t_r_eq = cp.utils.time_locator(r_eq, index, True)
                t_r_pred = cp.utils.time_locator(r_predicted, index, True)
                t_sigma = cp.utils.time_locator(Sigma, index, True)
            except KeyError as e:
                logging.debug('black_litterman_posterior_r_sigma: Missing a value for index {}, keeping current: {}'.
                              format(str(index), str(e)))

            if noise_mode == 'pass_through':
                Omega = O_view
                confidence_posterior[index] = 1
            elif noise_mode == 'static':
                # Confidence can't be 0, set it to 1e-6
                if view_confidence == 0:
                    view_confidence = 1e-6
                alpha = (1 - view_confidence) / view_confidence
                Omega = alpha * np.dot(np.dot(P, t_sigma), P.T)
                confidence_posterior[index] = view_confidence
            elif noise_mode == 'dynamic_cdf':
                # Validate input
                if len(P.shape) > 1:
                    rows, cols = P.shape
                    if rows > 1 and cols == 1:
                        raise NotImplementedError('dynamic_cdf: support for multiple views not available.')

                # Compute predicted return distribution for view given investor returns
                predicted_view_mean = np.dot(P, t_r_pred)
                predicted_view_var = np.dot(np.dot(P, t_sigma), P.T)

                # Compute CDF and PDF of view return
                if len(Q.shape) == 0:
                    view_return = float(Q)
                else:
                    view_return = Q_view[0]
                view_z_score = (view_return - predicted_view_mean) / predicted_view_var
                view_cdf = sts.norm.cdf(view_z_score)
                view_pdf = sts.norm.pdf(view_z_score)

                # Use confidence according to view return sign
                if view_return > 0:
                    view_confidence = 1 - view_cdf
                elif view_return < 0:
                    view_confidence = view_cdf
                else:
                    view_confidence = view_pdf

                # Construct Omega according to dynamic view confidence
                alpha = (1 - view_confidence) / view_confidence
                Omega = alpha * np.dot(np.dot(P, t_sigma), P.T)

                confidence_posterior[index] = view_confidence
            elif noise_mode == 'dynamic_sigmoid':
                # Validate input
                if len(P.shape) > 1:
                    rows, cols = P.shape
                    if rows > 1 and cols == 1:
                        raise NotImplementedError('dynamic_sigmoid: support for multiple views not available.')

                # Compute predicted return distribution for view given investor returns
                predicted_view_mean = np.dot(P, t_r_pred)

                # Compute CDF and PDF of view return
                if len(Q.shape) == 0:
                    view_return = float(Q)
                else:
                    view_return = Q_view[0]

                # Use confidence according to view return sign
                scalar = 10**-math.floor(math.log(view_return, 10))
                if view_return >= 0:
                    view_confidence = 1 / (1 + np.exp(-eta*scalar*predicted_view_mean +
                                                      eta*scalar*view_return))
                else:
                    view_confidence = 1 / (1 + np.exp(eta*scalar*predicted_view_mean +
                                                      eta*scalar*view_return))

                # Construct Omega according to dynamic view confidence
                alpha = (1 - view_confidence) / view_confidence
                Omega = alpha * np.dot(np.dot(P, t_sigma), P.T)
                confidence_posterior[index] = view_confidence
            else:
                raise NotImplementedError('black_litterman_posterior_r_sigma: support only for `pass_through` and '
                                          '`confidence` modes.')

            # Pre-compute some values
            view_var_term = tau * np.dot(np.dot(P, t_sigma), P.T) + Omega
            if view_var_term.size == 1:
                inverse_view_var_term = 1 / view_var_term
            else:
                inverse_view_var_term = np.linalg.inv(tau * np.dot(np.dot(P, t_sigma), P.T) + Omega)

            # Posterior return & covariance calculations
            r_posterior[index] = t_r_eq + np.dot(
                    np.dot(tau * np.dot(t_sigma, P.T),
                           inverse_view_var_term),
                    (Q - np.dot(P, t_r_eq))
                )
            sigma_posterior_candidate = t_sigma + tau * t_sigma - tau * np.dot(
                    np.dot(np.dot(t_sigma, P.T),
                           inverse_view_var_term),
                    tau * np.dot(P, t_sigma)
                )
            sigma_posterior[index] = sigma_posterior_candidate if is_pd(sigma_posterior_candidate) else nearest_pd(
                sigma_posterior_candidate)

            logging.debug('black_litterman_posterior_r_sigma: Completed {d} run.'.format(d=str(index)))

        # Convert posterior dict to DataFrame
        r_posterior_df = pd.DataFrame.from_dict(r_posterior, orient='index')
        r_posterior_df.columns = r_predicted.columns

        sigma_posterior_df = pd.concat([pd.DataFrame(v) for v in sigma_posterior.values()])
        sigma_posterior_df.index = pd.MultiIndex.from_product([sigma_posterior.keys(), r_predicted.columns])
        sigma_posterior_df.columns = r_predicted.columns

        confidence_posterior_df = pd.DataFrame.from_dict(confidence_posterior, orient='index')

        return r_posterior_df, sigma_posterior_df, confidence_posterior_df


if __name__ == '__main__':
    # Initialize model
    bl_ewm_model = SingleStockBLEWM('../examples/cvxpt_hmm.yml')

    # Training
    logging.basicConfig(level=logging.INFO)
    logging.warning('Fetching training data...')
    bl_ewm_model.train(force=False)

    # Realized Data for Simulation
    prices = bl_ewm_model.get('prices', 'realized', bl_ewm_model.cfg['returns']['sampling_freq']).iloc[1:, :]
    returns = bl_ewm_model.get('returns', 'realized', bl_ewm_model.cfg['returns']['sampling_freq'])
    volumes = bl_ewm_model.get('volumes', 'realized', bl_ewm_model.cfg['returns']['sampling_freq'])
    sigmas = bl_ewm_model.get('sigmas', 'realized', bl_ewm_model.cfg['returns']['sampling_freq'])

    simulated_tcost = cp.TcostModel(half_spread=0.0005 / 2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
    simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
    simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                                   market_volumes=volumes, cash_key=bl_ewm_model.risk_free_symbol)

    # Prediction
    logging.warning('Running return prediction...')
    bl_ewm_model.predict(w_market_cap_init=pd.Series(index=['SPY', 'EWJ', 'EWG', 'USDOLLAR'],
                                                     data=[0.65, 0.2, 0.15, 0]),
                         P_view=np.array([1, 0, -1, 0]), Q_view=np.array(0.05/252),
                         noise_mode='dynamic_sigmoid')
    bl_ewm_model.prediction_quality()

    r_pred = bl_ewm_model.get('returns', 'predicted')
    volumes_pred = bl_ewm_model.get('volumes', 'predicted')
    sigmas_pred = bl_ewm_model.get('sigmas', 'predicted')

    bl_ewm_model.generate_forward_scenario(r_pred.index[100], 5)

    # Equilibrium results
    start_date = datetime.strptime(bl_ewm_model.cfg['start_date'], '%Y%m%d') + \
                 timedelta(days=bl_ewm_model.cfg['train_len'] * 1.75)
    end_date = datetime.strptime(bl_ewm_model.cfg['end_date'], '%Y%m%d')

    w_equal = pd.Series(index=r_pred.columns, data=[1] * len(r_pred.columns))
    w_equal.loc['USDOLLAR'] = 0.
    w_equal = w_equal / sum(w_equal)

    optimization_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1.,
                                       sigma=sigmas_pred,
                                       volume=volumes_pred)
    optimization_hcost=cp.HcostModel(borrow_costs=0.0001)

    if bl_ewm_model.cfg['covariance']['method'] == 'SS':
        spo_risk_model = cp.FullSigma(bl_ewm_model.get('covariance', 'predicted'))
    elif bl_ewm_model.cfg['covariance']['method'] == 'FF5':
        spo_risk_model = cp.FactorModelSigma(bl_ewm_model.get('exposures', 'predicted'),
                                             bl_ewm_model.get('factor_sigma', 'predicted'),
                                             bl_ewm_model.get('idyos', 'predicted'))
    else:
        raise NotImplemented('The %s risk model is not implemented yet'.format(bl_ewm_model.cfg['risk']))

    logging.warning('Running simulation...')
    # Optimization parameters
    gamma_risk, gamma_trade, gamma_hold = 5., 15., 1.
    leverage_limit = cp.LeverageLimit(1)
    fully_invested = cp.ZeroCash()
    long_only = cp.LongOnly()

    # Optimization policy
    c_mps_policy = cp.MultiPeriodScenarioOpt(alphamodel=bl_ewm_model, horizon=5, scenarios=5,
                                             scenario_mode='c', costs=[gamma_risk*spo_risk_model,
                                                                       gamma_trade*optimization_tcost,
                                                                       gamma_hold*optimization_hcost],
                                             constraints=[leverage_limit, fully_invested, long_only],
                                             trading_freq='once')

    # Backtest
    c_mpc_results = simulator.run_multiple_backtest(1E6 * w_equal,
                                                    start_time=start_date, end_time=end_date,
                                                    policies=[c_mps_policy],
                                                    loglevel=logging.INFO, parallel=True)
    c_mpc_results[0].summary()
