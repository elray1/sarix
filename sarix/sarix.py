"""
Local Trend
================
"""

import os
import time

from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.transforms import AffineTransform, LowerCholeskyAffine
from numpyro.infer import MCMC, NUTS
from numpyro.distributions.util import is_prng_key, validate_sample


def diff(x, d=0, D=0, season_period=7, pad_na=False):
    """
    Apply differencing and seasonal differencing to all variables in a
    batch of matrices with observations in rows and variables in columns.
    
    Parameters
    ----------
    x: a numpy array to difference, with shape `batch_shape + (T, n_vars)`
    d: number of ordinary differences to compute
    D: number of seasonal differences to compute
    seasonal_period: number of time points per seasonal period
    pad_na: boolean; if True, result has shape `batch_shape + (T, n_vars)` and
        the leading D + d rows in axis -2 have values `np.nan`. Otherwise, the
        result has shape `batch_shape + (T - D - d, n_vars)`.
    
    Returns
    -------
    a copy of x after taking differences
    """
    # non-seasonal differencing
    x = onp.diff(x, n = d, axis = -2)
    
    # seasonal differencing
    for i in range(D):
        x = x[..., season_period:, :] - x[..., :-season_period, :]
    
    if pad_na:
        batch_shape = x.shape[:-2]
        n_vars = x.shape[-1]
        leading_nans = onp.full(batch_shape + (D * season_period + d, n_vars), onp.nan)
        x = onp.concatenate([leading_nans, x], axis = -2)
    
    return x


def inv_diff(x, dx, d=0, D=0, season_period=7):
    '''
    Invert ordinary and seasonal differencing (go from seasonally differenced
    time series to original time series).
    
    Inputs
    ------
    dx a (batch of) first-order and/or seasonally differenced time series
        with shape `batch_shape_dx + (T_dx, n_vars)`. For example, if d=0, D=1,
        `dx` has values like `x_{t} - x_{t - season_period}`.
    x a (batch of) time series with shape `batch_shape_x + (T_x, n_vars)`.
    d order of first differencing
    D order of seasonal differencing
    seasonal_period: number of time points per seasonal period
    
    Returns
    -------
    an array with the same shape as `dx` containing reconstructed values of
    the original time series `x` in the time points `T_x, ..., T_x + T_dx - 1`
    (with zero-based indexing so that x covers the time points `0, ..., T_x - 1`)
    
    Notes
    -----
    It is assumed that dx "starts" one time index after x "ends": that is, if
        d = 0 and D = 1 then if we had observed x[..., T_x, :] we could calculate
        dx[..., 0, :] = x[..., T_x, :] - x[..., T - ts_frequency, :]
    '''
    # record information about shapes
    batch_shape_x = x.shape[:-2]
    T_x = x.shape[-2]
    n_vars = x.shape[-1]
    batch_shape_dx = dx.shape[:-2]
    T_dx = dx.shape[-2]
    
    # validate shapes
    if dx.shape[-1] != n_vars:
        raise ValueError("x and dx must have the same size in their last dimension")
    
    try:
        broadcast_batch_shape = jnp.broadcast_shapes(batch_shape_x, batch_shape_dx)
        if broadcast_batch_shape != batch_shape_dx:
            raise ValueError()
    except ValueError:
        raise ValueError("The batch shapes of x and dx must be broadcastable to the batch shape of dx")
    
    if T_x < d + D:
        raise ValueError("There must be at least d + D observed values in x to invert differencing")
    
    # invert ordinary differencing
    for i in range(1, d + 1):
        print(f'invert ordinary differencing {i}')
        x_dm1 = diff(x, d=d-i, D=D, season_period=season_period, pad_na=True)

        x_dm1 = onp.broadcast_to(x_dm1, batch_shape_dx + x_dm1.shape[-2:])
        dx_full = onp.concatenate([x_dm1, dx], axis=-2)
        for t in range(T_dx):
            dx_full[..., T_x + t, :] = dx_full[..., T_x + t - 1, :] + dx_full[..., T_x + t, :]
        
        dx = dx_full[..., -T_dx:, :]
    
    # invert seasonal differencing
    for i in range(1, D + 1):
        print(f'invert seasonal differencing {i}')
        x_dm1 = diff(x, d=0, D=D-i, season_period=season_period, pad_na=True)
        
        x_dm1 = onp.broadcast_to(x_dm1, batch_shape_dx + x_dm1.shape[-2:])
        dx_full = onp.concatenate([x_dm1, dx], axis=-2)
        for t in range(T_dx):
            dx_full[..., T_x + t, :] = dx_full[..., T_x + t - season_period, :] + dx_full[..., T_x + t, :]
        
        dx = dx_full[..., -T_dx:, :]
    
    return dx


class SARProcess(Distribution):
    # arg_constraints = {"scale": constraints.positive}
    arg_constraints = {}
    support = constraints.real_vector
    # reparametrized_params = ["scale"]
    reparametrized_params = []

    def __init__(self,
                 n_x=0,
                 p=1,
                 P=0,
                 season_period=7,
                 init_state=jnp.array([[[0.0]]]),
                 update_X=jnp.array([[1.0]]),
                 theta=jnp.array([1.0]),
                 noise_distribution=dist.MultivariateNormal(loc=jnp.zeros((1,)), covariance_matrix=jnp.eye(1)),
                 num_steps=1,
                 validate_args=None):
        """
        Parameters
        ----------
        n_x: integer number of x covariates
        p: integer number of non-seasonal lags to include
        P: integer number of seasonal lags to include
        season_period: integer length of seasonal period; e.g., 7 for weekly data
        init_state: the value of the state at time 0. an array of shape (batch_shape, n_states, 1)
        theta: parameter vector of length (1 + 2*n_x) * (p + P * (p + 1))
        noise_distribution: distribution for state transition noise, with event shape `n_x + 1`.
            The state at time t, X_t, is computed as X_t = np.matmul(A, X_{t-1}) + epsilon_t,
            where epsilon_t ~ noise_distribution
        num_steps: number of steps the process iterates for.  If num_steps is 1, all values will
            be the value of the initial state
        """
        assert (
            num_steps > 0
        ), "`num_steps` argument should be an positive integer."
        assert (
            len(theta) == (1 + 2 * n_x) * (p + P * (p + 1))
        ), "`theta` argument should have length (1 + 2 * n_x) * (p + P * (p + 1))"
        self.n_x = n_x
        self.num_states = n_x + 1
        
        self.p = p
        self.P = P
        self.season_period = season_period
        self.max_lag = p + P * season_period
        
        self.init_state = init_state
        self.update_X = update_X
        
        self.noise_distribution = noise_distribution
        
        self.theta = theta
        
        self.num_steps = num_steps
        
        batch_shape, event_shape = (), (num_steps, self.num_states)
        
        # assemble state transition matrix A
        n_ar_coef = p + P * (p + 1)
        A_x_cols = [
            jnp.concatenate(
                    [
                        jnp.zeros((i * n_ar_coef, 1)),
                        theta[(i * n_ar_coef):((i + 1) * n_ar_coef)].reshape(n_ar_coef, 1),
                        jnp.zeros(((n_x - i) * n_ar_coef, 1))
                    ],
                    axis = 0) \
                for i in range(n_x)
        ]
        A_y_col = [ theta[(n_x * n_ar_coef):].reshape((1 + n_x) * n_ar_coef, 1) ]
        self.A = jnp.concatenate(A_x_cols + A_y_col, axis = 1)
        
        super(SARProcess, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )
    
    
    @validate_sample
    def log_prob(self, value):
        # step means are np.matmul(X, A) for states before time t,
        # with shape (sample_shape, batch_shape, num_states, num_steps - 1)
        # update_X = self.state_update_X(self.init_state, value)
        update_X = self.update_X
        step_means = jnp.matmul(
            update_X,
            self.A
        )
        assert(step_means.shape == (self.num_steps, self.n_x + 1))
        # step innovations are (state - step_means),
        # with shape (sample_shape, batch_shape, num_states, num_steps - 1)
        # step_innovations = value[..., 1:] - step_means
        step_innovations = value - step_means
        step_probs = self.noise_distribution.log_prob(step_innovations)
        return jnp.sum(step_probs, axis=-1)
    
    
    def tree_flatten(self):
        return (self.scale,), self.num_steps
    
    
    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, num_steps=aux_data)



class SARIX():
    def __init__(self,
                 xy,
                 p=1,
                 d=0,
                 P=0,
                 D=0,
                 season_period=1,
                 transform='none',
                 theta_pooling='none',
                 sigma_pooling='none',
                 forecast_horizon=1,
                 num_warmup=1000, num_samples=1000, num_chains=1):
        self.n_x = xy.shape[-1] - 1
        self.xy = xy.copy()
        self.p = p
        self.d = d
        self.P = P
        self.D = D
        self.max_lag = p + P * season_period
        self.transform = transform
        self.theta_pooling = theta_pooling
        self.sigma_pooling = sigma_pooling
        self.season_period = season_period
        self.forecast_horizon = forecast_horizon
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        
        # set up batch shapes for parameter pooling
        # xy has shape batch_shape + (T, n_x + 1)
        batch_shape = xy.shape[:-2]
        
        if theta_pooling == 'none':
            # separate parameters per batch
            self.theta_batch_shape = batch_shape
        elif theta_pooling == 'shared':
            # no batches for theta; will broadcast to share across all batches
            self.theta_batch_shape = ()
        else:
            raise ValueError("theta_pooling must be 'none' or 'shared'")
        
        if sigma_pooling == 'none':
            # separate parameters per batch
            self.sigma_batch_shape = batch_shape
        elif sigma_pooling == 'shared':
            # no batches for sigma; will broadcast to share across all batches
            self.sigma_batch_shape = ()
        else:
            raise ValueError("sigma_pooling must be 'none' or 'shared'")
        
        # do transformation
        self.xy_orig = xy.copy()
        if transform == "sqrt":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.sqrt(self.xy)
        elif transform == "fourthrt":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.power(self.xy, 0.25)
        elif transform == "log":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.log(self.xy)
        
        # do differencing; save xy before differencing for later use when
        # inverting differencing
        transformed_xy = self.xy
        self.xy = diff(self.xy, self.d, self.D, self.season_period, pad_na=False)
        
        # pre-calculate state update matrix
        self.update_X = self.state_update_X(self.xy[..., :self.max_lag, :],
                                            self.xy[..., self.max_lag:, :])
        
        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        self.run_inference(rng_key)
        
        # generate predictions
        self.predictions_modeled_scale = self.predict(rng_key_predict)
        
        # undo differencing
        self.predictions = inv_diff(transformed_xy,
                                    self.predictions_modeled_scale,
                                    self.d, self.D, self.season_period)
        
        # undo transformation to get predictions on original scale
        if transform == "log":
            self.predictions = onp.exp(self.predictions)
        elif transform == "fourthrt":
            self.predictions = jnp.maximum(0.0, self.predictions)**4
        elif transform == "sqrt":
            self.predictions = jnp.maximum(0.0, self.predictions)**2
    
    
    def run_inference(self, rng_key):
        '''
        helper function for doing hmc inference
        '''
        start = time.time()
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains,
                    progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
        mcmc.run(rng_key, self.xy, init_params={})
        mcmc.print_summary()
        print('\nMCMC elapsed time:', time.time() - start)
        self.samples = mcmc.get_samples()
    
    
    def make_state_transition_matrix(self, theta):
        batch_shape = theta.shape[:-1]
        n_ar_coef = self.p + self.P * (self.p + 1)
        
        A_x_cols = [
            jnp.concatenate(
                    [
                        jnp.zeros(batch_shape + (i * n_ar_coef, 1)),
                        jnp.expand_dims(theta[..., (i * n_ar_coef):((i + 1) * n_ar_coef)], -1),
                        jnp.zeros(batch_shape + ((self.n_x - i) * n_ar_coef, 1))
                    ],
                    axis = -2) \
                for i in range(self.n_x)
        ]
        A_y_col = [ jnp.expand_dims(theta[..., (self.n_x * n_ar_coef):], -1) ]
        
        A = jnp.concatenate(A_x_cols + A_y_col, axis = -1)
        
        return A
    
    
    def state_update_X(self, init_stoch_state, stoch_state):
        stoch_state = jnp.concatenate([init_stoch_state, stoch_state], axis=-2)
        
        # lagged values of x
        lagged_x = [
            self.build_lagged_var(stoch_state[..., :, i:(i+1)]) \
                for i in range(self.n_x)
        ]
        
        # lagged values of y
        lagged_y = self.build_lagged_var(stoch_state[..., :, self.n_x:(self.n_x + 1)])
        
        # concatenate
        return jnp.concatenate(lagged_x + [lagged_y], axis = -1)
    
    
    def build_lagged_var(self, x):
        # lagged state, highest degree term
        lagged_state = [
            self.lagged_vals_one_seasonal_lag(x=x,
                                              seasonal_lag=P_ind*self.season_period,
                                              p=self.p) for P_ind in range(self.P+1)]
        lagged_state = [x for x in lagged_state if x is not None]
        lagged_state = jnp.concatenate(lagged_state, axis = -1)
        
        # return entries in rows starting at the last row of init_stoch_shape,
        # going up to second-to-last column. These are the entries used to determine
        # means for stoch_state
        return lagged_state[..., (self.max_lag - 1):(-1), :]
    
    
    def lagged_vals_one_seasonal_lag(self, x, seasonal_lag, p):
        if seasonal_lag == 0:
            # no seasonal lag, just terms up to p
            to_concat = [self.lagged_col(x, l) for l in range(p)]
        else:
            # lags from seasonal_lag to (seasonal_lag + p)
            to_concat = [self.lagged_col(x, seasonal_lag - 1 + l) for l in range(p+1)]

        if to_concat == []:
            return None
        
        result = jnp.concatenate(to_concat, axis=-1)
        return result
    
    
    def lagged_col(self, x, lag):
        batch_shape = x.shape[:-2]
        T = x.shape[-2]
        return jnp.concatenate(
            [jnp.full(batch_shape + (lag, 1), jnp.nan), x[..., :(T-lag), 0:1]],
            axis=-2)
    
    
    def model(self, xy):
        # Vector of innovation standard deviations for the n_x + 1 variables
        sigma = numpyro.sample(
            "sigma",
            dist.HalfCauchy(jnp.ones(self.sigma_batch_shape + (self.n_x + 1,))))
        
        # Lower cholesky factor of the covariance matrix has
        # standard deviations on the diagonal
        # The first line below creates (potentially batched) diagonal matrices
        # with shape self.sigma_batch_shape + (n_x + 1, n_x + 1)
        # If xy has batch dimensions, we then insert another dimension at
        # postion -3 for appropriate broadcasting with the time dimension of
        # observed values
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])
        if len(xy.shape) > 2:
            Sigma_chol = jnp.expand_dims(Sigma_chol, -3)
        
        # state transition matrix parameters
        n_theta = (2 * self.n_x + 1) * (self.p + self.P * (self.p + 1))
        theta_sd = numpyro.sample(
            "theta_sd",
            dist.HalfCauchy(jnp.ones(1))
        )
        theta = numpyro.sample(
            "theta",
            dist.Normal(loc=jnp.zeros(self.theta_batch_shape + (n_theta,)),
                        scale=jnp.full(self.theta_batch_shape + (n_theta,),
                                       theta_sd))
        )
        
        # assemble state transition matrix A
        A = self.make_state_transition_matrix(theta)
        
        # predictive means based on AR structure
        step_means = jnp.matmul(
            self.update_X,
            A
        )
        
        # step innovations are (state - step_means),
        # with shape (batch_shape, T - self.max_lag, n_x + 1)
        step_innovations = xy[..., self.max_lag:, :] - step_means
        
        # sample innovations
        numpyro.sample(
            "step_innovations",
            dist.MultivariateNormal(
                loc=jnp.zeros((self.n_x + 1,)), scale_tril=Sigma_chol),
            obs=step_innovations
        )
    
    
    def predict(self, rng_key):
        '''
        Predict future values of all signals based on a single sample of
        parameter values from the posterior distribution.
        '''
        # load in parameter estimates and update to target batch size
        theta = self.samples['theta']
        sigma = self.samples['sigma']
        xy_batch_shape = self.xy.shape[:-2]
        theta_batch_shape = theta.shape[:-1]
        sigma_batch_shape = sigma.shape[:-1]
        
        if self.theta_pooling == 'shared':
            # goal is shape theta_batch_shape + xy_batch_shape + theta.shape[-1]
            # first insert 1's corresponding to xy_batch_shape, then broadcast
            ones = (1,) * len(xy_batch_shape)
            theta = theta.reshape(theta_batch_shape + ones + (theta.shape[-1],))
            theta = jnp.broadcast_to(theta,
                                     theta_batch_shape + xy_batch_shape + \
                                         (theta.shape[-1],))
        
        if self.sigma_pooling == 'shared':
            # goal is shape sigma_batch_shape + xy_batch_shape + sigma.shape[-1]
            # first insert 1's corresponding to xy_batch_shape, then broadcast
            ones = (1,) * len(xy_batch_shape)
            sigma = theta.reshape(sigma_batch_shape + ones + (sigma.shape[-1],))
            sigma = jnp.broadcast_to(sigma,
                                     sigma_batch_shape + xy_batch_shape + \
                                         (sigma.shape[-1],))
        
        batch_shape = theta.shape[:-1]
        
        # state transition matrix
        A = self.make_state_transition_matrix(theta)
        
        # convert sigma to a batch of covariance matrix Cholesky factors
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])
        
        # generate innovations
        # note that the use of sample_shape = forecast_horizon means that
        # innovations.shape = (forecast_horizon,) + batch_shape + (n_x + 1)
        # we would really like shape batch_shape + (forecast_horizon, n_x + 1)
        # we deal with this in the loop below when adding to the mean for each
        # forecast horizon by dropping the leading dimension when indexing, then
        # inserting an extra dimension at position -2 before adding.
        innovations = dist.MultivariateNormal(
                loc=jnp.zeros((self.n_x + 1,)),
                scale_tril=Sigma_chol) \
            .sample(rng_key, sample_shape=(self.forecast_horizon, ))
        
        # generate step-ahead forecasts iteratively
        y_pred = []
        recent_lags = jnp.broadcast_to(self.xy[..., -self.max_lag:, :],
                                       batch_shape + (self.max_lag, self.xy.shape[-1]))
        dummy_values = jnp.zeros(batch_shape + (1, self.xy.shape[-1]))
        for h in range(self.forecast_horizon):
            update_X = self.state_update_X(recent_lags, dummy_values)
            new_y_pred = jnp.matmul(update_X, A) + \
                jnp.expand_dims(innovations[h, ...], -2)
            y_pred.append(new_y_pred)
            recent_lags = jnp.concatenate([recent_lags[..., 1:, :], new_y_pred],
                                          axis=-2)
        
        y_pred = jnp.concatenate(y_pred, axis=-2)
        return onp.asarray(y_pred)
    
    
    def plot(self, save_path = None):
        t = onp.arange(self.y_nbhd.shape[0])
        t_pred = onp.arange(self.y_nbhd.shape[0] + self.forecast_horizon)
        n_betas = self.samples['betas'].shape[1]
        
        percentile_levels = [2.5, 97.5]
        median_prediction = onp.median(self.predictions, axis=0)
        percentiles = onp.percentile(self.predictions, percentile_levels, axis=0)
        median_prediction_orig = onp.median(self.predictions_orig, axis=0)
        percentiles_orig = onp.percentile(self.predictions_orig, percentile_levels, axis=0)
        
        fig, ax = plt.subplots(n_betas + 1, 1, figsize=(10,3 * (n_betas + 1)))
        
        ax[0].fill_between(t_pred, percentiles_orig[0, :], percentiles_orig[1, :], color='lightblue')
        ax[0].plot(t_pred, median_prediction_orig, 'blue', ls='solid', lw=2.0)
        ax[0].plot(t, self.y_orig, 'black', ls='solid')
        ax[0].set(xlabel="t", ylabel="y", title="Mean predictions with 95% CI")
        
        # plot 95% confidence level of predictions
        ax[1].fill_between(t_pred, percentiles[0, :], percentiles[1, :], color='lightblue')
        ax[1].plot(t_pred, median_prediction, 'blue', ls='solid', lw=2.0)
        ax[1].plot(t, self.y, 'black', ls='solid')
        ax[1].set(xlabel="t", ylabel="y (" + self.transform + " scale)", title="Mean predictions with 95% CI")
        
        for i in range(1, n_betas):
            beta_median = onp.median(self.samples['betas'][:, i, :], axis=0)
            beta_percentiles = onp.percentile(self.samples['betas'][:, i, :], percentile_levels, axis=0)
            ax[i + 1].fill_between(t_pred, beta_percentiles[0, :], beta_percentiles[1, :], color='lightblue')
            ax[i + 1].plot(t_pred, beta_median, 'blue', ls='solid', lw=2.0)
            ax[i + 1].set(xlabel="t", ylabel="incidence deriv " + str(i))
        
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()
