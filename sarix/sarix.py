"""
Local Trend
================
"""

import argparse
import os
import time

import pickle

from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

import jax
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

matplotlib.use('Agg')  # noqa: E402

import datetime
import covidcast


class SARProcess(Distribution):
    # arg_constraints = {"scale": constraints.positive}
    arg_constraints = {}
    support = constraints.real_vector
    # reparametrized_params = ["scale"]
    reparametrized_params = []

    def __init__(self,
                 p=1,
                 P=0,
                 season_period=7,
                 init_state=jnp.array([[[0.0]]]),
                 theta=jnp.array([1.0]),
                 noise_distribution=dist.MultivariateNormal(loc=jnp.zeros((1,)), covariance_matrix=jnp.eye(1)),
                 num_steps=1,
                 freeze_steps=0,
                 validate_args=None):
        """
        init_state: the value of the state at time 0. an array of shape (batch_shape, num_states, 1)
        A: state transition matrices, with shape (batch_shape, num_states, num_states).
            the state at time t, X_t, is computed as X_t = np.matmul(A, X_{t-1}) + epsilon_t
        noise_distribution: distribution for state transition noise. The state at time t, X_t,
            is computed as X_t = np.matmul(A, X_{t-1}) + epsilon_t, where
            epsilon_t ~ noise_distribution
        num_steps: number of steps the process iterates for.  If num_steps is 1, all values will
            be the value of the initial state
        freeze_steps: number of steps at the end of the process for which we should freeze dynamics;
            no additional noise is added to the process evolution past this point
        """
        assert (
            num_steps > 0
        ), "`num_steps` argument should be an positive integer."
        assert (
            len(theta) == 3 * (p + P * (p + 1))
        ), "`theta` argument should have length 3 * (p + P * (p + 1))"
        self.p = p
        self.P = P
        self.season_period = season_period
        self.max_lag = p + P * season_period
        self.init_state = init_state
        self.noise_distribution = noise_distribution
        self.num_states = 2
        # self.num_states = self.noise_distribution.event_shape[0]
        # self.init_state = init_state
        self.theta = theta
        self.num_steps = num_steps
        batch_shape, event_shape = (), (num_steps, self.num_states)
        n_ar_coef = (p + P * (p + 1))
        self.A = jnp.concatenate(
            [
                jnp.concatenate(
                    [theta[:n_ar_coef].reshape(n_ar_coef, 1), jnp.zeros((n_ar_coef, 1))],
                    axis = 0),
                theta[n_ar_coef:].reshape(2 * n_ar_coef, 1)
                # jnp.concatenate(
                #     [jnp.tanh(theta[:n_ar_coef].reshape(n_ar_coef, 1)), jnp.zeros((n_ar_coef, 1))],
                #     axis = 0),
                # jnp.tanh(theta[n_ar_coef:].reshape(2 * n_ar_coef, 1))
            ],
            axis = 1)
        super(SARProcess, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    @validate_sample
    def log_prob(self, value):
        # step means are np.matmul(A, states) for states before time t,
        # with shape (sample_shape, batch_shape, num_states, num_steps - 1)
        update_X = self.state_update_X(self.init_state, value)
        step_means = jnp.matmul(
            update_X,
            self.A
        )
        assert(step_means.shape == (self.num_steps, 2))
        # step innovations are (state - step_means),
        # with shape (sample_shape, batch_shape, num_states, num_steps - 1)
        # step_innovations = value[..., 1:] - step_means
        step_innovations = value - step_means
        step_probs = self.noise_distribution.log_prob(
            step_innovations[..., :self.noise_distribution.event_shape[0]])
        return jnp.sum(step_probs, axis=-1)
    
    
    def state_update_X(self, init_stoch_state, stoch_state):
        stoch_state = jnp.concatenate([init_stoch_state, stoch_state], axis=0)
        
        # lagged values of x
        lagged_x = self.build_lagged_var(stoch_state[:, 0:1])
        
        # lagged values of y
        lagged_y = self.build_lagged_var(stoch_state[:, 1:2])
        
        # concatenate
        return jnp.concatenate([lagged_x, lagged_y], axis = 1)
    
    
    def build_lagged_var(self, x):
        # lagged state, highest degree term
        lagged_state = [
            self.lagged_vals_one_seasonal_lag(x=x,
                                              seasonal_lag=P_ind*self.season_period,
                                              p=self.p) for P_ind in range(self.P+1)]
        lagged_state = [x for x in lagged_state if x is not None]
        lagged_state = jnp.concatenate(lagged_state, axis = 1)
        
        # state_update_X = jnp.concatenate(
        #     [stoch_state, lagged_state],
        #     axis=0
        # )
        # return entries in rows starting at the last row of init_stoch_shape,
        # going up to second-to-last column. These are the entries used to determine
        # means for stoch_state
        # return state_update_X[..., (init_stoch_state.shape[-1] - 1):(-1)]
        return lagged_state[(self.max_lag - 1):(-1), :]
    
    
    def lagged_vals_one_seasonal_lag(self, x, seasonal_lag, p):
        if seasonal_lag == 0:
            # no seasonal lag, just terms up to p
            to_concat = [self.lagged_col(x, l) for l in range(p)]
        else:
            # lags from seasonal_lag to (seasonal_lag + p)
            to_concat = [self.lagged_col(x, seasonal_lag - 1 + l) for l in range(p+1)]

        if to_concat == []:
            return None
        
        result = jnp.concatenate(to_concat, axis=1)
        return result
    
    
    def lagged_col(self, x, lag):
        T = x.shape[0]
        return jnp.concatenate(
            [jnp.full((lag, 1), jnp.nan), x[:(T-lag), 0:1]],
            axis=0)


    # @property
    # def mean(self):
    #     return np.zeros(self.batch_shape + self.event_shape)

    # @property
    # def variance(self):
    #     return jnp.broadcast_to(
    #         jnp.expand_dims(self.scale, -1) ** 2 * jnp.arange(1, self.num_steps + 1),
    #         self.batch_shape + self.event_shape,
    #     )

    def tree_flatten(self):
        return (self.scale,), self.num_steps

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, num_steps=aux_data)





class SARIX():
    def __init__(self,
                 xy,
                 p,
                 P,
                 season_period,
                 transform,
                 forecast_horizon,
                 num_warmup, num_samples, num_chains):
        self.xy = xy
        self.p = p
        self.P = P
        self.transform = transform
        self.season_period = season_period
        self.forecast_horizon = forecast_horizon
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains

        # do transformation
        self.xy_orig = xy.copy()
        if transform == "sqrt":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.sqrt(self.xy)
        elif transform == "fourth_rt":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.power(self.xy, 0.25)
        elif transform == "log":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.log(self.xy)

        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        self.run_inference(rng_key)

        # undo transformation to get predictions on original scale
        if transform == "log":
            self.predictions_orig = onp.exp(self.samples['xy_future'])
        elif transform == "fourth_rt":
            self.predictions_orig = jnp.maximum(0.0, self.samples['xy_future'])**4
        elif transform == "sqrt":
            self.predictions_orig = jnp.maximum(0.0, self.samples['xy_future'])**2
        else:
            self.predictions_orig = self.samples['xy_future']
        
    

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

    
    def model(self, xy):
        ## Vector of variances for each of the 2 state variables
        ar_update_sd = numpyro.sample("betas_update_var", dist.HalfCauchy(jnp.ones(2)))

        ## Lower cholesky factor of the covariance matrix
        ## we can also use a faster formula `Sigma_eta_chol = sigma[..., None] * L_sigma_eta`
        # Sigma_eta_chol = jnp.matmul(jnp.diag(sigma), L_sigma_eta)
        Sigma_ar_update_chol = jnp.diag(ar_update_sd)

        # state transition matrix parameters
        n_theta = 3 * (self.p + self.P * (self.p + 1))
        theta_sd = numpyro.sample(
            "theta_sd",
            dist.HalfCauchy(jnp.ones(1))
        )
        theta = numpyro.sample(
            "theta",
            dist.Normal(loc=jnp.zeros(n_theta), scale=jnp.full((n_theta,), theta_sd))
        )

        max_lag = self.p + self.P * 7
        betas = numpyro.sample(
            "xy",
            SARProcess(
                p=self.p,
                P=self.P,
                season_period=7,
                init_state=xy[:max_lag, :],
                theta=theta,
                noise_distribution=dist.MultivariateNormal(
                    loc=jnp.zeros((2,)), scale_tril=Sigma_ar_update_chol),
                num_steps=xy.shape[0] - max_lag,
                freeze_steps=0
            ),
            obs = xy[max_lag:, :]
        )
        
        # predict future values
        numpyro.sample(
            "xy_future",
            SARProcess(
                p=self.p,
                P=self.P,
                season_period=7,
                init_state=xy[-max_lag:, :],
                theta=theta,
                noise_distribution=dist.MultivariateNormal(
                    loc=jnp.zeros((2,)), scale_tril=Sigma_ar_update_chol),
                num_steps=self.forecast_horizon,
                freeze_steps=0
            )
        )


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

        # if n_betas >= 3:
        #     mean_curvature = onp.mean(self.curvatures, axis=0)
        #     curvature_percentiles = onp.percentile(self.curvatures, percentile_levels, axis=0)
        #     ax[3].fill_between(t_pred, curvature_percentiles[0, :], curvature_percentiles[1, :], color='lightblue')
        #     ax[3].plot(t_pred, mean_curvature, 'blue', ls='solid', lw=2.0)
        #     ax[3].set(xlabel="t", ylabel="curvature")

        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()
