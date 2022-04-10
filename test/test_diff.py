import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as onp
import jax.numpy as jnp
import unittest

from sarix.sarix import diff, inv_diff


# reference differencing function using pandas.DataFrame.diff
def diff_df(x, d, D, season_period, dropna = True):
    df = pd.DataFrame(x)
    
    for d_val in range(d):
        df = df.diff()
    
    for D_val in range(D):
        df = df.diff(season_period)
    
    if dropna:
        df = df.dropna()
    
    return df.values



class Test_diff(unittest.TestCase):
    def test_diff_no_padding(self):
        d = 3
        D = 2
        season_period = 7
        T = 30

        # batch shape is (2, 3), T time points, 5 variables
        x = onp.random.default_rng().normal(scale=100., size = (2, 3, T, 5)).astype('float32')

        # calculate expected result manually per batch
        # we lose D*season_period + d leading time points
        expected_result = onp.full((2, 3, T - (D*season_period + d), 5), onp.nan)
        for i in range(2):
            for j in range(3):
                expected_result[i, j, ...] = diff_df(x[i, j, ...], d, D, season_period, dropna=True)

        actual_result = diff(x, d, D, season_period, pad_na=False)

        self.assertTrue(onp.allclose(actual_result, expected_result))
    
    
    def test_diff_padding(self):
        d = 3
        D = 2
        season_period = 7
        T = 30

        # batch shape is (2, 3), T time points, 5 variables
        x = onp.random.default_rng().normal(scale=100., size = (2, 3, T, 5)).astype('float32')

        # calculate expected result manually per batch
        expected_result = onp.full((2, 3, T, 5), onp.nan)
        for i in range(2):
            for j in range(3):
                expected_result[i, j, ...] = diff_df(x[i, j, ...], d, D, season_period, dropna=False)

        actual_result = diff(x, d, D, season_period, pad_na=True)

        self.assertTrue(onp.allclose(actual_result, expected_result, equal_nan = True))
    
    
    def test_inv_diff_no_broadcast_required(self):
        d = 3
        D = 2
        season_period = 7
        T = 30

        # batch shape is (2, 3), T time points, 5 variables
        x = onp.random.default_rng().normal(scale=100., size = (2, 3, T, 5)).astype('float64')

        # compute differences and then invert; expect to get original values back
        x_diff = diff(x, d, D, season_period)
        x_reconstructed = inv_diff(x[..., :(D*season_period + d), :], x_diff.astype('float64'), d, D, season_period)

        self.assertTrue(onp.allclose(x_reconstructed, x[..., (D*season_period + d):, :]))
    
    
    def test_inv_diff_broadcast_required(self):
        d = 3
        D = 2
        season_period = 7
        T = 30

        # batch shape is (4, 2, 3), T time points, 5 variables
        # first batch dimension will be kept only in dx, x will have shape (2, 3, T, 5)
        x = onp.random.default_rng().normal(scale=100., size = (4, 2, 3, T, 5)).astype('float64')
        x[:, :, :, :(D*season_period + d), :] = x[0, :, :, :(D*season_period + d), :]
        x_first_batch = x[0, :, :, :(D*season_period + d), :]
        assert(x_first_batch.shape == (2, 3, (D*season_period + d), 5))

        # compute differences and then invert using only
        # the common leading D*season_period + d time points from x
        # expect to get original values back
        x_diff = diff(x, d, D, season_period)
        x_reconstructed = inv_diff(x_first_batch, x_diff.astype('float64'), d, D, season_period)

        self.assertTrue(onp.allclose(x_reconstructed, x[..., (D*season_period + d):, :]))


