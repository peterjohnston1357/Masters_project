"""
Log-normal duration modelling utilities.

Surgical procedure durations are modelled as log-normal, following
Strum, May & Vargas (2000, Anesthesiology) - cited in the dissertation
proposal's literature review - rather than normal, since duration data
is right-skewed and bounded at zero.

A log-normal distribution is defined by two parameters, mu and sigma,
which are the mean and standard deviation of the underlying normal
distribution of log(duration). These are derived from an ordinary
mean and standard deviation using the standard transform below.
"""

import numpy as np


def lognormal_params_from_mean_sd(mean, sd):
    """
    Convert an ordinary mean and standard deviation into the mu, sigma
    parameters of the corresponding log-normal distribution.
    """
    variance = sd ** 2
    sigma_sq = np.log(1 + variance / mean ** 2)
    sigma = np.sqrt(sigma_sq)
    mu = np.log(mean) - sigma_sq / 2
    return mu, sigma


def sample_lognormal_durations(rng, mean, sd, size):
    """
    Draw `size` random durations from the log-normal distribution
    implied by the given mean and standard deviation.
    """
    mu, sigma = lognormal_params_from_mean_sd(mean, sd)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)