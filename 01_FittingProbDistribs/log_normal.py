import numpy as np

def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    N = X.shape[0]
    loglik = -0.5*N*np.log(2*np.pi)-0.5*N*np.log(sigma**2)-0.5*np.sum((X-mu)**2/sigma**2)
    return loglik
 