import numpy as np

def normal(X, mu, sigma):
    """Return likelihood of data given parameters"

    Computes the likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar likelihood
    """
    N = X.shape[0]
    param1 = (2*np.pi*sigma**2)**(0.5*N)
    param2 = -0.5*np.sum((X-mu)**2/sigma**2)
    lik = 1/param1*np.exp(param2)

    #param1 = np.sqrt(2*np.pi*(sigma**2))
    #param2 = -0.5*(X-mu)**2/(sigma**2)
    #lik = np.prod(1/param1*np.exp(param2))

    #lik = np.prod((1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*(X-mu)**2/(sigma**2)))
    return lik