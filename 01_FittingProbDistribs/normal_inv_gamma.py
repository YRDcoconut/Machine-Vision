import numpy as np
from scipy.special import gamma as gamma_function

def normal_inv_gamma(alpha, beta, delta, gamma, mu, sigma):
    """Return the probability density function for the normal
    inverse gamma density at (mu, sigma)
    
    Args:
        alpha: shape of variance
        beta: scale of variance
        delta: mean of mu
        gamma: precision of mu
        mu: normal mean
        sigma: normal standard deviation
    Returns:
        a probability density function
    """
    # You will find scipy.special.gamma useful
    param1 = np.sqrt(gamma) / (sigma*np.sqrt(2*np.pi))
    param2 = (beta**alpha)/gamma_function(alpha)
    param3 = (1/(sigma**2))**(alpha+1)
    param4 = -(2*beta+gamma*(delta-mu)**2)/(2*(sigma**2))
    lik = param1*param2*param3*np.exp(param4)



    return lik
