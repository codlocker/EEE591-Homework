import scipy.integrate as spi
import numpy as np

##############################################################################
# This program computes the integral of a function using using integration by#
# substitution. Here we replace x with tan^2 (theta) as per the question.    #
# The limits are also changed from 0 to infinity to inverse tan values of it.#
##############################################################################
if __name__ == "__main__":
    # Refer to attached image file for the substitution that forms this integrand 
    integrand = lambda z : 1 / np.sqrt(z * (1 - z))

    # Integration limits
    lower_limit = 0
    upper_limit = 1

    # Perform integration using the declared integrands and limits.
    y, err = spi.quad(
        func=integrand,
        a=lower_limit,
        b=upper_limit,
    )

    print("Pi is {}".format(np.round(y, 8)))
    print("Difference from numpy.pi is: {}".format(
        np.round(
            np.subtract(y, np.pi),
            15)))