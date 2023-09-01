import scipy.integrate as spi
import numpy as np

##############################################################################
# This program computes the integral of a function using using integration by#
# substitution. Here we replace x with tan^2 (theta) as per the question.    #
# The limits are also changed from 0 to infinity to inverse tan values of it.#
##############################################################################
if __name__ == "__main__":
    integrand = lambda theta : (2 * np.tan(theta) * np.power(1 / np.cos(theta), 2)) / ((1 + np.power(np.tan(theta), 2)) * np.tan(theta))

    lower_limit = np.arctan(0)
    upper_limit = np.arctan(np.sqrt(np.inf))


    y, err = spi.quad(
        func=integrand,
        a=lower_limit,
        b=upper_limit,
    )

    print("Pi is {}".format(np.round(y, 8)))
    # print(y, np.pi)
    print("Difference from numpy.pi is: {}".format(np.subtract(y, np.pi)))