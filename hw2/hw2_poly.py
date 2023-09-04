import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np

###########################################################################
# Integration of a quadriatic function with 2 different coefficients. The #
# results are then plotted using matplotlib. So we see 2 curves in the    #
# result.                                                                 #
###########################################################################
if __name__ == "__main__":
    R = 5
    STEPS = 0.01
    no_points = int(R / STEPS)
    values = np.linspace(0, R, no_points)

    # Integrate quadriatic equation with coefficients a = 2, b = 3 and c = 4
    y_1 = [spi.quad(
    	func=lambda x, a, b, c : a * x**2 + b * x + c,
        a=0,
        b=value,
        args=(2, 3, 4))[0] for value in values]

	# Integrate quadriatic equation with coefficients a = 2, b = 1 and c = 1
    y_2 = [spi.quad(
        func=lambda x, a, b, c : a * x**2 + b * x + c,
        a=0,
        b=value,
        args=(2, 1, 1))[0] for value in values]

	# Plot the result
    plt.plot(values, y_1, color='blue', label='(a, b, c) = (2, 3, 4)')
    plt.plot(values, y_2, color='red', label='(a, b, c) = (2, 1, 1)')
    
	# The next steps are formatting the graph.
    leg = plt.legend(
        loc='upper center')
    plt.title('Integration of ax^2 + bx + c')
    plt.xlabel("X: Values ranging from 0 to 5 [step size of 0.01]")
    plt.ylabel("Y: Computed result of integration over the interval")
    plt.show()