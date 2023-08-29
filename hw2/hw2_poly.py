import scipy.integrate as spi
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    R = 5
    STEPS = 0.01
    no_points = int(R / STEPS)
    values = np.linspace(0, R, no_points)

    y_1 = [spi.quad(
          func=lambda x, a, b, c : a * x**2 + b * x + c,
          a=0,
          b=value,
          args=(2, 3, 4))[0] for value in values]

    y_2 = [spi.quad(
          func=lambda x, a, b, c : a * x**2 + b * x + c,
          a=0,
          b=value,
          args=(2, 1, 1))[0] for value in values]

    plt.plot(values, y_1, color='blue', label='(2, 3, 4)')
    plt.plot(values, y_2, color='red', label='(2, 1, 1)')
    leg = plt.legend(loc='upper center')

    plt.title('Integration of ax^2 + bx + c')
    plt.xlabel("X: Values ranging from 0 to 5 [step size of 0.01]")
    plt.ylabel("Y: Computed result of integration over the interval")

    plt.show()