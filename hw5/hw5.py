import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the common stuff
time_vec = np.linspace(0, 7, 700)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
###############################################################
# Problem 1: y' = cos(t) solver using odeint.                 #
###############################################################
def calc_derivative_1(ypos: int, t: np.array):
    return np.cos(t)

yvec_1 = odeint(calc_derivative_1, 1, time_vec)

ax1.plot(time_vec, yvec_1, marker='*', label='y = sin(t)')
ax1.set_xlabel('Time')
ax1.set_ylabel('y', rotation=0)
ax1.set_title('Problem 1 graph')
ax1.legend()


###############################################################
# Problem 2: y' = -y + t2e-2t + 10 solver using odeint.       #
###############################################################
def calc_derivative_2(ypos: int, t: np.array):
    return -ypos + np.power(t, 2) * np.exp(-2 * t) + 10

yvec_2 = odeint(calc_derivative_2, 0, time_vec)

ax2.plot(time_vec, yvec_2, marker='*', label='y vs t')
ax2.set_xlabel('Time')
ax2.set_ylabel('y', rotation=0)
ax2.set_title('Problem 2 graph')
ax2.legend()

#######################################################################
# Problem 3: y'' + 4y' + 4y = 25cos(t) + 25sin(t) solver using odeint.#
#######################################################################

# Init y values and its corresponding derivatives.
y_init_prob = [1, 1]

def calc_derivative_3(y_init: list, t: np.array):
    y = y_init[0]
    dy = y_init[1]
    d2y = 25 * (np.cos(t) + np.sin(t)) - 4 * (dy + y)
    return dy, d2y

yvec_3 = odeint(calc_derivative_3, y_init_prob, time_vec) 
ax3.plot(time_vec, yvec_3[:, 0], label='y vs t', marker='.')
ax3.plot(time_vec, yvec_3[:, 1], label="y' vs t", marker='*')
ax3.set_xlabel('Time')
ax3.set_ylabel('y', rotation=0)
ax3.set_title('Problem 3 graph')
ax3.legend()
plt.show()