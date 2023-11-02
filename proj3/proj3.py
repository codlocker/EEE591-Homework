import numpy as np
import pandas as pd
from scipy.optimize import fsolve, leastsq
import scipy.constants as ct
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Well known constants
Q = ct.physical_constants['atomic unit of charge'][0]
KB = ct.physical_constants['Boltzmann constant'][0]


# Constants for Problem 1
# Saturation Current
I_SAT=1e-9

# Ideality
N=1.7

# Resistance
R_1 = 11000

# Temperature
T_1 = 350

# Voltage
init_voltage = np.arange(0.1, 2.5, 0.1)
step_size = (2.5 - 0.1) / 0.1
V = np.ones(int(step_size + 1))

# Current through Diode
# Returns the current diode
def Idiode(
    V: float,
    n: float,
    T: int,
    Is: float):
    return Is * (np.exp((Q * V) / (n * KB * T)) - 1)

# Solve for diode voltage
def solve_diode_v(
    prev_v: float,
    src_v: float,
    r_val: int,
    ideality: float,
    temp: int,
    isat_val: float):
    # This is the constant in diode current equation
    vt = (ideality * KB * temp) / Q
    diode_current = isat_val * (np.exp(prev_v / vt) - 1)
    # Solve diode voltage
    return (prev_v - src_v) / r_val + diode_current



diode_voltage = fsolve(solve_diode_v, V, args=(init_voltage, R_1, N, T_1, I_SAT))
diode_voltage




diode_current = Idiode(
    V=diode_voltage,
    n=N,
    T=T_1,
    Is=I_SAT)
diode_current




log_diode_current = np.log10(diode_current)




# Problem 2
# Resistance
R_2 = 10000

# initial ideality
IDEALITY = 1.5

# Initial phi
PHI = 0.8

# temperature
T_2 = 375

# Area
AREA = 1e-8

# Initial voltage
P1_VDD_STEP = 0.1

# Tolerance
THRESHOLD = 1e-6

# MAx iterations
MAX_ITER = 1e6




# Read the contents of diode.txt file
data = pd.read_csv('DiodeIV.txt', header=None, sep=' ')
source_voltages = []
measured_current = []
for V_s, I_d in data.values:
    source_voltages.append(V_s)
    measured_current.append(I_d)
    
source_voltages = np.asarray(source_voltages)
measured_current = np.asarray(measured_current)




print("Measured Current: ", measured_current)

def solve_current_diode(
    A: float, 
    phi: float,
    R: float,
    n: float, 
    temp: float,
    v_s: np.array):
    
    # create zero array to store computed diode current/voltage
    diode_voltage_est = np.zeros_like(v_s)
    current_diode = np.zeros_like(v_s)
    # specify initial diode voltage for fsolve()
    v_guess = P1_VDD_STEP
    is_val = A * temp * temp * np.exp(-phi * Q / ( KB * temp ) )
    
    # for every given source voltage, calculate diode voltage by solving nodal analysis
    for index in range(len(v_s)):
        v_guess = fsolve(solve_diode_v, v_guess, (v_s[index], R, n, temp, is_val),
                                xtol = 1e-12)[0]
        diode_voltage_est[index] = v_guess    
    # compute the diode current
    vt = (n * KB * temp) / Q  # calc constant for diode current equation
    current_diode = is_val * (np.exp(diode_voltage_est / vt) - 1) # calc diode current by its definition
    return current_diode

def opt_r(R_guess: float, 
          phi_guess: float,
          n_guess: float,
          A: float,
          T: int,
          v_src: np.array,
          current_meas: np.array):  
    # diode current is obtained using optimized parameters
    current_diode = solve_current_diode(A, phi_guess, R_guess, n_guess, T, v_src)
    # Absolute error
    return (current_diode - measured_current)

    
def opt_phi(phi_guess: float,
            R_guess: float,
            n_guess: float,
            A: float,
            T: int,
            v_src: np.array,
            current_meas: np.array):
    # diode current is obtained using optimized parameters
    current_diode = solve_current_diode(A, phi_guess, R_guess, n_guess, T, v_src)
    # Normalized error is obtained by adding a contant in denominator for handling 0/0 case
    return (current_diode - measured_current) / (current_diode + measured_current + 1e-15)
    
def opt_n(n_guess: float,
          R_guess: float,
          phi_guess: float,
          A: float,
          T: int,
          v_src: np.array,
          current_meas: np.array):
    # diode current is obtained using optimized parameters
    current_diode = solve_current_diode(A, phi_guess, R_guess, n_guess, T, v_src)
    # Normalized error is obtained by adding a contant in denominator for handling 0/0 case
    return (current_diode - measured_current) / (current_diode + measured_current + 1e-15)


err = 10
iteration = 1
r_val = R_2
n_val = IDEALITY
phi_val = PHI
current_pred = None
while err > THRESHOLD and iteration < MAX_ITER:
    r_val_opt = leastsq(opt_r, r_val, args=(phi_val, n_val, AREA, T_2, source_voltages, measured_current))
    r_val = r_val_opt[0][0]
    
    n_val_opt = leastsq(opt_n, n_val, args=(r_val, phi_val, AREA, T_2, source_voltages, measured_current))
    n_val = n_val_opt[0][0]
    
    phi_val_opt = leastsq(opt_phi, phi_val, args=(r_val, n_val, AREA, T_2, source_voltages, measured_current))
    phi_val = phi_val_opt[0][0]
    
    current_pred = solve_current_diode(AREA, phi_val, r_val, n_val, T_2, source_voltages)
    # calc error values array for optimizing result check
    err = np.linalg.norm((current_pred - measured_current) / (current_pred + measured_current + 1e-15), ord = 1)
    
    # Print iteration progress
    if iteration % 1 == 0:
        print(f"Iter#: {iteration} ; phi: {phi_val} ; n: {n_val} ; R: {r_val} ; Residual Error: {err}")

    # Increment the iteration count
    iteration += 1



print("\n\n==========\n")
print(f"Took {iteration} iterations to complete...")
print(f"ESTIMATED RESISTANCE (R) : {r_val}")
print(f"ESTIMATED Phi : {phi_val}")
print(f"ESTIMATED Ideality factor (n) : {n_val}")




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# Problem 1 graph
ax1.plot(init_voltage, log_diode_current, label="Source Voltage vs Diode current", marker="*")
ax1.plot(diode_voltage, log_diode_current, label="Diode Voltage vs Diode current", marker="*")
ax1.set_ylabel("Diode current (log scale)")
ax1.set_xlabel("Diode voltage (volts)")
ax1.set_title("Problem 1 plot")
ax1.legend()

# Set axes params for the plot
ax2.set_xlabel("Diode voltage in volts")
ax2.set_ylabel("Diode current in log scale")
ax2.set_title("Problem 2 plot")

# Plot the given values of Source voltage and diode current in DiodeIV.txt (Actual values)
ax2.plot(source_voltages, np.log(measured_current), label="Measured Current", marker='*')

# Plot the given values of diode current and diode voltage in DiodeIV.txt (Predicted values)
ax2.plot(source_voltages, np.log(current_pred), label="Predicted Current", marker='*')
ax2.legend()
plt.show()
