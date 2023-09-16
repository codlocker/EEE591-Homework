import numpy as np
from tkinter import *
from tkinter import ttk

def calculate_noise(sigma: float):
    noise = (sigma / 100) * np.random.randn(MAX_YEARS)
    return np.asarray(noise)

def calculate_wealth(
        curr_wealth: float,
        spend_at_retirement: float,
        age: int,
        rate: int,
        years_of_contrib: int, 
        years_to_retire: int,
        yearly_contrib: float, 
        noise: float):
    new_wealth = 0
    if age <= years_of_contrib:
        new_wealth = curr_wealth * (1 + rate / 100 + noise) + yearly_contrib
    elif age > years_of_contrib and age <= years_of_contrib + years_to_retire:
        new_wealth = curr_wealth * (1 + rate / 100 + noise)
    elif age > MAX_YEARS:
        new_wealth = curr_wealth * (1 + rate / 100 + noise) - spend_at_retirement
    else:
        print("Incorrect Age")

    # print(new_wealth)
    return 0 if new_wealth < 0 else new_wealth

MAX_YEARS = 70
if __name__ == "__main__":
    # GUI Window
    root = Tk()
    
    # Create a frame.
    frm = ttk.Frame(root, padding=10)
    frm.grid()

    # Create a entry box of mean returns.
    ttk.Label(frm, text="Mean Return (%)", padding=10).grid(column=0, row=0)
    ttk.Entry(frm).grid(column=1, row=0)

    # Create a entry box of standard deviation.
    ttk.Label(frm, text="Std Dev Return (%)", padding=10).grid(column=0, row=1)
    ttk.Entry(frm).grid(column=1, row=1)

    # Create a entry box of yearly contribution.
    ttk.Label(frm, text="Yearly Contribution ($)", padding=10).grid(column=0, row=2)
    ttk.Entry(frm).grid(column=1, row=2)

    # Create a entry box of years of contribution.
    ttk.Label(frm, text="No. Years of Contribution", padding=10).grid(column=0, row=3)
    ttk.Entry(frm).grid(column=1, row=3)

    # Create an entry box of no. of years to retirement.
    ttk.Label(frm, text="No. Years to Retirement", padding=10).grid(column=0, row=4)
    ttk.Entry(frm).grid(column=1, row=4)

    # Create an entry box of annual retirement spend.
    ttk.Label(frm, text="Annual Retirement Spend", padding=10).grid(column=0, row=5)
    ttk.Entry(frm).grid(column=1, row=5)

    # Create 2 buttons for calculate and exit.
    ttk.Button(frm, text="Calculate", command=root.destroy).grid(column=0, row=6)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=6)

    root.mainloop()

    mean_return = 6
    stddev_return = 20
    yearly_contrib = 10000
    no_of_years_contrib = 30
    no_of_years_retire = 40
    annual_spend = 80000

    wealth = np.zeros((10, 71))

    for i in range(10):
        noise = calculate_noise(
            sigma=stddev_return)
        for j in range(1, 71):
            wealth[i, j] = calculate_wealth(
                curr_wealth=wealth[i, j-1],
                spend_at_retirement=annual_spend,
                age=i,
                rate=mean_return,
                noise=noise[j-1],
                yearly_contrib=yearly_contrib,
                years_to_retire=no_of_years_retire,
                years_of_contrib=no_of_years_contrib
            )
    print(wealth)

