import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk

# Calculate the noise from the standard deviation
def calculate_noise(sigma: float):
    noise = (sigma / 100) * np.random.randn(MAX_YEARS)
    return np.asarray(noise)

# calulate the wealth given the current wealth
# using the contribution and retirement age.
def calculate_wealth(
        curr_wealth: float,
        spend_at_retirement: float,
        age: int,
        rate: int,
        years_of_contrib: int, 
        years_to_retire: int,
        yearly_contrib: float, 
        noise: float):
    # Initilize new wealth
    new_wealth = 0

    # Conditions for updating current wealth
    if age <= years_of_contrib:
        new_wealth = curr_wealth * (1 + rate / 100 + noise) + yearly_contrib
    elif age > years_of_contrib and age <= years_to_retire:
        new_wealth = curr_wealth * (1 + rate / 100 + noise)
    elif age > years_to_retire and age <= MAX_YEARS:
        new_wealth = curr_wealth * (1 + rate / 100 + noise) - spend_at_retirement
    else:
        print("You should be dead by now at", age)

    # print(new_wealth)
    return 0 if new_wealth <= 0 else new_wealth

# Calculate accumulated wealth
def calculate_accumulated_wealth(
        m1: tk.Entry,
        m2: tk.Entry,
        m3: tk.Entry,
        m4: tk.Entry,
        m5: tk.Entry,
        m6: tk.Entry,
        l1: tk.Label):
    try:
        # Get the user inputs from the Entity.
        mean_return = float(m1.get())
        stddev_return = float(m2.get())
        yearly_contrib = float(m3.get())
        no_of_years_contrib = int(m4.get())
        no_of_years_retire = int(m5.get())
        annual_spend = float(m6.get())

        wealth = []

        # Run the entire for 10 times.
        for i in range(10):
            noise = calculate_noise(
                sigma=stddev_return)
            curr_wealth = [0]
            for j in range(1, 71):
                new_wealth = calculate_wealth(
                    curr_wealth=curr_wealth[j-1],
                    spend_at_retirement=annual_spend,
                    age=j,
                    rate=mean_return,
                    noise=noise[j-1],
                    yearly_contrib=yearly_contrib,
                    years_to_retire=no_of_years_retire,
                    years_of_contrib=no_of_years_contrib
                )

                # New wealth is zero then we break of current iteration.
                # Wealth cant be negative.
                if new_wealth > 0:
                    curr_wealth.append(new_wealth)
                else:
                    break

            # print("No. of entries in wealth: ", len(curr_wealth))
            # Plot the current wealth in the graph.
            plt.plot(
                list(range(len(curr_wealth))), curr_wealth, marker='x')
            
            wealth.append(curr_wealth[-1])

        mean_wealth = f"{int(np.mean(wealth)): ,}"

        # Update the mean wealth in the text label.
        l1.config(text=f"${mean_wealth}")
        
        # Graph the x-y details for the graph.
        plt.title('Wealth over 70 years')
        plt.xlabel("years")
        plt.xticks(np.arange(0, 71, 10))
        plt.ylabel("wealth")
        plt.show()

    except Exception as e:
        print(f'Exception : {e}')

MAX_YEARS = 70
if __name__ == "__main__":
    # GUI Window
    root = tk.Tk()
    
    # Initialize the variables that is placeholder for the text in the input. 
    mean_return = tk.DoubleVar()
    stddev_return = tk.DoubleVar()
    yearly_contrib = tk.DoubleVar()
    no_of_years_contrib = tk.IntVar()
    no_of_years_retire = tk.IntVar()
    annual_spend = tk.DoubleVar()

    # Create a frame.
    frm = ttk.Frame(root, padding=10)
    frm.grid()

    # Create a entry box of mean returns.
    ttk.Label(frm, text="Mean Return (%)", padding=10).grid(column=0, row=0)
    mean_return_wgt = ttk.Entry(frm, textvariable=mean_return)
    mean_return_wgt.grid(column=1, row=0)

    # Create a entry box of standard deviation.
    ttk.Label(frm, text="Std Dev Return (%)", padding=10).grid(column=0, row=1)
    stdev_wgt = ttk.Entry(frm, textvariable=stddev_return)
    stdev_wgt.grid(column=1, row=1)

    # Create a entry box of yearly contribution.
    ttk.Label(frm, text="Yearly Contribution ($)", padding=10).grid(column=0, row=2)
    yearly_contrib_wgt = ttk.Entry(frm, textvariable=yearly_contrib)
    yearly_contrib_wgt.grid(column=1, row=2)

    # Create a entry box of years of contribution.
    ttk.Label(frm, text="No. Years of Contribution", padding=10).grid(column=0, row=3)
    years_contrib_wgt = ttk.Entry(frm, textvariable=no_of_years_contrib)
    years_contrib_wgt.grid(column=1, row=3)

    # Create an entry box of no. of years to retirement.
    ttk.Label(frm, text="No. Years to Retirement", padding=10).grid(column=0, row=4)
    years_retire_wgt = ttk.Entry(frm, textvariable=no_of_years_retire)
    years_retire_wgt.grid(column=1, row=4)

    # Create an entry box of annual retirement spend.
    ttk.Label(frm, text="Annual Retirement Spend", padding=10).grid(column=0, row=5)
    ret_spent_wgt = ttk.Entry(frm, textvariable=annual_spend)
    ret_spent_wgt.grid(column=1, row=5)

    # Create the final result text as label
    ttk.Label(frm, text='Wealth at retirement:', padding=10).grid(column=0, row=6)

    ret_weath_wgt = ttk.Label(frm , text='', padding=10)
    ret_weath_wgt.grid(column=1, row=6)

    # Create 2 buttons for calculate and exit.
    ttk.Button(
        frm, 
        text="Calculate", 
        command= lambda: calculate_accumulated_wealth(
            mean_return_wgt, 
            stdev_wgt, 
            yearly_contrib_wgt,
            years_contrib_wgt,
            years_retire_wgt,
            ret_spent_wgt,
            ret_weath_wgt)).grid(column=0, row=7)
    
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=7)

    # Register event.
    root.register(calculate_accumulated_wealth)
    # Run the mainloop.
    root.mainloop()