import numpy as np

# Define constants
NO_OF_TRIALS = 100 # Trials per precision
PRECISIONS = [10 ** (-i) for i in range(1, 8)] # set the range of precision values to look from
MAX_POINTS_IN_TRIAL = 10000 # This is the maximum number of points we see till we get the precision desired.

# Calculate the distance from origin.
# Returns:
#    float: _description_
def calculate_radius() -> float:
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)

    return np.sqrt(np.power(x, 2) + np.power(y, 2))

if __name__ == "__main__":
    for precision in PRECISIONS:
        pi_vals = 0
        success_count = 0

        # Loop through the 100 trials for PI
        for _ in range(NO_OF_TRIALS):
            inside = 0

            # Loops through 10000 trials to get number of times point si inside the circle
            for points in range(1, MAX_POINTS_IN_TRIAL + 1):
                dist = calculate_radius()

                if dist <= 1:
                    inside += 1
            
                pi_val = 4.0 * inside / points

                if abs(pi_val - np.pi) < precision:
                    pi_vals += pi_val
                    success_count += 1
                    break

        if success_count > 0:
            print(f'{precision} success {success_count} times {pi_vals / success_count}')
        else:
            print(f"{precision} no success")
