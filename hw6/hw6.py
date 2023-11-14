import numpy as np

# Define constants
NO_OF_TRIALS = 100 # Trials per precision
PRECISIONS = [10 ** (-i) for i in range(1, 8)] # set the range of precision values to look from
MAX_POINTS_IN_TRIAL = 10000

def calculate_radius() -> float:
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)

    return np.sqrt(np.power(x, 2) + np.power(y, 2))

if __name__ == "__main__":
    estimated_pi_vals = list()

    for precision in PRECISIONS:
        print(f'At precision level : {precision}')

        for _ in range(NO_OF_TRIALS):
            inside = 0

            for point in range(MAX_POINTS_IN_TRIAL):
                dist = calculate_radius()

                if dist <= 1:
                    inside += 1
            
            pi_val = 4.0 * (inside) / MAX_POINTS_IN_TRIAL

            if (pi_val - np.pi) < MAX_POINTS_IN_TRIAL:
                estimated_pi_vals.append(pi_val)

        print(f'Estimated value of PI = {np.average(estimated_pi_vals)}')
