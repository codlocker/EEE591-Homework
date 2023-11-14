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
    for precision in PRECISIONS:
        pi_vals = 0
        success_count = 0
        for _ in range(NO_OF_TRIALS):
            inside = 0

            for point in range(MAX_POINTS_IN_TRIAL):
                dist = calculate_radius()

                if dist <= 1:
                    inside += 1
            
            pi_val = 4.0 * (inside) / MAX_POINTS_IN_TRIAL

            if abs(pi_val - np.pi) < precision:
                pi_vals += pi_val
                success_count += 1

        if success_count > 0:
            print(f'{precision} success {success_count} times {pi_vals / success_count}')
        else:
            print(f"{precision} no success")
