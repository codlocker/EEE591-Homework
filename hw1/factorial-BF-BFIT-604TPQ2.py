import numpy as np

"""
Basic recursion factorial program:

def factorial(n: int) -> int:
    if n == 0:
        return 1
    if n == 1:
        return 1
    
    return n * factorial(n-1)

n = int(input("Enter n whose factorial you want to find:"))
print(factorial(n))

"""

# Declare Constants
EPSILON = 0.01

def babylonian_tech(N: int, n_prev: float) -> float:
    """Use Babylonian technique to generate square root of a number.

    Args:
        N (int): The number whose square root is to be calculated
        n_prev (float): the n_(i-1) to calculate n_i

    Returns:
        float: Best possible square root.
    """
    n_new = np.round((n_prev + N / n_prev) / 2, 2)

    if np.abs(n_new - n_prev) <= EPSILON:
        return np.round(n_new, 2)
    else:
        return babylonian_tech(
            N=N,
            n_prev=n_new)

if __name__ == "__main__":
    N = int(input("Enter a number whose square root is desired: "))
    n_0 = int(input("Enter an initial guess: "))
    print(babylonian_tech(
        N=N,
        n_prev=n_0))