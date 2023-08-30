import numpy as np

# Basic recursion factorial program:

# def factorial(n: int) -> int:
#    if n == 0:
#        return 1
#    if n == 1:
#        return 1   
#    return n * factorial(n-1)

# n = int(input("Enter n whose factorial you want to find:"))
# print(factorial(n))

def babylonian_tech(number: int, n_prev: float, epsilon: float) -> float:
    # Use Babylonian technique to generate square root of a number.

    # Args:
    #     N (int): The number whose square root is to be calculated
    #     n_prev (float): the n_(i-1) to calculate n_i
    #     epsilon (float): the tolerance to look for final result.

    # Returns:
    #     float: Best possible square root.
    n_new = np.round((n_prev + number / n_prev) / 2, 2)

    # Base clase: check the delta from epsilon to determine
    # whether to recurse further or return the value rounded
    # to 2 decimal places.
    if np.abs(n_new - n_prev) <= epsilon:
        return np.round(n_new, 2)
    else:
        return babylonian_tech(
            number=number,
            n_prev=n_new,
            epsilon= epsilon)

if __name__ == "__main__":
    # User provides the number whose square root is desired.
    number = int(input("Enter a number whose square root is desired: "))

    # An initial guess is required for using the Babylonian technique.
    n_0 = int(input("Enter an initial guess: "))
    
    # This constant helps in computing the delta.
    EPSILON = 0.01

    # Output format for the result.
    print("The square root of {} is {}".format(number, babylonian_tech(
        epsilon=EPSILON,
        number=number,
        n_prev=n_0)))
    
    exit(0)