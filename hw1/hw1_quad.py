import cmath
import numpy as np

# This program gets the roots of a quadriatic equation.

def get_discriminant(a: int, b: int, c: int) -> int:
    # Get discriminant of a quadriatic equation.
    # Args:
    #     a (int): First coefficient of a quadriatic equation.
    #     b (int): Second coefficient of a quadriatic equation.
    #     c (int): Third coefficient of a quadriatic equation.

    # Returns:
    #     int: The discriminant of the quadriatic equation. 

    return b**2 - 4 * a * c

if __name__ == "__main__":
    # Get user input for cefficient a and convert to integer
    a = int(input("Input coefficient a: "))

    # Get user input for cefficient b
    b = int(input("Input coefficient b: "))

    # Get user input for cefficient c
    c = int(input("Input coefficient c: "))

    # Tolerance is a constant that signifies how close the disriminant is to zero
    # in case of floating point results.
    TOLERANCE = 1e-5

    # Base cases for when a, b and c paramters are zero.
    if a == 0:
        if b == 0:
            print("x can take any value in the real number space.")
            exit(0)
        else:
            root = -c / b
            print("Single Root :", root)
            exit(0)

    # get the discriminant and compute the delta.
    discriminant = get_discriminant(a, b, c)
    delta = discriminant - 0.0

    # checks if the discriminant is positive or negative and compute the roots
    # accordingly for displaying the roots to user.
    if discriminant < 0 and abs(delta) > TOLERANCE:
        root = cmath.sqrt(discriminant)

        rootA = (-b + root) / (2 * a)
        rootB = (-b - root) / (2 * a)
        print("Root 1:", rootA)
        print("Root 2:", rootB)
    elif discriminant > 0 and abs(delta) > TOLERANCE:
        root = np.sqrt(discriminant)
        rootA = (-b + root) / (2 * a)
        rootB = (-b - root) / (2 * a)

        print("Root 1:", rootA)
        print("Root 2:", rootB)
    elif abs(delta) < TOLERANCE:
        root = -b / (2 * a)

        print("Double root:", root)

    exit(0)