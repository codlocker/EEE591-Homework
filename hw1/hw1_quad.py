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
    a = int(input("Input coefficient a: "))
    b = int(input("Input coefficient b: "))
    c = int(input("Input coefficient c: "))
    TOLERANCE = 1e-5

    if a == 0:
        if b == 0:
            print("x can take any value in the real number space.")
            exit(0)
        else:
            root = -c / b
            print("Single Root :", root)
            exit(0)

    det = get_discriminant(a, b, c)
    delta = det - 0.0

    if det < 0 and abs(delta) > TOLERANCE:
        root = cmath.sqrt(det)

        rootA = (-b + root) / (2 * a)
        rootB = (-b - root) / (2 * a)
        print("Root 1:", rootA)
        print("Root 2:", rootB)
    elif det > 0 and abs(delta) > TOLERANCE:
        root = np.sqrt(det)
        rootA = (-b + root) / (2 * a)
        rootB = (-b - root) / (2 * a)

        print("Root 1:", rootA)
        print("Root 2:", rootB)
    elif abs(delta) < TOLERANCE:
        root = -b / (2 * a)

        print("Double root:", root)

    exit(0)