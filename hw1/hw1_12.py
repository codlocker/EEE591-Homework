import numpy as np

##########################################################################
# This program calculates the prime numbers from 1 upto the given number.#
# The number given as input here is 10000.                               #
##########################################################################

def get_primes(n: int) -> list[int]:
    
    # Generate prime numbers for all numbers to at most n

    # Args:
    #     n (int): the value till which prime numbers ae checked

    # Returns:
    #     list[int]: list of prime numbers

    if n < 2:
        return []
    
    # This is a boolean array that stores which numbers are prime / non-prime
    # from 0 to n.
  
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False

    for i in range(2, int(np.sqrt(n + 1))):
        if is_prime[i]:
            # Loop through multiples and mark them as non-prime.
            for x in range(i * i, n + 1, i):
                is_prime[x] = False 

    return [i for i in range(n + 1) if is_prime[i]]

if __name__ == "__main__":
    print(get_primes(10000))