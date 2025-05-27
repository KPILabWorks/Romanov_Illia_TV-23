import math
from typing import List

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(math.sqrt(n))
    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
    return True

def filter_primes(numbers: List[int]) -> List[int]:
    return [x for x in numbers if is_prime(x)]

if __name__ == "__main__":
    data = [0, 1, 2, 3, 4, 16, 17, 19, 20, 23, 24, 29]
    primes = filter_primes(data)
    print("Вхідні дані:", data)
    print("Прості числа:", primes)