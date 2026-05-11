"""
Initial program for top-k largest numbers optimization example.

Problem:
Given n unsorted numbers and an integer k, return the top-k largest
numbers in descending order.

The goal is to evolve a more efficient implementation. The best general
algorithm for k << n is to maintain a min-heap of size k, then sort the
remaining k values:

    O(n log k + k log k)

The initial implementation below intentionally uses a simple full bubble
sort, which is inefficient for large arrays.
"""

import random


RANDOM_SEED = 12345


def generate_test_array(size=100):
    """Generate a random array for testing."""
    return [random.randint(-100000, 100000) for _ in range(size)]


# EVOLVE-BLOCK-START
def top_k_largest(arr, k):
    """
    Return the k largest values from arr in descending order.

    Current implementation: fully sort the whole array using bubble sort,
    then take the largest k values. This is correct, but inefficient.
    """
    if k <= 0:
        return []

    arr = arr.copy()  # Don't modify the original
    n = len(arr)

    if k >= n:
        k = n

    # Bubble sort in ascending order.
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    # Largest k values are at the end, so reverse them for descending order.
    result = []
    for i in range(k):
        result.append(arr[n - 1 - i])

    return result
# EVOLVE-BLOCK-END


def verify_top_k(original, result, k):
    """Verify that result contains exactly the top-k largest values in order."""
    if k <= 0:
        return result == []

    expected = sorted(original, reverse=True)[:k]
    return result == expected


def main():
    """Run the top-k algorithm on test cases."""
    import time

    random.seed(RANDOM_SEED)

    # Each case is (array size n, requested k).
    # These are chosen so the naive version is slow but still evaluable.
    test_cases = [
        (100, 10),
        (500, 25),
        (1500, 50),
        (3000, 75),
        (5000, 100),
    ]

    total_time = 0.0

    for size, k in test_cases:
        test_array = generate_test_array(size)

        start_time = time.perf_counter()
        result = top_k_largest(test_array, k)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        total_time += elapsed

        # Verify correctness.
        if not verify_top_k(test_array, result, k):
            print(f"ERROR: top-k incorrect for size {size}, k {k}!")
            return None

        print(f"Size {size:5d}, k {k:4d}: {elapsed*1000:.3f} ms")

    print(f"Total time: {total_time*1000:.3f} ms")
    return total_time


if __name__ == "__main__":
    main()
