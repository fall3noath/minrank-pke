"""
Utility functions for matrix arithmetic over GF(2).

All matrices are represented as numpy arrays of dtype uint8
with entries in {0, 1}. All operations are performed mod 2.
"""

import numpy as np


def zeros(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.uint8)


def eye(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.uint8)


def random_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random matrix over GF(2)."""
    return rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)


def random_vector(length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random vector over GF(2)."""
    return rng.integers(0, 2, size=(length,), dtype=np.uint8)


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication over GF(2)."""
    return np.mod(A.astype(np.int32) @ B.astype(np.int32), 2).astype(np.uint8)


def add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix addition over GF(2)."""
    return np.bitwise_xor(A, B)


def scalar_mul(s: int, A: np.ndarray) -> np.ndarray:
    """Scalar multiplication over GF(2)."""
    if s % 2 == 0:
        return zeros(A.shape[0], A.shape[1])
    return A.copy()


def rank(A: np.ndarray) -> int:
    """
    Compute the rank of a matrix over GF(2) via Gaussian elimination.
    """
    M = A.copy().astype(np.int32)
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        # Find pivot
        found = -1
        for row in range(pivot_row, rows):
            if M[row, col] % 2 == 1:
                found = row
                break
        if found == -1:
            continue
        # Swap
        M[[pivot_row, found]] = M[[found, pivot_row]]
        # Eliminate
        for row in range(rows):
            if row != pivot_row and M[row, col] % 2 == 1:
                M[row] = (M[row] + M[pivot_row]) % 2
        pivot_row += 1
    return pivot_row


def random_low_rank_matrix(n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random n×n matrix over GF(2) of rank exactly r.
    Strategy: draw random L (n×r) and R (r×n), compute L*R mod 2.
    Retry until rank is exactly r (almost always first try for r << n).
    """
    while True:
        L = random_matrix(n, r, rng)
        R = random_matrix(r, n, rng)
        E = matmul(L, R)
        if rank(E) == r:
            return E


def frobenius_inner_product(A: np.ndarray, B: np.ndarray) -> int:
    """
    Frobenius inner product of two matrices over GF(2):
      <A, B>_F = Tr(A^T B) = sum_{i,j} A_{ij} * B_{ij}  (mod 2)
    Returns a scalar in {0, 1}.
    """
    return int(np.sum(np.bitwise_and(A, B)) % 2)


def blockwise_inner_product(A: np.ndarray, B: np.ndarray, t: int) -> np.ndarray:
    """
    t-blockwise inner product <A, B>_t as defined in Definition 3.7.

    A and B are n×n matrices over GF(2). t must divide n.
    The result is a t×t matrix over GF(2) where the (i,j)-th entry
    is the Frobenius inner product of the (i,j)-th (n/t)×(n/t) blocks.

    Parameters
    ----------
    A, B : np.ndarray
        n×n GF(2) matrices.
    t : int
        Block parameter; must divide n.

    Returns
    -------
    np.ndarray
        t×t GF(2) matrix.
    """
    n = A.shape[0]
    assert n % t == 0, "t must divide n"
    b = n // t  # block size

    result = np.zeros((t, t), dtype=np.uint8)
    for i in range(t):
        for j in range(t):
            A_block = A[i * b:(i + 1) * b, j * b:(j + 1) * b]
            B_block = B[i * b:(i + 1) * b, j * b:(j + 1) * b]
            result[i, j] = frobenius_inner_product(A_block, B_block)
    return result


def blockwise_inner_product_sequence(
    A_seq: list[np.ndarray], B: np.ndarray, t: int
) -> list[np.ndarray]:
    """
    Compute <A_seq, B>_t = (<A_1, B>_t, <A_2, B>_t, ..., <A_k, B>_t).

    Parameters
    ----------
    A_seq : list of np.ndarray
        Sequence of k n×n GF(2) matrices.
    B : np.ndarray
        n×n GF(2) matrix.
    t : int
        Block parameter; must divide n.

    Returns
    -------
    list of np.ndarray
        List of k t×t GF(2) matrices.
    """
    return [blockwise_inner_product(Ai, B, t) for Ai in A_seq]


def linear_combination(A_seq: list[np.ndarray], s: np.ndarray) -> np.ndarray:
    """
    Compute A(s) = sum_i s_i * A_i  over GF(2).

    Parameters
    ----------
    A_seq : list of np.ndarray
        Sequence of k n×n GF(2) matrices.
    s : np.ndarray
        Length-k GF(2) vector.

    Returns
    -------
    np.ndarray
        n×n GF(2) matrix.
    """
    n = A_seq[0].shape[0]
    result = zeros(n, n)
    for si, Ai in zip(s, A_seq):
        if si % 2 == 1:
            result = add(result, Ai)
    return result
