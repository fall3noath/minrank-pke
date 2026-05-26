"""
PKE from MinRank (Chatterjee, Mu, Vasudevan 2025)
https://eprint.iacr.org/2025/1833

Implementation of the PKE scheme described in Figure 1 / Section 4.

The scheme is parameterized by:
  n  - matrix dimension (security parameter)
  k  - number of public matrices
  r  - rank bound for noise / low-rank matrices
  t  - blockwise inner product parameter (must divide n)

Constraints from Theorem 4.1:
  1. r^2 < t - log(n)
  2. (n/t)^2 - 2k - 1 = omega(log n)
  3. t divides n

All matrices are n×n over GF(2).

Public key:  pk = (A, Y)  where A = (A_1,...,A_k) and Y = A(s) + E
Secret key:  sk = s  (length-k binary vector)

Ciphertext encrypting bit x:
  x=0: ct = <R, (A,Y)>_t   (k+1 matrices of size t×t)
  x=1: ct = (V_1,...,V_{k+1})  (k+1 uniform random t×t matrices)

Decryption:
  M = C_{k+1} - sum_i s_i * C_i
  Output 0 if rank(M) < t - log^{2/3}(n), else 1
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .gf2 import (
    random_matrix,
    random_vector,
    random_low_rank_matrix,
    linear_combination,
    add,
    blockwise_inner_product_sequence,
    blockwise_inner_product,
    rank,
    zeros,
)


# ---------------------------------------------------------------------------
# Parameter set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    """
    Parameters for the MinRank PKE scheme.

    Attributes
    ----------
    n : int
        Matrix dimension (all matrices are n×n over GF(2)).
    k : int
        Number of public matrices in the MinRank instance.
    r : int
        Rank bound for noise matrices and encryption randomness.
    t : int
        Block parameter for the blockwise inner product (must divide n).
    """
    n: int
    k: int
    r: int
    t: int

    def __post_init__(self):
        assert self.n % self.t == 0, "t must divide n"
        assert self.r * self.r < self.t - math.log2(self.n), (
            f"Correctness condition r^2 < t - log2(n) violated: "
            f"{self.r**2} >= {self.t} - {math.log2(self.n):.2f}"
        )
        b = self.n // self.t
        margin = b * b - 2 * self.k - 1
        assert margin > 0, (
            f"Duality condition (n/t)^2 - 2k - 1 > 0 violated: "
            f"{b**2} - {2*self.k} - 1 = {margin}"
        )

    @classmethod
    def toy(cls) -> "Params":
        """
        Small parameter set for unit tests and demos.

        n=64, t=16, k=4, r=1

        Constraints satisfied:
          r^2 = 1  <  t - log2(n) = 16 - 6 = 10  ✓
          (n/t)^2 - 2k - 1 = 16 - 8 - 1 = 7 > 0  ✓

        Security: none — for functional correctness only.
        """
        return cls(n=64, t=16, k=4, r=1)

    @classmethod
    def small(cls) -> "Params":
        """
        Small parameter set.

        n=128, t=32, k=5, r=2

        Constraints satisfied:
          r^2 = 4  <  t - log2(n) = 32 - 7 = 25  ✓
          (n/t)^2 - 2k - 1 = 16 - 10 - 1 = 5 > 0  ✓

        Security: none — for demonstration only.
        """
        return cls(n=128, t=32, k=5, r=2)

    @classmethod
    def medium(cls) -> "Params":
        """
        Medium parameter set following setting 1 from Section 4 of the paper:
          t = Theta(n^{1/2}), k = Theta(n), r = Theta(n^{1/4})

        n=256, t=32, k=10, r=2

        Constraints satisfied:
          r^2 = 4  <  t - log2(n) = 32 - 8 = 24  ✓
          (n/t)^2 - 2k - 1 = 64 - 20 - 1 = 43 > 0  ✓

        Security level (heuristic): low — increase n significantly for real use.
        """
        return cls(n=256, t=32, k=10, r=2)


# ---------------------------------------------------------------------------
# Key types
# ---------------------------------------------------------------------------

class PublicKey(NamedTuple):
    """
    pk = (A, Y) where:
      A : list of k n×n GF(2) matrices  (the MinRank instance basis)
      Y : n×n GF(2) matrix              (= A(s) + E, the noisy codeword)
    """
    A: list       # list of k np.ndarray, each shape (n, n)
    Y: np.ndarray # shape (n, n)

    @property
    def as_sequence(self) -> list:
        """Return (A_1,...,A_k, Y) as a flat list — used in encryption."""
        return list(self.A) + [self.Y]


class SecretKey(NamedTuple):
    """sk = s, a length-k GF(2) vector."""
    s: np.ndarray  # shape (k,)


class Ciphertext(NamedTuple):
    """
    ct = (C_1, ..., C_{k+1}), each C_i is a t×t GF(2) matrix.
    """
    matrices: list  # list of k+1 np.ndarray, each shape (t, t)


# ---------------------------------------------------------------------------
# Scheme
# ---------------------------------------------------------------------------

class MinRankPKE:
    """
    Public-Key Encryption from the MinRank Problem.

    Reference: Chatterjee, Mu, Vasudevan (2025), Figure 1.

    Usage
    -----
    >>> params = Params.toy()
    >>> scheme = MinRankPKE(params)
    >>> pk, sk = scheme.keygen()
    >>> ct = scheme.encrypt(pk, 1)
    >>> bit = scheme.decrypt(sk, ct, len(pk.A))
    >>> assert bit == 1
    """

    def __init__(self, params: Params, seed: int | None = None):
        self.params = params
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # KeyGen
    # ------------------------------------------------------------------

    def keygen(self) -> tuple[PublicKey, SecretKey]:
        """
        KeyGen(1^n) from Figure 1.

        1. Sample s <-- GF(2)^k
        2. Sample A = (A_1,...,A_k) <-- (GF(2)^{n×n})^k  uniformly
        3. Sample E <-- GF(2)^{n×n}  s.t. rank(E) <= r
        4. Set sk = s,  pk = (A,  A(s) + E)

        Returns
        -------
        (PublicKey, SecretKey)
        """
        p = self.params
        # Secret key: random binary vector of length k
        s = random_vector(p.k, self.rng)

        # Public matrices: k random n×n matrices
        A = [random_matrix(p.n, p.n, self.rng) for _ in range(p.k)]

        # Low-rank noise matrix
        E = random_low_rank_matrix(p.n, p.r, self.rng)

        # Public key: Y = A(s) + E
        As = linear_combination(A, s)
        Y = add(As, E)

        return PublicKey(A=A, Y=Y), SecretKey(s=s)

    # ------------------------------------------------------------------
    # Encrypt
    # ------------------------------------------------------------------

    def encrypt(self, pk: PublicKey, x: int) -> Ciphertext:
        """
        Enc(pk, x) from Figure 1.

        x=0:
          Sample R <-- GF(2)^{n×n} with rank(R) <= r
          ct = <R, (A_1,...,A_k, Y)>_t   (k+1 matrices of size t×t)

        x=1:
          Sample k+1 uniformly random t×t matrices
          ct = (V_1, ..., V_{k+1})

        Parameters
        ----------
        pk : PublicKey
        x  : int in {0, 1}

        Returns
        -------
        Ciphertext
        """
        assert x in (0, 1), "plaintext must be 0 or 1"
        p = self.params
        A_prime = pk.as_sequence  # (A_1,...,A_k, Y), length k+1

        if x == 1:
            # Encrypt 1: k+1 uniformly random t×t matrices
            matrices = [random_matrix(p.t, p.t, self.rng) for _ in range(p.k + 1)]
        else:
            # Encrypt 0: blockwise inner products with a random low-rank R
            R = random_low_rank_matrix(p.n, p.r, self.rng)
            matrices = [blockwise_inner_product(R, Ai, p.t) for Ai in A_prime]

        return Ciphertext(matrices=matrices)

    # ------------------------------------------------------------------
    # Decrypt
    # ------------------------------------------------------------------

    def decrypt(self, sk: SecretKey, ct: Ciphertext, k: int) -> int:
        """
        Dec(sk, ct) from Figure 1.

        1. Parse ct = (C_1, ..., C_k, C_{k+1})
        2. M = C_{k+1} - sum_{i in [k]} s_i * C_i   (over GF(2))
        3. If rank(M) < t - log^{2/3}(n): output 0
           Else:                           output 1

        Parameters
        ----------
        sk : SecretKey
        ct : Ciphertext
        k  : int — number of public matrices (= len(pk.A))

        Returns
        -------
        int : decrypted bit in {0, 1}
        """
        p = self.params
        s = sk.s
        matrices = ct.matrices

        C = matrices[:k]       # C_1, ..., C_k
        C_last = matrices[k]   # C_{k+1}

        # M = C_{k+1} - sum_i s_i * C_i   (subtraction = addition in GF(2))
        M = C_last.copy()
        for si, Ci in zip(s, C):
            if si % 2 == 1:
                M = add(M, Ci)

        rank_M = rank(M)
        threshold = p.t - math.ceil(math.log2(p.n) ** (2 / 3))

        if rank_M < threshold:
            return 0
        else:
            return 1
