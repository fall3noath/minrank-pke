
"""
Tests for the MinRank PKE scheme.

We test:
  1. Correctness: Dec(sk, Enc(pk, x)) == x for x in {0, 1}
  2. GF(2) utilities: rank, blockwise inner product, low-rank sampling
  3. Ciphertext structure
"""

import math
import numpy as np
import pytest

from minrank_pke import MinRankPKE, Params
from minrank_pke.gf2 import (
    rank,
    random_low_rank_matrix,
    blockwise_inner_product,
    frobenius_inner_product,
    linear_combination,
    add,
    matmul,
    random_matrix,
)


# ---------------------------------------------------------------------------
# GF(2) utility tests
# ---------------------------------------------------------------------------

class TestGF2Utilities:

    def test_rank_identity(self):
        I = np.eye(4, dtype=np.uint8)
        assert rank(I) == 4

    def test_rank_zeros(self):
        Z = np.zeros((4, 4), dtype=np.uint8)
        assert rank(Z) == 0

    def test_rank_known_matrix(self):
        # [[1,0],[1,0]] has rank 1 over GF(2)
        M = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        assert rank(M) == 1

    def test_rank_random_low_rank(self):
        rng = np.random.default_rng(42)
        for r_target in [1, 2, 3]:
            E = random_low_rank_matrix(16, r_target, rng)
            assert rank(E) == r_target

    def test_frobenius_orthogonal(self):
        A = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        B = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        assert frobenius_inner_product(A, B) == 0

    def test_frobenius_self(self):
        # <A,A>_F = sum of squared entries = sum of entries (mod 2) since 0^2=0, 1^2=1
        A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        expected = int(np.sum(A) % 2)
        assert frobenius_inner_product(A, A) == expected

    def test_blockwise_inner_product_shape(self):
        rng = np.random.default_rng(0)
        n, t = 8, 2
        A = random_matrix(n, n, rng)
        B = random_matrix(n, n, rng)
        result = blockwise_inner_product(A, B, t)
        assert result.shape == (t, t)
        assert result.dtype == np.uint8
        assert set(result.flatten().tolist()).issubset({0, 1})

    def test_blockwise_inner_product_bilinearity(self):
        rng = np.random.default_rng(1)
        n, t = 8, 2
        A = random_matrix(n, n, rng)
        B = random_matrix(n, n, rng)
        C = random_matrix(n, n, rng)
        # <A+B, C>_t = <A,C>_t + <B,C>_t
        lhs = blockwise_inner_product(add(A, B), C, t)
        rhs = add(blockwise_inner_product(A, C, t), blockwise_inner_product(B, C, t))
        np.testing.assert_array_equal(lhs, rhs)

    def test_blockwise_inner_product_symmetry(self):
        rng = np.random.default_rng(2)
        n, t = 8, 4
        A = random_matrix(n, n, rng)
        B = random_matrix(n, n, rng)
        np.testing.assert_array_equal(
            blockwise_inner_product(A, B, t),
            blockwise_inner_product(B, A, t),
        )

    def test_linear_combination_zero_vector(self):
        rng = np.random.default_rng(3)
        A = [random_matrix(4, 4, rng) for _ in range(3)]
        s = np.zeros(3, dtype=np.uint8)
        result = linear_combination(A, s)
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.uint8))

    def test_linear_combination_all_ones(self):
        rng = np.random.default_rng(4)
        A = [random_matrix(4, 4, rng) for _ in range(3)]
        s = np.ones(3, dtype=np.uint8)
        expected = A[0].copy()
        for Ai in A[1:]:
            expected = add(expected, Ai)
        np.testing.assert_array_equal(linear_combination(A, s), expected)


# ---------------------------------------------------------------------------
# Param validation tests
# ---------------------------------------------------------------------------

class TestParams:

    def test_toy_params_valid(self):
        p = Params.toy()
        assert p.n == 64
        assert p.t == 16
        assert p.n % p.t == 0

    def test_small_params_valid(self):
        p = Params.small()
        assert p.n == 128
        assert p.n % p.t == 0

    def test_invalid_t_does_not_divide_n(self):
        with pytest.raises(AssertionError):
            Params(n=64, k=4, r=1, t=3)  # 3 does not divide 64

    def test_correctness_condition(self):
        # r^2 must be < t - log2(n)
        # n=64, t=16, log2(64)=6 => t - log2(n) = 10; r=4 => r^2=16 >= 10 → fails
        with pytest.raises(AssertionError):
            Params(n=64, k=4, r=4, t=16)

    def test_duality_condition(self):
        # (n/t)^2 - 2k - 1 must be > 0
        # n=64, t=16 => (n/t)^2=16; k=8 => 16-16-1=-1 < 0 → fails
        with pytest.raises(AssertionError):
            Params(n=64, k=8, r=1, t=16)


# ---------------------------------------------------------------------------
# Scheme correctness tests
# ---------------------------------------------------------------------------

class TestCorrectness:

    @pytest.fixture
    def params(self):
        return Params.toy()

    @pytest.fixture
    def scheme(self, params):
        return MinRankPKE(params, seed=42)

    def test_keygen_types(self, scheme, params):
        pk, sk = scheme.keygen()
        assert len(pk.A) == params.k
        assert pk.Y.shape == (params.n, params.n)
        assert sk.s.shape == (params.k,)
        assert set(sk.s.tolist()).issubset({0, 1})

    def test_pk_is_noisy_codeword(self, scheme, params):
        """Y = A(s) + E where rank(Y - A(s)) <= r."""
        pk, sk = scheme.keygen()
        As = linear_combination(pk.A, sk.s)
        noise = add(pk.Y, As)
        assert rank(noise) <= params.r

    def test_encrypt_zero_ciphertext_shape(self, scheme, params):
        pk, sk = scheme.keygen()
        ct = scheme.encrypt(pk, 0)
        assert len(ct.matrices) == params.k + 1
        for C in ct.matrices:
            assert C.shape == (params.t, params.t)

    def test_encrypt_one_ciphertext_shape(self, scheme, params):
        pk, sk = scheme.keygen()
        ct = scheme.encrypt(pk, 1)
        assert len(ct.matrices) == params.k + 1
        for C in ct.matrices:
            assert C.shape == (params.t, params.t)

    def test_decrypt_zero(self, scheme, params):
        """Dec(sk, Enc(pk, 0)) should return 0."""
        pk, sk = scheme.keygen()
        ct = scheme.encrypt(pk, 0)
        result = scheme.decrypt(sk, ct, params.k)
        assert result == 0, f"Expected 0, got {result}"

    def test_decrypt_one(self, scheme, params):
        """Dec(sk, Enc(pk, 1)) should return 1."""
        pk, sk = scheme.keygen()
        ct = scheme.encrypt(pk, 1)
        result = scheme.decrypt(sk, ct, params.k)
        assert result == 1, f"Expected 1, got {result}"

    def test_correctness_many_trials(self):
        """Run many encrypt/decrypt cycles to catch probabilistic failures."""
        params = Params.toy()
        scheme = MinRankPKE(params)
        successes = 0
        trials = 50
        for _ in range(trials):
            pk, sk = scheme.keygen()
            for x in (0, 1):
                ct = scheme.encrypt(pk, x)
                result = scheme.decrypt(sk, ct, params.k)
                if result == x:
                    successes += 1
        # Expect near-perfect correctness (allow 1 failure in 100 for safety)
        assert successes >= 2 * trials - 1, (
            f"Only {successes}/{2*trials} correct decryptions"
        )

    def test_wrong_key_does_not_decrypt(self):
        """Decrypting with a wrong secret key should not reliably give 0."""
        params = Params.toy()
        scheme = MinRankPKE(params, seed=99)
        rng2 = np.random.default_rng(999)

        wrong_correct = 0
        trials = 30
        for _ in range(trials):
            pk, sk = scheme.keygen()
            # Flip all bits of secret key
            wrong_s = 1 - sk.s
            from minrank_pke import SecretKey
            wrong_sk = SecretKey(s=wrong_s)
            ct = scheme.encrypt(pk, 0)
            result = scheme.decrypt(wrong_sk, ct, params.k)
            if result == 0:
                wrong_correct += 1

        # With wrong key, should not always decrypt correctly
        # (for a toy scheme this is a soft check — just ensure it's not perfect)
        # If it's always 0 something weird is happening
        assert wrong_correct < trials, (
            "Wrong key always decrypts to 0 — something is wrong"
        )

    def test_small_params_correctness(self):
        """Verify correctness also holds for the 'small' parameter set."""
        params = Params.small()
        scheme = MinRankPKE(params, seed=7)
        for _ in range(10):
            pk, sk = scheme.keygen()
            for x in (0, 1):
                ct = scheme.encrypt(pk, x)
                result = scheme.decrypt(sk, ct, params.k)
                assert result == x


# ---------------------------------------------------------------------------
# Decrypt correctness derivation test
# ---------------------------------------------------------------------------

class TestDecryptionLogic:
    """
    Verify the core claim: when x=0,
      M = C_{k+1} - sum_i s_i * C_i = <R, E>_t
    and rank(<R, E>_t) <= r^2 < t - log(n).
    """

    def test_M_equals_blockwise_RE(self):
        """
        Manually verify that M in decryption equals <R, E>_t.
        """
        params = Params.toy()
        scheme = MinRankPKE(params, seed=0)
        rng = np.random.default_rng(0)

        pk, sk = scheme.keygen()
        A_prime = pk.as_sequence  # (A_1,...,A_k, Y)

        # Manually encrypt 0
        R = random_low_rank_matrix(params.n, params.r, rng)
        ct_matrices = [blockwise_inner_product(R, Ai, params.t) for Ai in A_prime]
        from minrank_pke import Ciphertext
        ct = Ciphertext(matrices=ct_matrices)

        # Manually compute M
        C = ct_matrices[:params.k]
        C_last = ct_matrices[params.k]
        M = C_last.copy()
        for si, Ci in zip(sk.s, C):
            if si % 2 == 1:
                M = add(M, Ci)

        # Compute expected: <R, E>_t where E = Y - A(s)
        As = linear_combination(pk.A, sk.s)
        E = add(pk.Y, As)
        expected_M = blockwise_inner_product(R, E, params.t)

        np.testing.assert_array_equal(M, expected_M)

    def test_rank_M_bounded_when_x_is_0(self):
        """rank(<R,E>_t) <= r^2 (from Claim 3.15 in the paper)."""
        params = Params.toy()
        scheme = MinRankPKE(params, seed=1)

        for _ in range(20):
            pk, sk = scheme.keygen()
            ct = scheme.encrypt(pk, 0)

            C = ct.matrices[:params.k]
            C_last = ct.matrices[params.k]
            M = C_last.copy()
            for si, Ci in zip(sk.s, C):
                if si % 2 == 1:
                    M = add(M, Ci)

            assert rank(M) <= params.r ** 2, (
                f"rank(M)={rank(M)} > r^2={params.r**2}"
            )
