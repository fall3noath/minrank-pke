"""
MinRank Public Key Encryption - from the MinRank problem over GF(2)

Implementation of the PKE scheme from:
Chatterjee, Mu, Vasudevan (2025) - Public-Key Encryption from the MinRank Problem
https://eprint.iacr.org/2025/1833
"""

from .scheme import MinRankPKE, Params, PublicKey, SecretKey, Ciphertext
from .gf2 import (
    rank,
    random_matrix,
    random_vector,
    random_low_rank_matrix,
    blockwise_inner_product,
    blockwise_inner_product_sequence,
    frobenius_inner_product,
    linear_combination,
    add,
    matmul,
    zeros,
    eye,
)

__all__ = [
    'MinRankPKE',
    'Params',
    'PublicKey',
    'SecretKey',
    'Ciphertext',
    'rank',
    'random_matrix',
    'random_vector',
    'random_low_rank_matrix',
    'blockwise_inner_product',
    'blockwise_inner_product_sequence',
    'frobenius_inner_product',
    'linear_combination',
    'add',
    'matmul',
    'zeros',
    'eye',
]
