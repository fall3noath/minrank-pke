# minrank-pke

A Python implementation of the public-key encryption scheme introduced in:

> **Public-Key Encryption from the MinRank Problem**  
> Rohit Chatterjee, Changrui Mu, Prashant Nalini Vasudevan  
> IACR ePrint 2025/1833 · https://eprint.iacr.org/2025/1833

This is the **first public implementation** of this scheme.

---

## What is this?

The MinRank problem asks: given `k` matrices `A_1, ..., A_k` over GF(2) and a target matrix `Y`, find a binary vector `s` such that `Y - sum_i s_i * A_i` has rank at most `r`.

This scheme builds a PKE from the average-case hardness of that problem, using:
- A **blockwise inner product** (Definition 3.7 in the paper) as the core algebraic tool
- A **duality reduction** showing the dual MinRank problem is as hard as the primal
- An Alekhnovich-style construction adapted to the rank metric

All arithmetic is over `n x n` matrices over GF(2).

---

## Scheme (Figure 1 of the paper)

**Parameters:** $n$, $k$, $r$, $t$ where $t \mid n$, $r^2 < t - \log n$, and $(n/t)^2 - 2k - 1 = \omega(\log n)$

**KeyGen:**
1. Sample $s \overset{\$}{\leftarrow} \mathbb{F}_2^k$, $\mathbf{A} = (A_1,\ldots,A_k) \overset{\$}{\leftarrow} (\mathbb{F}_2^{n \times n})^k$, $E \overset{\$}{\leftarrow} \mathbb{F}_2^{n \times n}$ with $\mathrm{rank}(E) \leq r$
2. Set $\mathrm{sk} = s$ and $\mathrm{pk} = (\mathbf{A},\ Y = \mathbf{A}(s) + E)$

> $\mathbf{A}(s)$ denotes the linear combination $\sum_i s_i A_i$ over $\mathbb{F}_2$.

**Encrypt** bit $x \in \{0, 1\}$:
- If $x = 0$: sample $R \overset{\$}{\leftarrow} \mathbb{F}_2^{n \times n}$ with $\mathrm{rank}(R) \leq r$, output $\mathrm{ct} = \langle R,\ (A_1,\ldots,A_k, Y) \rangle_t$
- If $x = 1$: output $\mathrm{ct} = (V_1,\ldots,V_{k+1})$ where each $V_i \overset{\$}{\leftarrow} \mathbb{F}_2^{t \times t}$

**Decrypt** $\mathrm{ct} = (C_1,\ldots,C_k,C_{k+1})$ with secret key $s$:
1. Compute $M = C_{k+1} - \sum_{i=1}^k s_i \cdot C_i$ over $\mathbb{F}_2$
2. If $\mathrm{rank}(M) < t - \log^{2/3} n$, output $0$; otherwise output $1$

**Blockwise inner product** $\langle A, B \rangle_t$: a $t \times t$ matrix whose $(i,j)$-th entry is the Frobenius inner product of the $(i,j)$-th $(n/t) \times (n/t)$ blocks of $A$ and $B$.

**Why it works:**
- When $x=0$: $M = \langle R, E \rangle_t$, so $\mathrm{rank}(M) \leq r^2 < t - \log n$ by Claim 3.15 — decrypts to $0$ correctly.
- When $x=1$: $M$ is a random $t \times t$ matrix, which has full rank with overwhelming probability — decrypts to $1$ correctly.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/minrank-pke
cd minrank-pke
pip install numpy
```

**Requirements:** Python >= 3.11, NumPy.

---

## Quick Start

```python
from minrank_pke import MinRankPKE, Params

params = Params.toy()
scheme = MinRankPKE(params)

pk, sk = scheme.keygen()

ct0 = scheme.encrypt(pk, 0)
ct1 = scheme.encrypt(pk, 1)

print(scheme.decrypt(sk, ct0, params.k))  # 0
print(scheme.decrypt(sk, ct1, params.k))  # 1
```

---

## Parameter Sets

| Name | n | t | k | r | Notes |
|------|---|---|---|---|-------|
| `Params.toy()` | 64 | 16 | 4 | 1 | Functional tests only, no security |
| `Params.small()` | 128 | 32 | 5 | 2 | Demo only, no security |
| `Params.medium()` | 256 | 32 | 10 | 2 | Heuristic security, low — increase $n$ for real use |

For meaningful security follow Section 4 of the paper. Setting 1 uses $t = \Theta(n^{1/2})$, $k = \Theta(n)$, $r = \Theta(n^{1/4})$, giving best known attack cost around $2^{O(n^{1/4})}$.

---

## Running Tests

```bash
pytest tests/ -v
```

All 27 tests pass, covering GF(2) arithmetic, parameter validation, encrypt/decrypt correctness, and the core algebraic identity `M = <R, E>_t`.

---

## Project Structure

```
minrank-pke/
├── minrank_pke/
│   ├── __init__.py
│   ├── gf2.py       # GF(2) matrix arithmetic and utilities
│   └── scheme.py    # KeyGen, Enc, Dec (Figure 1 of the paper)
├── tests/
│   └── test_scheme.py
├── README.md
└── pyproject.toml
```

---

## Limitations

- Encrypts a **single bit** per ciphertext, as in the paper's construction.
- Research implementation — not audited, not constant-time, not production-ready.
- For real security, `n` needs to be substantially larger than the demo parameter sets.

---

## Citation

```bibtex
@misc{cryptoeprint:2025/1833,
  author = {Rohit Chatterjee and Changrui Mu and Prashant Nalini Vasudevan},
  title  = {Public-Key Encryption from the {MinRank} Problem},
  howpublished = {Cryptology ePrint Archive, Paper 2025/1833},
  year   = {2025},
  url    = {https://eprint.iacr.org/2025/1833}
}
```
