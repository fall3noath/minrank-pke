# MinRank PKE

A Python implementation of the public-key encryption scheme introduced in:

> **Public-Key Encryption from the MinRank Problem**  
> Rohit Chatterjee, Changrui Mu, Prashant Nalini Vasudevan  
> IACR ePrint 2025/1833 · https://eprint.iacr.org/2025/1833

This is the **first public implementation** of this scheme.

---

## What is this?

The MinRank problem asks: given $k$ matrices $A_1,\ldots,A_k$ over $\mathbb{F}_2$ and a target matrix $Y$, find a binary vector $s$ such that $Y - \sum_i s_i A_i$ has rank $\leq r$.

This scheme builds a PKE from the average-case hardness of that problem, using:
- A **blockwise inner product** (Definition 3.7 in the paper) as the core algebraic tool
- A **duality reduction** showing the dual MinRank problem is as hard as the primal
- An Alekhnovich-style construction adapted to the rank metric

The algebra lives in the ring of $n \times n$ matrices over $\mathbb{F}_2$.

---

## Scheme (Figure 1 of the paper)

**Parameters:** $n$, $k$, $r$, $t$ where $t \mid n$, $r^2 < t - \log n$, $(n/t)^2 - 2k - 1 = \omega(\log n)$

**KeyGen:**
1. Sample $s \xleftarrow{\$} \mathbb{F}_2^k$, $\mathbf{A} = (A_1,\ldots,A_k) \xleftarrow{\$} (\mathbb{F}_2^{n \times n})^k$, $E \xleftarrow{\$} \mathbb{F}_2^{n \times n}$ with $\text{rank}(E) \leq r$
2. $\text{sk} = s$, $\text{pk} = (\mathbf{A},\ Y = \mathbf{A}(s) + E)$

**Encrypt** bit $x \in \{0,1\}$:
- If $x = 0$: sample $R$ with $\text{rank}(R) \leq r$, output $\text{ct} = \langle R, (\mathbf{A}, Y) \rangle_t$
- If $x = 1$: output $\text{ct} = (V_1,\ldots,V_{k+1})$ with each $V_i \xleftarrow{\$} \mathbb{F}_2^{t \times t}$

**Decrypt** $\text{ct} = (C_1,\ldots,C_k,C_{k+1})$:
1. $M = C_{k+1} - \sum_{i=1}^k s_i \cdot C_i$
2. Output $0$ if $\text{rank}(M) < t - \log^{2/3} n$, else output $1$

The blockwise inner product $\langle A, B \rangle_t$ is a $t \times t$ matrix whose $(i,j)$-th entry is the Frobenius inner product of the $(i,j)$-th $(n/t) \times (n/t)$ blocks of $A$ and $B$.

**Correctness:** When $x=0$, $M = \langle R, E \rangle_t$, so $\text{rank}(M) \leq r^2 < t - \log n$ by Claim 3.15. When $x=1$, $M$ is a random $t \times t$ matrix, which has full rank with overwhelming probability.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/minrank-pke
cd minrank-pke
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11, NumPy.

---

## Quick Start

```python
from minrank_pke import MinRankPKE, Params

# Use the toy parameter set (small, for testing only)
params = Params.toy()
scheme = MinRankPKE(params)

# Key generation
pk, sk = scheme.keygen()

# Encrypt a bit
ct0 = scheme.encrypt(pk, 0)
ct1 = scheme.encrypt(pk, 1)

# Decrypt
print(scheme.decrypt(sk, ct0, params.k))  # 0
print(scheme.decrypt(sk, ct1, params.k))  # 1
```

---

## Parameter Sets

| Name | n | t | k | r | Notes |
|------|---|---|---|---|-------|
| `Params.toy()` | 16 | 4 | 4 | 2 | Functional tests only, **no security** |
| `Params.small()` | 32 | 8 | 6 | 2 | Demo only, **no security** |
| `Params.medium()` | 256 | 16 | 64 | 4 | Heuristic security ~$2^4$ against known attacks |

For real security, follow Section 4 of the paper:
- **Setting 1:** $t = \Theta(n^{1/2})$, $k = \Theta(n)$, $r = \Theta(n^{1/4})$ → best attack cost ~$2^{O(n^{1/4})}$
- **Setting 2:** $t = \Theta\!\left(\frac{n}{\log n}\right)$ etc. → maximises attack cost for given $n$

---

## Running Tests

```bash
pytest tests/ -v
```

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

- Encrypts a **single bit** per ciphertext (as in the paper's construction).
- This is a **research implementation** — not audited, not constant-time, not production-ready.
- The `medium` parameter set has very low security; large $n$ is needed for real security.

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
