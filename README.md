# minrank-pke

Python implementation of [Public-Key Encryption from the MinRank Problem](https://eprint.iacr.org/2025/1833) (Chatterjee, Mu, Vasudevan 2025).

## Scheme
 
**Parameters:** $n$, $k = k(n)$, $r = r(n)$, $t = t(n)$ where $t \mid n$, $r^2 < t - \log n$, and $(n/t)^2 - 2k = \omega(\log n)$.
 
**KeyGen$(1^n)$:**
1. Sample $s \xleftarrow{\$} \mathbb{F}_2^k$; $A \xleftarrow{\$} (\mathbb{F}_2^{n \times n})^k$; $E \xleftarrow{\$} \mathbb{F}_2^{n \times n}$ s.t. $\mathrm{rank}(E) \le r$.
2. Set $\mathrm{sk} = s$ and $\mathrm{pk} = (A,\, A(s) + E)$.
**Enc$(\mathrm{pk},\, x \in \{0,1\})$:**
1. Parse $\mathrm{pk} = (A'_1, A'_2, \ldots, A'_{k+1}) = A'$.
2. If $x = 1$, sample $k+1$ random matrices $(V_1, \ldots, V_{k+1}) \xleftarrow{\$} (\mathbb{F}_2^{t \times t})^{k+1}$; and set these to be the ciphertext $\mathrm{ct}$.
3. Else if $x = 0$, sample a random matrix $R \xleftarrow{\$} \mathbb{F}_2^{n \times n}$ under the constraint that $\mathrm{rank}(R) \le r$; set the ciphertext to be $\mathrm{ct} = \langle R, A' \rangle_t$.
4. The encryption algorithm outputs $\mathrm{ct}$.
**Dec$(\mathrm{sk} = s,\, \mathrm{ct})$:**
1. Parse $\mathrm{ct} = (C_1, C_2, \cdots, C_k, C_{k+1})$.
2. Set $M = C_{k+1} - \sum_{i \in [k]} s_i \cdot C_i$.
3. If $\mathrm{rank}(M) < t - \log^{2/3} n$ output $0$; otherwise output $1$.

## Usage

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

## Parameters

| Name | n | t | k | r | Notes |
|------|---|---|---|---|-------|
| `Params.toy()` | 64 | 16 | 4 | 1 | Functional tests only, no security |
| `Params.small()` | 128 | 32 | 5 | 2 | Demo only, no security |
| `Params.medium()` | 256 | 32 | 10 | 2 | Heuristic security, low — increase $n$ for real use |

For meaningful security follow Section 4 of the paper. Setting 1 uses $t = \Theta(n^{1/2})$, $k = \Theta(n)$, $r = \Theta(n^{1/4})$, giving best known attack cost around $2^{O(n^{1/4})}$.

---

## Tests

```bash
pytest tests/ -v
```

All 27 tests pass, covering GF(2) arithmetic, parameter validation, encrypt/decrypt correctness, and the core algebraic identity `M = <R, E>_t`.
