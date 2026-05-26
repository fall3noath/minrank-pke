# minrank-pke

Python implementation of [Public-Key Encryption from the MinRank Problem](https://eprint.iacr.org/2025/1833) (Chatterjee, Mu, Vasudevan 2025).

## Scheme

**Parameters:** $n$, $k$, $r$, $t$ where $t \mid n$, $r^2 < t - \log n$

**KeyGen:**
- $s \gets \{0,1\}^k$, $A_i \gets \mathbb{F}_2^{n \times n}$, $E \gets \mathbb{F}_2^{n \times n}$ with $\mathrm{rank}(E) \le r$
- $\mathrm{sk} = s$, $\mathrm{pk} = (A_1,\ldots,A_k,\; Y = \sum s_i A_i + E)$

**Encrypt** $x \in \{0,1\}$:
- If $x=0$: sample $R$ ($\mathrm{rank}(R) \le r$), output $\langle R, (A_1,\ldots,A_k,Y) \rangle_t$
- If $x=1$: output $k+1$ random $t \times t$ matrices

**Decrypt** $(C_1,\ldots,C_{k+1})$:
- $M = C_{k+1} - \sum s_i C_i$
- If $\mathrm{rank}(M) < t - \log^{2/3} n$ output $0$, else $1$

## Usage

```python
from minrank_pke import MinRankPKE, Params

scheme = MinRankPKE(Params.toy())
pk, sk = scheme.keygen()
ct = scheme.encrypt(pk, 0)
print(scheme.decrypt(sk, ct))  # 0
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
