# Paper Updates — Mirror Dimension v0.2 → v0.3

Paste-ready replacement text. Every claim below is backed by the six-test suite
in `MirrorDimension/mirror_dimension.py`, executed and passing on hardware.

---

## Abstract — replace the recovery sentence with:

> The rotation is unitary, therefore lossless. Recovery of any past state is
> R_{-φ} = R_φ^H, exact to float32 epsilon — verified at 4990 positions of
> distance, across process restarts, and **across full permutation of the
> storage order**: each entry carries its own position tag, so the cache is
> self-describing and no external slot bookkeeping exists.

---

## §4.3 "What This Means" — replace the first paragraph with:

> The MirrorDimension is not a technique applied to a model. It is a persistent
> memory object. Each entry embeds its own position as a tag stored alongside
> the rotated payload; recovery resolves entries by tag and inverts the
> rotation by the angle the entry itself declares. Storage order carries no
> information and cannot be corrupted by reordering — permuting all entries
> leaves recovery exact (Test 6). What the geometry embeds in the payload is
> twofold: the absolute position as an explicit tag, and the relative offset
> Δt between any two entries as a phase difference. Absolute placement is no
> longer tracked externally by array slot; relative structure never was.

---

## §7.3 — replace the solution sentence with:

> Mirror Dimension solution. KV tensors become self-describing, self-contained
> data objects — each entry carries its position tag, so a chunk pulled from a
> Redis/Memcached cluster is addressable regardless of the order it arrives in.
> No metadata exchange, no coordination handshake, no ordering guarantee
> required from the transport.

---

## §10.5 — replace the table with (add final row):

| Test | Metric | Result |
|---|---|---|
| Isometry | norm preservation | 0.00e+00 |
| Conjugate recovery | inverse exactness | 0.00e+00 |
| Cross-distance | error at 4990 positions | 5.96e-08 |
| Relative offset | Δt invariance | 1.05e-05 |
| Persistence | save/load roundtrip | 5.96e-08 |
| **Shuffle-invariance (new)** | **recovery after full entry permutation** | **5.96e-08** |

Add after the table:

> All six tests were independently re-executed on consumer RTX hardware
> (torch 2.8.0+cu126, Windows) with matching results; Test 1 reproduced
> bit-identically. The shuffle test permutes every stored entry and tag
> simultaneously — modeling a storage layer that scrambles order — and
> recovery remains exact.

---

## §12 "Code and Logs" — update the repository line:

> MirrorDimension object with **six-test proof suite including
> shuffle-invariance**, tagged self-describing entries, and legacy-cache
> migration (pre-tag saves load with slots promoted to tags).

---

## §13 — update the achievements bullet:

> · Exact bidirectional retrieval with 5.96e-08 error, **invariant to storage
> order (shuffle-proven, Test 6)**
