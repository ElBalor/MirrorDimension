# Mirror Dimension — Persistent KV Cache (Six-Proof Build)

The MirrorDimension object from the Mirror Dimension paper (Heylel Yaka, v0.2),
upgraded with **embedded position tags** and a **sixth proof: shuffle-invariance**.

## The discovery (Sept 21, 2026 session)

The original five tests are real and reproduce bit-for-bit (Test 1 identical to
the paper on an RTX GPU). But a sixth attack — permuting every stored entry
across slots, values intact — exposed that the position lived in the **array
slot**, not the payload:

```
[before shuffle] recover(10) max err = 5.96e-08   (exact)
[after shuffle]  recover(10) max err = 7.62e-01   (garbage — address destroyed)
```

The fix: every entry now carries its own position tag (`t`, one long per slot).
`recover()` finds entries **by tag**, and takes the un-rotation angle from the
entry itself. The slot is parking, not addressing. Same attack, after the fix:

```
[before shuffle] recover(10) max err = 5.96e-08
[after shuffle]  recover(10) max err = 5.96e-08   (SURVIVED)
```

The mirror is now genuinely self-describing: lossless AND order-independent.

## Files

| File | What it is |
|---|---|
| `mirror_dimension.py` | The tagged MirrorDimension + all six tests (`python mirror_dimension.py`) |
| `test6_shuffle.py` | The shuffle attack, standalone — the before/after evidence |
| `mirror_test.pt` | Persistence-test artifact (written by Test 5) |

## Reproduce

```bash
pip install torch          # CUDA optional; CPU works
python mirror_dimension.py # runs all six tests
python test6_shuffle.py    # runs the shuffle attack standalone
```

Measured results this machine (RTX GPU, torch 2.8.0+cu126, Windows):

| Test | Property | Result |
|---|---|---|
| 1 | Isometry (norm preservation) | diff **0.00e+00** |
| 2 | Conjugate recovery | **0.00e+00** both parts |
| 3 | Cross-distance (4990 writes) | **5.96e-08** = float32 eps |
| 4 | Relative-offset invariance | diff **1.00e-05** |
| 5 | Save/load persistence | **5.96e-08** |
| 6 | **Shuffle-invariance (new)** | **5.96e-08 after full permutation** |

Legacy saves (no tags) load cleanly — slots become tags automatically.

## Cost of the upgrade

One int64 per entry (~4 bytes vs 2×head_dim float32s) plus a tag lookup on
read. Zero change to the rotation, the isometry, or the attention math.
