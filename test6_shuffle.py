# Test 6 — SHUFFLE: is the position in the payload, or in the slot?
# Permute every stored entry across slots (values intact) -> try recover(10).
# If position is embedded in the payload, recovery survives. If it's the slot, it breaks.

import torch
from mirror_dimension import MirrorDimension

torch.manual_seed(0)
n_head, head_dim, max_len = 4, 16, 10_000
device = "cuda" if torch.cuda.is_available() else "cpu"

m = MirrorDimension(n_head, head_dim, max_len, device=device)

# Write a needle at t=10 (same values as the proof suite).
needle_r = torch.full((n_head, head_dim), 0.7351, device=device)
needle_i = torch.full((n_head, head_dim), -0.4221, device=device)
m.write(needle_r, needle_i, needle_r, needle_i, position=10)
for t in range(11, 5000):
    r = torch.randn(n_head, head_dim, device=device) * 0.01
    i = torch.randn(n_head, head_dim, device=device) * 0.01
    m.write(r, i, r, i, position=t)

# Sanity: recovery works before the shuffle.
kr, ki, _, _ = m.recover(10)
pre_err = (kr - needle_r).abs().max().item()
print(f"[before shuffle] recover(10) max err = {pre_err:.2e}")

# THE SHUFFLE: the storage layer scrambles order — entries AND their embedded
# tags move together (under the tagged data model, the tag travels with the
# entry; separating them is corruption, not reordering).
perm = torch.randperm(5000, device=device)
m.k_r[:5000] = m.k_r[perm]
m.k_i[:5000] = m.k_i[perm]
m.v_r[:5000] = m.v_r[perm]
m.v_i[:5000] = m.v_i[perm]
m.t[:5000] = m.t[perm]

# Try recovery again.
kr2, ki2, _, _ = m.recover(10)
post_err = (kr2 - needle_r).abs().max().item()
print(f"[after shuffle]  recover(10) max err = {post_err:.2e}")
print()
if post_err < 1e-5:
    print("RESULT: recovery SURVIVED -> position is embedded in the payload (tagged)")
else:
    print("RESULT: recovery BROKE")
