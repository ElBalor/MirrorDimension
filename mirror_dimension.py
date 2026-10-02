"""
Mirror Dimension — persistent KV cache with isometric rotation.
"""

import math
import torch


class MirrorDimension:
    def __init__(self, n_head, head_dim, max_len, device="cpu", dtype=torch.float32):
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_len = max_len
        self.device = device
        self.dtype = dtype

        self.k_r = torch.zeros(max_len, n_head, head_dim, device=device, dtype=dtype)
        self.k_i = torch.zeros(max_len, n_head, head_dim, device=device, dtype=dtype)
        self.v_r = torch.zeros(max_len, n_head, head_dim, device=device, dtype=dtype)
        self.v_i = torch.zeros(max_len, n_head, head_dim, device=device, dtype=dtype)
        # Position tag: the entry's own address, stored WITH the payload.
        # -1 = slot empty. This is what makes the cache self-describing:
        # entries remain addressable no matter what order they sit in.
        self.t = torch.full((max_len,), -1, dtype=torch.long, device=device)
        self.pos = 0

        freqs = 1.0 / (10000.0 ** (torch.arange(head_dim, dtype=dtype, device=device) / head_dim))
        self.freqs = freqs

    def _rotation(self, position):
        phi = position * self.freqs
        return torch.cos(phi), torch.sin(phi)

    def write(self, k_r, k_i, v_r, v_i, position=None):
        if position is None:
            position = self.pos
        cp, sp = self._rotation(position)
        self.k_r[position] = k_r * cp - k_i * sp
        self.k_i[position] = k_r * sp + k_i * cp
        self.v_r[position] = v_r * cp - v_i * sp
        self.v_i[position] = v_r * sp + v_i * cp
        self.t[position] = position
        self.pos = max(self.pos, position + 1)

    def write_batch(self, k_r, k_i, v_r, v_i, start_pos=None):
        if start_pos is None:
            start_pos = self.pos
        T = k_r.shape[0]
        for t in range(T):
            self.write(k_r[t], k_i[t], v_r[t], v_i[t], position=start_pos + t)

    def read(self, positions):
        return (self.k_r[positions], self.k_i[positions],
                self.v_r[positions], self.v_i[positions])

    def size(self):
        return self.pos

    def recover(self, position):
        """Recover by TAG lookup, not slot: finds the entry whose embedded
        address equals `position`, then inverts the rotation by the angle the
        entry itself declares. Slot order is irrelevant."""
        matches = (self.t[:self.pos] == position).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            raise KeyError(f"no entry tagged position {position}")
        s = matches[0]
        true_pos = int(self.t[s].item())          # angle comes from the entry
        cp, sp = self._rotation(true_pos)
        kr = self.k_r[s] * cp + self.k_i[s] * sp
        ki = -self.k_r[s] * sp + self.k_i[s] * cp
        vr = self.v_r[s] * cp + self.v_i[s] * sp
        vi = -self.v_r[s] * sp + self.v_i[s] * cp
        return kr, ki, vr, vi

    def state_dict(self):
        return {
            "k_r": self.k_r.cpu(), "k_i": self.k_i.cpu(),
            "v_r": self.v_r.cpu(), "v_i": self.v_i.cpu(),
            "t": self.t.cpu(),
            "pos": self.pos,
            "n_head": self.n_head, "head_dim": self.head_dim,
            "max_len": self.max_len,
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32):
        sd = torch.load(path, map_location=device, weights_only=False)
        obj = cls(sd["n_head"], sd["head_dim"], sd["max_len"],
                  device=device, dtype=dtype)
        obj.k_r = sd["k_r"].to(device, dtype)
        obj.k_i = sd["k_i"].to(device, dtype)
        obj.v_r = sd["v_r"].to(device, dtype)
        obj.v_i = sd["v_i"].to(device, dtype)
        if "t" in sd:
            obj.t = sd["t"].to(device)
        else:  # legacy save without tags: slots were the addresses
            obj.t = torch.arange(obj.max_len, dtype=torch.long, device=device)
        obj.pos = sd["pos"]
        return obj


def l2_norm(a, b):
    """L2 norm of a complex tensor stored as (real, imag)."""
    return torch.sqrt((a**2 + b**2).sum()).item()


if __name__ == "__main__":
    print("=" * 70)
    print("MIRROR DIMENSION — PROOF")
    print("=" * 70)

    torch.manual_seed(0)
    n_head, head_dim = 4, 16
    max_len = 10_000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    mirror = MirrorDimension(n_head, head_dim, max_len, device=device)

    # ── Test 1 ──
    print("[Test 1] Isometry — rotation preserves norm")
    k_r = torch.randn(n_head, head_dim, device=device)
    k_i = torch.randn(n_head, head_dim, device=device)
    v_r = torch.randn(n_head, head_dim, device=device)
    v_i = torch.randn(n_head, head_dim, device=device)

    orig_norm = l2_norm(k_r, k_i)
    mirror.write(k_r, k_i, v_r, v_i, position=0)
    stored_norm = l2_norm(mirror.k_r[0], mirror.k_i[0])
    print(f"  original |K| = {orig_norm:.10f}")
    print(f"  stored   |K| = {stored_norm:.10f}")
    print(f"  diff         = {abs(orig_norm - stored_norm):.2e}")
    assert abs(orig_norm - stored_norm) < 1e-5, "isometry violated"
    print("  ✓ lossless\n")

    # ── Test 2 ──
    print("[Test 2] Conjugate recovery — inverse is exact")
    rec_r, rec_i, _, _ = mirror.recover(0)
    err_r = (rec_r - k_r).abs().max().item()
    err_i = (rec_i - k_i).abs().max().item()
    print(f"  max |ΔK_real| = {err_r:.2e}")
    print(f"  max |ΔK_imag| = {err_i:.2e}")
    assert err_r < 1e-5 and err_i < 1e-5, "recovery not exact"
    print("  ✓ exact\n")

    # ── Test 3 ──
    print("[Test 3] Cross-distance — write at t=10, recover at t=5000")
    needle_r = torch.full((n_head, head_dim), 0.7351, device=device)
    needle_i = torch.full((n_head, head_dim), -0.4221, device=device)
    mirror.write(needle_r, needle_i, needle_r, needle_i, position=10)

    for t in range(11, 5000):
        r = torch.randn(n_head, head_dim, device=device) * 0.01
        i = torch.randn(n_head, head_dim, device=device) * 0.01
        mirror.write(r, i, r, i, position=t)

    rec_r, rec_i, _, _ = mirror.recover(10)
    err_r = (rec_r - needle_r).abs().max().item()
    err_i = (rec_i - needle_i).abs().max().item()
    print(f"  after 4990 writes")
    print(f"  max |ΔK_real| at pos 10 = {err_r:.2e}")
    print(f"  max |ΔK_imag| at pos 10 = {err_i:.2e}")
    assert err_r < 1e-5 and err_i < 1e-5, "long-distance recovery failed"
    print("  ✓ exact across 4990 positions\n")

    # ── Test 4 ──
    print("[Test 4] Relative offset — Q·K depends only on Δt")
    q_r = torch.randn(n_head, head_dim, device=device)
    q_i = torch.randn(n_head, head_dim, device=device)
    key_r = torch.randn(n_head, head_dim, device=device)
    key_i = torch.randn(n_head, head_dim, device=device)

    mirror2 = MirrorDimension(n_head, head_dim, max_len, device=device)
    mirror2.write(key_r, key_i, key_r, key_i, position=50)
    cp_q, sp_q = mirror2._rotation(100)
    q_rot_r = q_r * cp_q - q_i * sp_q
    q_rot_i = q_r * sp_q + q_i * cp_q
    S_a = (q_rot_r * mirror2.k_r[50] + q_rot_i * mirror2.k_i[50]).sum().item()

    mirror3 = MirrorDimension(n_head, head_dim, max_len, device=device)
    mirror3.write(key_r, key_i, key_r, key_i, position=500)
    cp_q2, sp_q2 = mirror3._rotation(550)
    q_rot_r2 = q_r * cp_q2 - q_i * sp_q2
    q_rot_i2 = q_r * sp_q2 + q_i * cp_q2
    S_b = (q_rot_r2 * mirror3.k_r[500] + q_rot_i2 * mirror3.k_i[500]).sum().item()

    print(f"  Δt = 50 in both cases")
    print(f"  Case A (q@100, k@50):  S = {S_a:.10f}")
    print(f"  Case B (q@550, k@500): S = {S_b:.10f}")
    print(f"  diff = {abs(S_a - S_b):.2e}")
    assert abs(S_a - S_b) < 1e-4, "relative offset not preserved"
    print("  ✓ attention depends only on distance\n")

    # ── Test 5 ──
    print("[Test 5] Persistence — save, reload, recover still exact")
    path = "mirror_test.pt"
    mirror.save(path)
    mirror_loaded = MirrorDimension.load(path, device=device)
    print(f"  loaded pos = {mirror_loaded.pos} (saved = {mirror.pos})")
    assert mirror_loaded.pos == mirror.pos
    rec_r2, rec_i2, _, _ = mirror_loaded.recover(10)
    err_r2 = (rec_r2 - needle_r).abs().max().item()
    err_i2 = (rec_i2 - needle_i).abs().max().item()
    print(f"  recovered pos 10 from reloaded cache")
    print(f"  max |ΔK_real| = {err_r2:.2e}")
    print(f"  max |ΔK_imag| = {err_i2:.2e}")
    assert err_r2 < 1e-5 and err_i2 < 1e-5
    print("  ✓ persists across process boundaries\n")

    # ── Test 6 ──
    print("[Test 6] Shuffle — entries self-identify, storage order is irrelevant")
    # Permute ALL stored entries AND their tags together (a storage layer that
    # scrambles order). Values untouched — only placement changes.
    n = mirror.pos
    perm = torch.randperm(n, device=device)
    mirror.k_r[:n] = mirror.k_r[perm]
    mirror.k_i[:n] = mirror.k_i[perm]
    mirror.v_r[:n] = mirror.v_r[perm]
    mirror.v_i[:n] = mirror.v_i[perm]
    mirror.t[:n] = mirror.t[perm]

    kr3, ki3, _, _ = mirror.recover(10)
    err_r3 = (kr3 - needle_r).abs().max().item()
    err_i3 = (ki3 - needle_i).abs().max().item()
    print(f"  after full shuffle of {n} entries")
    print(f"  max |ΔK_real| at pos 10 = {err_r3:.2e}")
    print(f"  max |ΔK_imag| at pos 10 = {err_i3:.2e}")
    assert err_r3 < 1e-5 and err_i3 < 1e-5, "shuffle broke recovery"
    print("  ✓ position lives in the entry — the mirror is self-describing\n")

    print("=" * 70)
    print("ALL 6 TESTS PASSED — Mirror Dimension works")
    print("=" * 70)
