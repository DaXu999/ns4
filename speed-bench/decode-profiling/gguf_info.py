#!/usr/bin/env python3
"""Minimal GGUF reader: dump metadata + per-tensor sizes to compute the
active-weight byte budget of one decode token."""
import struct, sys, collections

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"

# ggml type -> (block size in elements, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}

f = open(F, "rb")
magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
assert magic == b"GGUF", magic

def rd_str():
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")

def rd_val(t):
    if t == 0: return struct.unpack("<B", f.read(1))[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return struct.unpack("<?", f.read(1))[0]
    if t == 8: return rd_str()
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4)); (n,) = struct.unpack("<Q", f.read(8))
        FIXED = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if et in FIXED:                      # skip the bulk, keep a preview
            head = [rd_val(et) for _ in range(min(n, 8))]
            f.seek(FIXED[et] * (n - min(n, 8)), 1)
            return head + (["...", f"n={n}"] if n > 8 else [])
        out = []                             # strings must be walked to advance
        for i in range(n):
            v = rd_str()
            if i < 4: out.append(v)
        return out + ([f"...n={n}"] if n > 4 else [])
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(t)

meta = {}
for _ in range(n_kv):
    k = rd_str(); (t,) = struct.unpack("<I", f.read(4)); meta[k] = rd_val(t)

print(f"tensors={n_tensors} kv={n_kv}")
for k in sorted(meta):
    if "tokenizer" in k: continue
    print(f"  {k} = {meta[k]}")

tensors = []
for _ in range(n_tensors):
    name = rd_str()
    (nd,) = struct.unpack("<I", f.read(4))
    dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
    (tt,) = struct.unpack("<I", f.read(4))
    (off,) = struct.unpack("<Q", f.read(8))
    n_elem = 1
    for d in dims: n_elem *= d
    tname, blk, bpb = TYPES.get(tt, (f"T{tt}", 1, 4))
    nbytes = n_elem // blk * bpb if blk > 1 else n_elem * bpb
    tensors.append((name, dims, tname, nbytes))

print(f"\n=== per-layer tensor inventory (layer 3 as representative) ===")
per_kind = collections.OrderedDict()
total = 0
for name, dims, tname, nb in tensors:
    total += nb
    if ".3." in name or name.startswith("blk.3."):
        per_kind[name] = (dims, tname, nb)
for n, (d, t, nb) in per_kind.items():
    print(f"  {n:<44} {str(d):<24} {t:<9} {nb/1e6:9.2f} MB")
print(f"\ntotal file tensor bytes: {total/2**30:.2f} GiB")
