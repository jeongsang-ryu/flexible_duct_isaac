"""Generate a tiling plain-weave normal map for the duct fabric.

Colour alone cannot make a surface read as cloth: what the eye uses is the way
a weave breaks a highlight into hundreds of small ones. That is surface normal,
not albedo, so this writes a normal map rather than a diffuse texture.

Plain weave: warp runs along v, weft along u, and which one is on top flips
every cell. Each thread is a rounded cylinder, so its height across its own
width is a half-cosine; the thread underneath is pushed down by the one on top.

Output tiles seamlessly -- the pattern period divides the image size exactly.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def weave_height(size=512, threads=32, depth=1.0):
    """Height field in [0, 1]. `threads` = thread pairs across the tile."""
    p = size // threads                      # pixels per thread; divides evenly
    y, x = np.mgrid[0:size, 0:size]
    # position within the thread, 0..1
    u = (x % p) / p
    v = (y % p) / p
    # rounded thread cross-section
    warp = np.sin(np.pi * u) ** 0.7          # runs vertically, varies across x
    weft = np.sin(np.pi * v) ** 0.7          # runs horizontally, varies across y
    # which thread is on top in this cell
    top_is_warp = ((x // p) + (y // p)) % 2 == 0
    h = np.where(top_is_warp, warp * 1.0 + weft * 0.35,
                              weft * 1.0 + warp * 0.35)
    h -= h.min()
    h /= h.max()
    return h * depth


def to_normal_map(h, strength=2.5):
    """Tangent-space normals from a height field, wrapping at the edges."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    n = np.dstack([-dx * strength, -dy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def to_roughness(h, lo=0.85, hi=1.0):
    """Thread crowns catch marginally more light than the valleys between."""
    r = hi - (hi - lo) * h
    return (np.clip(r, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--strength", type=float, default=2.5)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "duct_sim", "textures"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    h = weave_height(a.size, a.threads)
    Image.fromarray(to_normal_map(h, a.strength)).save(
        os.path.join(a.out, "weave_normal.png"))
    Image.fromarray(to_roughness(h), mode="L").save(
        os.path.join(a.out, "weave_rough.png"))
    print(f"wrote weave_normal.png and weave_rough.png "
          f"({a.size}px, {a.threads} thread pairs) to {a.out}")


if __name__ == "__main__":
    main()
