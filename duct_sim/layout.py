"""Load a track layout drawn in the planner and spawn ducts along it.

The planner exports metres in the same arena-centred frame the builder uses,
so nothing needs converting on the way in:

    {"units":"m",
     "arena":{"width":30,"depth":20},
     "duct":{"diameter":0.40,"hoop_spacing":0.05},
     "runs":[{"name":"outer barrier","points":[[x,y],[x,y],...]}]}

A run is a CENTRELINE, not a hoop list. The points are however many the person
drew -- roughly one every 0.6 m -- so the loader resamples the polyline at the
hoop spacing and places a hoop at each station, turned to face along the path.
That is what makes a drawn curve come out as a duct that follows it instead of
a straight duct dropped near it.
"""

from __future__ import annotations

import json
import math

import numpy as np


def load(path):
    with open(path) as fh:
        doc = json.load(fh)
    if "runs" not in doc:
        raise ValueError(f"{path}: no 'runs' key -- is this a planner export?")
    return doc


def resample(points, spacing):
    """Even stations every `spacing` metres along the polyline.

    Returns [(x, y, heading_deg), ...]. The heading is the local tangent, so a
    hoop placed there stands square to the path.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return [(float(p[0][0]), float(p[0][1]), 0.0)] if len(p) else []

    seg = np.diff(p, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 1e-9
    if not keep.any():
        return [(float(p[0][0]), float(p[0][1]), 0.0)]
    p = np.vstack([p[0], p[1:][keep]])
    seg = np.diff(p, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    n = max(2, int(round(total / spacing)) + 1)
    want = np.linspace(0.0, total, n)

    out = []
    for w in want:
        i = int(np.searchsorted(cum, w, side="right") - 1)
        i = min(max(i, 0), len(seg) - 1)
        t = (w - cum[i]) / seg_len[i] if seg_len[i] > 1e-12 else 0.0
        x = p[i, 0] + seg[i, 0] * t
        y = p[i, 1] + seg[i, 1] * t
        heading = math.degrees(math.atan2(seg[i, 1], seg[i, 0]))
        out.append((float(x), float(y), float(heading)))
    return out


def describe(doc):
    lines = []
    sp = float(doc.get("duct", {}).get("hoop_spacing", 0.05))
    total_hoops = 0
    for r in doc["runs"]:
        pts = r.get("points", [])
        st = resample(pts, sp)
        total_hoops += len(st)
        length = 0.0
        for a, b in zip(pts, pts[1:]):
            length += math.hypot(b[0] - a[0], b[1] - a[1])
        lines.append(f"  {r.get('name', 'run'):<18} {length:7.2f} m  "
                     f"{len(pts):4d} drawn pts -> {len(st):5d} hoops")
    lines.append(f"  {'TOTAL':<18} {'':>7}    {'':>4}              {total_hoops:5d} hoops")
    return "\n".join(lines)


def anchor_stations(points, anchors, spacing):
    """Map anchored NODE indices from the planner onto hoop station indices.

    The planner anchors a node of the drawn chain; the builder works in hoops,
    which are a finer resampling of the same polyline. Both are parameterised
    by arc length, so converting is just "how far along was that node".
    """
    if not anchors:
        return []
    import numpy as _np

    p = _np.asarray(points, dtype=_np.float64)
    if len(p) < 2:
        return []
    seg = _np.hypot(*(_np.diff(p, axis=0).T))
    cum = _np.concatenate([[0.0], _np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return []
    n_station = max(2, int(round(total / spacing)) + 1)
    out = []
    for a in anchors:
        a = int(a)
        if 0 <= a < len(cum):
            frac = cum[a] / total
            out.append(int(round(frac * (n_station - 1))))
    return sorted(set(out))
