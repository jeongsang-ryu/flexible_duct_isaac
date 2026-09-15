"""Visual fabric for approach A: a sleeve redrawn each frame from ring poses.

Approach A does not simulate the cloth, so the fabric has to be *driven*. Each
frame this reads the current world transform of every ring and rewrites the
sleeve mesh's point array so the sleeve passes through all of them.

WHY REWRITE POINTS RATHER THAN USE USD SKINNING. UsdSkel would be the
"correct" authoring answer, but it binds at stage-load time and is awkward to
drive from a per-frame physics readback; rewriting a few thousand Vec3f is
about 0.1 ms here and keeps the data flow obvious: rings move -> points move.
The cost only matters if the ring count grows by an order of magnitude.

The sleeve bulges slightly between rings (`slack`) because a real flex duct's
fabric is not pulled taut between hoops -- without it the duct reads as a
faceted cone rather than fabric.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from .spec import DuctSpec


def _orthonormalise(m: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to `m`, via SVD (Kabsch/polar decomposition).

    Used because interpolating between two ring orientations must give a
    ROTATION; anything with a scale component silently resizes the sleeve.
    """
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:          # reflection -> flip the least-significant axis
        u[:, -1] *= -1
        r = u @ vt
    return r


class DuctSkin:
    def __init__(
        self,
        stage: Usd.Stage,
        mesh_path: str,
        ring_paths: list[str],
        spec: DuctSpec,
        n_circ: int = 32,
        rings_per_gap: int = 3,
        slack: float = 0.06,
        radius_offset: float | None = None,
    ):
        self.stage = stage
        self.ring_paths = ring_paths
        self.spec = spec
        self.n_circ = n_circ
        self.rings_per_gap = rings_per_gap
        self.slack = slack
        # The sleeve is drawn just OUTSIDE the hoops. Drawn at exactly
        # spec.radius the hoops protrude through it by their tube radius and
        # the duct reads as a bare coil spring rather than a fabric hose --
        # and because a smooth tube looks identical when spun about its own
        # axis, only the protruding hoops appeared to rotate, which is what
        # made a rolling duct look like hoops spinning inside a static sleeve.
        self.radius_offset = (
            spec.ring_thickness * 1.6 if radius_offset is None else radius_offset
        )

        n_ring = len(ring_paths)
        # one loop of vertices per ring, plus `rings_per_gap` loops inside each
        # gap so the fabric can bulge instead of being a straight chamfer
        self.n_loops = (n_ring - 1) * (rings_per_gap + 1) + 1

        self._unit = np.array(
            [[math.cos(2 * math.pi * i / n_circ), math.sin(2 * math.pi * i / n_circ), 0.0]
             for i in range(n_circ)],
            dtype=np.float64,
        )

        counts, idx = [], []
        for j in range(self.n_loops - 1):
            for i in range(n_circ):
                i2 = (i + 1) % n_circ
                a, b = j * n_circ + i, j * n_circ + i2
                c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
                counts.append(4)
                idx.extend([a, b, c, d])

        self.mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        self.mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        self.mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        self.mesh.CreateDoubleSidedAttr(True)
        self._points_attr = self.mesh.CreatePointsAttr()
        self._xf_cache = UsdGeom.XformCache()
        self.update()

    def _ring_frames(self):
        """World centre + basis of every ring, read fresh from the stage."""
        self._xf_cache.Clear()
        out = []
        for p in self.ring_paths:
            m = self._xf_cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(p))
            t = m.ExtractTranslation()
            r = m.ExtractRotationMatrix()
            out.append((
                np.array([t[0], t[1], t[2]]),
                np.array([[r[0][0], r[0][1], r[0][2]],
                          [r[1][0], r[1][1], r[1][2]],
                          [r[2][0], r[2][1], r[2][2]]]),
            ))
        return out

    def update(self) -> None:
        frames = self._ring_frames()
        R = self.spec.radius + self.radius_offset
        pts = np.empty((self.n_loops, self.n_circ, 3), dtype=np.float64)

        loop = 0
        for k in range(len(frames) - 1):
            (c0, b0), (c1, b1) = frames[k], frames[k + 1]
            steps = self.rings_per_gap + 1
            for s in range(steps):
                t = s / steps
                c = c0 * (1 - t) + c1 * t
                # Blend the two ring bases -- but a LINEAR average of rotation
                # matrices is not a rotation: it contracts by cos(theta/2),
                # which shrinks the sleeve exactly where the duct bends most.
                # Measured: a 30 deg bend loses 6.8 mm of radius and a 45 deg
                # bend 15.2 mm, while the hoops stand 8 mm proud of the sleeve
                # -- so past ~30 deg the hoops emerge THROUGH the fabric, and
                # only on the inside of tight bends, which is exactly where it
                # was observed. Re-orthonormalising restores a true rotation.
                b = _orthonormalise(b0 * (1 - t) + b1 * t)
                # sin() bulge: zero exactly at each hoop (fabric is sewn there),
                # maximum midway between them
                bulge = 1.0 + self.slack * math.sin(math.pi * t)
                ring_local = self._unit * (R * bulge)
                pts[loop] = c + ring_local @ b.T
                loop += 1
        c, b = frames[-1]
        pts[loop] = c + (self._unit * R) @ b.T

        flat = pts.reshape(-1, 3).astype(np.float32)
        self._points_attr.Set(Vt.Vec3fArray.FromNumpy(flat))
