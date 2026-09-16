"""Build the duct straight, then bend it into the layout by moving the hoops.

Why this exists: spawn_duct_path places the hoops along the drawn polyline and
sweeps the fabric around them, so the duct is BORN curved. The corners
therefore carry zero strain -- the duct was not bent, it was moulded in a bent
shape -- and the creasing there is whatever the mesh generator happened to
produce rather than what the material would do. It looks wrong because it is
wrong.

A real duct starts straight and is pushed into place. Doing the same here means
the corner folds come out of the solver: the outside of a bend stretches, the
inside gathers, and the fabric finds where to crease on its own.

The hoops are rigid bodies, so they can be driven to the target pose
kinematically while the fabric simply follows and deforms. The one parameter
that matters is how many steps the move takes. Too fast and the fabric cannot
keep up -- it gets dragged through itself and explodes; too slow and the build
takes minutes. `steps` is exposed so it can be measured rather than guessed.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf


def straight_stations(stations, z=None):
    """The same hoops, laid along +x with the same arc spacing, heading 0.

    Arc length is preserved so the fabric between hoops is neither stretched
    nor slack at the start -- only the path's curvature changes during the
    bend, which is the whole point.
    """
    pts = [(float(s[0]), float(s[1])) for s in stations]
    arc = [0.0]
    for a, b in zip(pts, pts[1:]):
        arc.append(arc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    half = arc[-1] * 0.5
    return [(cx + (a - half), cy, 0.0) for a in arc]


class Bender:
    """Drives one duct's hoops from their straight start to the layout pose.

    Call `update()` once per simulation step. Returns True while still moving.
    """

    def __init__(self, stage, ring_paths, targets, steps=600, hold=120):
        from isaacsim.core.prims import RigidPrim
        self.stage = stage
        self.ring_paths = list(ring_paths)
        self.steps = int(steps)
        self.hold = int(hold)
        self.i = 0
        self._view = None
        self._RigidPrim = RigidPrim

        # start poses, read from the stage as authored
        from pxr import UsdGeom
        self.start_p, self.start_q = [], []
        for p in self.ring_paths:
            prim = stage.GetPrimAtPath(p)
            t = q = None
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    t = op.Get()
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    q = op.Get()
            self.start_p.append([float(t[0]), float(t[1]), float(t[2])])
            self.start_q.append(q)

        self.goal_p, self.goal_q = [], []
        for (x, y, h), sp in zip(targets, self.start_p):
            self.goal_p.append([float(x), float(y), sp[2]])
            r = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
                 * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(h)))
            self.goal_q.append(Gf.Quatf(r.GetQuat()))

        self.start_p = np.array(self.start_p, dtype=np.float64)
        self.goal_p = np.array(self.goal_p, dtype=np.float64)

    def _ensure_view(self):
        if self._view is None:
            # built AFTER physics has stepped, or it reports authored poses
            # forever and every read is poisoned
            self._view = self._RigidPrim(self.ring_paths)
            self._view.initialize()
        return self._view

    def update(self):
        if self.i >= self.steps + self.hold:
            return False
        self.i += 1
        if self.i > self.steps:
            return True                      # holding, let the fabric settle

        # smoothstep, so the hoops start and stop gently. A linear ramp steps
        # the whole duct at once on frame 1 and that impulse is what tears the
        # fabric.
        u = self.i / self.steps
        s = u * u * (3.0 - 2.0 * u)

        pos = self.start_p + (self.goal_p - self.start_p) * s
        quats = []
        for q0, q1 in zip(self.start_q, self.goal_q):
            q = Gf.Slerp(float(s), Gf.Quatf(q0), Gf.Quatf(q1))
            quats.append([q.GetReal(), *q.GetImaginary()])

        view = self._ensure_view()
        try:
            from duct_sim.mouse_drag import _to_backend
            view.set_world_poses(_to_backend(pos.astype(np.float32)),
                                 _to_backend(np.array(quats, dtype=np.float32)))
        except Exception as exc:
            print(f"[bend] set_world_poses failed: {exc}", flush=True)
            self.i = self.steps + self.hold
            return False
        return True
