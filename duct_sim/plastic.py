"""Make a bend PERMANENT: rewrite the cloth's rest shape to its current shape.

The duct is elastic. Push it into a curve and it springs back, because the
solver is always pulling the fabric toward a rest configuration that still
describes the straight tube it was built as. PhysX surface deformables have no
plasticity parameter -- there is no yield stress to set -- so "stays where I
bent it" has to be done by moving the target, not by adding a material.

Two things define that target (OmniPhysicsSurfaceDeformableSimAPI):

  restShapePoints / restTriVtxIndices
      the per-triangle planar rest shape: what resists STRETCH and SHEAR
  restAdjTriPairs / restBendAngles
      the rest dihedral angle across each shared edge: what resists BENDING,
      and therefore what actually pulls a curved duct straight again.
      restBendAnglesDefault = "flatDefault" means any pair NOT listed here
      rests at zero degrees -- i.e. flat.

So baking is: read where the fabric is right now, write that in as the rest
state, and list a rest angle for every adjacent triangle pair so nothing is
left on the flat default. After that the bent duct is, as far as the solver is
concerned, already relaxed.

    from duct_sim.plastic import bake_rest_shape
    bake_rest_shape(stage)        # press R in duct_layout_gui.py
"""

from __future__ import annotations

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt


def _fabric_points(path):
    """Current simulated points. USD cannot answer this under GPU physics."""
    try:
        from usdrt import Usd as RtUsd
        rt = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
        prim = rt.GetPrimAtPath(path)
        if prim is None or not prim.IsValid() or not prim.HasAttribute("points"):
            return None
        pts = prim.GetAttribute("points").Get()
        return None if pts is None else np.array(pts, dtype=np.float64)
    except Exception:
        return None


def _adjacent_triangle_pairs(tris):
    """[(t0, t1, shared_edge_v0, shared_edge_v1), ...] for every interior edge."""
    edges = {}
    for ti, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append(ti)
    out = []
    for (u, v), ts in edges.items():
        if len(ts) == 2:
            out.append((ts[0], ts[1], u, v))
    return out


def _dihedral_angles(points, tris, pairs):
    """Signed dihedral angle in DEGREES for each adjacent triangle pair."""
    angles = np.zeros(len(pairs), dtype=np.float32)
    for i, (t0, t1, u, v) in enumerate(pairs):
        a, b = points[u], points[v]
        e = b - a
        ne = np.linalg.norm(e)
        if ne < 1e-12:
            continue
        e = e / ne

        # the vertex of each triangle that is NOT on the shared edge
        o0 = [x for x in tris[t0] if x not in (u, v)]
        o1 = [x for x in tris[t1] if x not in (u, v)]
        if not o0 or not o1:
            continue
        p0, p1 = points[o0[0]], points[o1[0]]

        # project both opposite vertices into the plane perpendicular to the edge
        d0 = p0 - a
        d1 = p1 - a
        d0 = d0 - np.dot(d0, e) * e
        d1 = d1 - np.dot(d1, e) * e
        n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if n0 < 1e-12 or n1 < 1e-12:
            continue
        d0, d1 = d0 / n0, d1 / n1

        cos = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
        # sign from which side of the edge the second wing folds to
        sin = float(np.dot(np.cross(d0, d1), e))
        # flat sheet -> the wings point opposite ways -> angle 0
        angles[i] = np.degrees(np.arctan2(sin, -cos))
    return angles


def bake_rest_shape(stage=None, verbose=True):
    """Freeze every sleeve's CURRENT shape in as its rest shape.

    Returns the number of sleeves baked.
    """
    stage = stage or omni.usd.get_context().get_stage()
    n_done = 0

    for prim in stage.Traverse():
        if not prim.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"):
            continue
        path = str(prim.GetPath())

        tri_attr = prim.GetAttribute("omniphysics:restTriVtxIndices")
        if not tri_attr or not tri_attr.IsValid() or tri_attr.Get() is None:
            if verbose:
                print(f"[plastic] {path}: no restTriVtxIndices, skipped", flush=True)
            continue
        tris = np.array(tri_attr.Get(), dtype=np.int64)

        pts = _fabric_points(path)
        if pts is None:
            # the sim mesh may carry the points on the prim itself
            pa = prim.GetAttribute("points")
            pts = np.array(pa.Get(), dtype=np.float64) if pa and pa.Get() is not None else None
        if pts is None:
            if verbose:
                print(f"[plastic] {path}: no live points, skipped", flush=True)
            continue

        if tris.max() >= len(pts):
            if verbose:
                print(f"[plastic] {path}: index/point mismatch "
                      f"({tris.max()} >= {len(pts)}), skipped", flush=True)
            continue

        prim.GetAttribute("omniphysics:restShapePoints").Set(
            Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))

        pairs = _adjacent_triangle_pairs(tris)
        if pairs:
            angles = _dihedral_angles(pts, tris, pairs)
            ap = prim.GetAttribute("omniphysics:restAdjTriPairs")
            if not ap or not ap.IsValid():
                ap = prim.CreateAttribute("omniphysics:restAdjTriPairs",
                                          Sdf.ValueTypeNames.Int2Array)
            ap.Set(Vt.Vec2iArray([(int(t0), int(t1)) for t0, t1, _, _ in pairs]))

            ba = prim.GetAttribute("omniphysics:restBendAngles")
            if not ba or not ba.IsValid():
                ba = prim.CreateAttribute("omniphysics:restBendAngles",
                                          Sdf.ValueTypeNames.FloatArray)
            ba.Set(Vt.FloatArray(angles.tolist()))

        n_done += 1
        if verbose:
            print(f"[plastic] baked {path}: {len(pts)} pts, {len(tris)} tris, "
                  f"{len(pairs)} edges", flush=True)

    if verbose:
        print(f"[plastic] rest shape updated on {n_done} sleeves", flush=True)
    return n_done


def bake_and_restart(stage=None, sim=None, verbose=True):
    """The full "make it stay" operation. This is what the Bake button runs.

    Rewriting the rest shape alone does nothing visible, for two reasons:

      1. PhysX already holds the deformable COOKED in GPU memory. Editing USD
         attributes underneath a running simulation does not reach the solver,
         so the change only takes effect on a stop/play.
      2. On that restart PhysX re-reads USD -- where the `points` are still the
         straight duct, because GPU results live in Fabric and never write
         back. So a naive restart would snap the duct straight again and leave
         it resting against a bent target: worse than before.

    So all three steps have to happen together:

        capture + apply   the arranged geometry becomes the authored geometry
        bake rest shape   the arranged geometry also becomes the rest state
        stop / play       PhysX re-cooks from that

    After this the duct starts bent, rests bent, and no longer springs back.
    """
    from duct_sim.freeze import apply_to_stage, capture

    stage = stage or omni.usd.get_context().get_stage()

    live = capture(stage)
    n_pts, n_xf = apply_to_stage(live, stage, verbose=False)
    if verbose:
        print(f"[plastic] geometry baked into USD: {n_pts} meshes, "
              f"{n_xf} transforms", flush=True)

    n = bake_rest_shape(stage, verbose=verbose)

    if sim is not None:
        # Without this the solver keeps running on the old cooked state and the
        # button appears to do nothing at all.
        sim.stop()
        sim.play()
        if verbose:
            print("[plastic] simulation restarted so PhysX re-cooks the new "
                  "rest state", flush=True)
    elif verbose:
        print("[plastic] NOTE no sim handle given: stop/play the simulation "
              "for the bake to take effect", flush=True)
    return n


def _duct_centreline(stage, sim_mesh_path):
    """Hoop centres of the duct this sim mesh belongs to, in path order.

    Each ring carries its own translate op, so this is exact for a duct that
    curves -- which is the whole point: a track duct turns, and any global axis
    stops describing it the moment it does.
    """
    root = Sdf.Path(sim_mesh_path)
    while root.pathElementCount > 2:
        root = root.GetParentPath()
    prim = stage.GetPrimAtPath(root)
    if not prim:
        return None
    centres = []
    for child in sorted(prim.GetChildren(), key=lambda c: c.GetName()):
        if not child.GetName().startswith("ring_"):
            continue
        for op in UsdGeom.Xformable(child).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get()
                centres.append([float(t[0]), float(t[1]), float(t[2])])
                break
    return np.array(centres, dtype=np.float64) if len(centres) >= 2 else None


def shrink_rest_shape(stage=None, factor=0.85, axis=None, verbose=True):
    """Make the fabric WANT to be narrower than it is, so it pulls tight.

    The other route -- drawing the fabric inside the hoops and constraining it
    as if it touched them -- cannot work: an attachment records the gap it was
    built with and preserves it, measured at 146.0 mm with and without
    enableRigidSurfaceAttachments. Tension has to come from the rest state
    rather than from the constraint.

    So: leave the fabric where it is and rewrite its REST shape to a narrower
    tube. The solver pulls every vertex inward; the hoops block that at each
    station, so the fabric goes taut over them and cinches between.

    Only the cross-section shrinks, never the length. Scaling the rest shape
    uniformly would shorten the duct and drag the hoops together, since nothing
    joins them but the fabric.

    Each vertex is assigned to its NEAREST HOOP and scaled in the plane normal
    to the centreline there. The first version binned along a world axis, which
    is right only for a duct lying along that axis: on the track, a duct running
    in y put its whole length into one bin -- 8,040 points in 60 "rings" where a
    ring is 40 points -- and got squashed about the duct's own centre instead of
    narrowed. `axis` is kept for call compatibility and ignored.
    """
    stage = stage or omni.usd.get_context().get_stage()
    n_done = 0
    for prim in stage.Traverse():
        if not prim.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"):
            continue
        path = str(prim.GetPath())
        pts = _fabric_points(path)
        if pts is None:
            a = prim.GetAttribute("points")
            pts = np.array(a.Get(), dtype=np.float64) if a and a.Get() is not None else None
        if pts is None or not len(pts):
            continue

        centres = _duct_centreline(stage, path)
        if centres is None:
            if verbose:
                print(f"[taut] {path}: no centreline found, skipped", flush=True)
            continue

        # nearest hoop for every vertex
        d = pts[:, None, :] - centres[None, :, :]
        near = np.argmin((d * d).sum(axis=2), axis=1)

        # local tangent, central difference along the centreline
        tang = np.gradient(centres, axis=0)
        nrm = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = tang / np.where(nrm < 1e-12, 1.0, nrm)

        c, t = centres[near], tang[near]
        rel = pts - c
        along = (rel * t).sum(axis=1, keepdims=True) * t   # length preserved
        perp = rel - along                                  # section shrunk
        rest = c + along + perp * factor

        prim.CreateAttribute("omniphysics:restShapePoints",
                             Sdf.ValueTypeNames.Point3fArray).Set(
            Vt.Vec3fArray([Gf.Vec3f(*q) for q in rest]))

        tris = prim.GetAttribute("omniphysics:restTriVtxIndices")
        tris = tris.Get() if tris else None
        if tris is None:
            tris = prim.GetAttribute("faceVertexIndices").Get()
        if tris is not None:
            tri = np.array(tris, dtype=np.int64).reshape(-1, 3)
            pairs = _adjacent_triangle_pairs(tri)
            if len(pairs):
                ang = _dihedral_angles(rest, tri, pairs)
                prim.CreateAttribute("omniphysics:restBendAngles",
                                     Sdf.ValueTypeNames.FloatArray).Set(
                    Vt.FloatArray([float(a) for a in ang]))

        n_done += 1
        if verbose:
            print(f"[taut] {path}: rest cross-section x{factor:.2f} "
                  f"({len(pts)} pts over {len(centres)} hoops)", flush=True)

    if verbose:
        print(f"[taut] rest shape narrowed on {n_done} sleeve(s)", flush=True)
    return n_done
