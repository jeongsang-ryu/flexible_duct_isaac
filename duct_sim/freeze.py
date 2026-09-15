"""Save the SIMULATED duct state to USD, so a hand-arranged layout can be reloaded.

WHY THIS MODULE EXISTS. With GPU dynamics the simulated state lives in Fabric
and is never written back to USD -- `mesh.GetPointsAttr().Get()` returns the
authored pose forever, no matter how far the duct has actually moved. So
`stage.Export()` after dragging a duct into place saves the STRAIGHT duct, not
the arranged one. (Measured: USD reports 0.000000 m of movement while Fabric
reports metres on the same prims.)

The fix is to read Fabric through usdrt, write those values onto the USD prims,
and only then export. What comes out is a plain USD file with no Fabric
dependency: the duct is authored in its arranged shape.

    from duct_sim.freeze import freeze_to_usd
    freeze_to_usd("/path/to/track_layout.usd")
"""

from __future__ import annotations

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt


def _fabric_stage():
    from usdrt import Usd as RtUsd
    return RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())


def _fabric_get(rt_prim, name):
    """Read one Fabric attribute, or None if it is not there."""
    if rt_prim is None or not rt_prim.IsValid():
        return None
    if not rt_prim.HasAttribute(name):
        return None
    attr = rt_prim.GetAttribute(name)
    return attr.Get() if attr else None


def capture(stage=None):
    """Pull the live state out of Fabric. Returns {path: {...}} with no USD writes."""
    stage = stage or omni.usd.get_context().get_stage()
    rt = _fabric_stage()
    out = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        rp = rt.GetPrimAtPath(path)
        if rp is None or not rp.IsValid():
            continue
        rec = {}

        pts = _fabric_get(rp, "points")
        if pts is not None:
            rec["points"] = np.array(pts, dtype=np.float32)

        if rec:
            out[path] = rec

    # RIGID BODIES DO NOT COME FROM FABRIC HERE. Reading xformOp:translate
    # through usdrt returned the AUTHORED pose (ring_00 still at z=1.2 while
    # its own cloth had fallen to z=0.02), because omni.physx.fabric only
    # publishes rigid transforms when it is running -- and the cloth points
    # happened to be there while the transforms were not. PhysX itself always
    # knows, so ask it directly instead of trusting Fabric for this half.
    try:
        from isaacsim.core.prims import RigidPrim
        paths = [str(pm.GetPath()) for pm in stage.Traverse()
                 if pm.HasAPI(UsdPhysics.RigidBodyAPI)]
        if paths:
            view = RigidPrim(paths)
            pos, orn = view.get_world_poses()
            pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, dtype=np.float64)
            orn = np.asarray(orn.cpu() if hasattr(orn, "cpu") else orn, dtype=np.float64)
            for i, pth in enumerate(paths):
                rec = out.setdefault(pth, {})
                rec["position"] = pos[i]
                rec["orientation"] = orn[i]      # wxyz from Isaac
                rec["orientation_wxyz"] = True
    except Exception as exc:
        print(f"[freeze] WARNING rigid poses unavailable: {exc}", flush=True)

    return out


def apply_to_stage(state, stage=None, verbose=True):
    """Write a captured state onto the USD prims. Returns (n_meshes, n_xforms)."""
    stage = stage or omni.usd.get_context().get_stage()
    n_pts = n_xf = 0
    for path, rec in state.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue

        if "points" in rec:
            attr = prim.GetAttribute("points")
            if attr and attr.IsValid():
                old = np.array(attr.Get()) if attr.Get() is not None else None
                new = rec["points"]
                if old is None or old.shape == new.shape:
                    attr.Set(Vt.Vec3fArray.FromNumpy(new))
                    n_pts += 1

        if "position" in rec and prim.IsA(UsdGeom.Xformable):
            xf = UsdGeom.Xformable(prim)
            # ClearXformOpOrder() only clears the ORDER -- the xformOp:*
            # attributes stay on the prim, and AddTranslateOp() then throws
            # "the xformOp already exists". The properties have to go.
            for name in list(prim.GetPropertyNames()):
                if name.startswith("xformOp:"):
                    prim.RemoveProperty(name)
            xf.ClearXformOpOrder()

            p = rec["position"]
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            if "orientation" in rec:
                q = rec["orientation"]
                if rec.get("orientation_wxyz"):
                    w, x, y, z = q          # Isaac RigidPrim order
                else:
                    x, y, z, w = q          # usdrt order
                # Pin the precision. The ring prims already carry a float-typed
                # xformOp:orient, and writing a Quatd into it fails with
                # "Type mismatch: expected GfQuatf, got GfQuatd".
                xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
                    Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
            n_xf += 1

    if verbose:
        print(f"[freeze] baked {n_pts} meshes and {n_xf} transforms", flush=True)
    return n_pts, n_xf


def freeze_to_usd(out_path, stage=None, verbose=True):
    """Capture the live state, bake it onto the stage, and export a plain USD."""
    stage = stage or omni.usd.get_context().get_stage()
    state = capture(stage)
    n_pts, n_xf = apply_to_stage(state, stage, verbose=verbose)
    stage.GetRootLayer().Export(out_path)
    if verbose:
        print(f"[freeze] wrote {out_path}", flush=True)
    return n_pts, n_xf
