"""Does --taut actually pull the fabric out onto the hoops?

The fabric is authored well INSIDE the hoops. If enableRigidSurfaceAttachments
works, the hoop surfaces become the attachment targets and the fabric is dragged
out to the hoop radius on play. If it does not, the fabric stays where it was
drawn and the whole idea is dead.

Measured directly: the radius of the fabric AT a hoop station, before and after
simulating, against the two candidate answers.

    stays near the authored radius  -> taut does nothing
    moves to the hoop radius        -> it works
"""

import argparse
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=float, default=2.0)
ap.add_argument("--spacing", type=float, default=0.10)
ap.add_argument("--clearance", type=float, default=-0.050)
ap.add_argument("--shrink", type=float, default=0.80,
                help="rest cross-section factor for the shrink case")
ap.add_argument("--n-circ", type=int, default=32)
ap.add_argument("--steps", type=int, default=900)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.builder import min_segments, spawn_duct_path  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

spec = DuctSpec()
spec = dataclasses.replace(spec, ring_thickness=0.004)
spec = dataclasses.replace(
    spec, ring_segments=max(spec.ring_segments,
                            min_segments(spec.ring_thickness, spec.radius)))

n = max(2, int(round(args.length / args.spacing)) + 1)
stations = [(-args.length / 2 + args.spacing * k, 0.0, 0.0) for k in range(n)]
authored_r = spec.radius - spec.ring_thickness + args.clearance
hoop_inner = spec.radius - spec.ring_thickness

print(f"\nfabric authored at {authored_r*1e3:.1f} mm; hoop inner surface at "
      f"{hoop_inner*1e3:.1f} mm; gap {(hoop_inner-authored_r)*1e3:.1f} mm",
      flush=True)
print(f"\n{'case':>10} {'r at hoop':>12} {'r between':>12} {'verdict':>28}", flush=True)
print("-" * 78, flush=True)

# third case: fabric drawn ON the hoops, but its REST cross-section narrowed so
# it pulls itself tight against them
# The shrink case contracted the fabric AT the hoops as well as between them,
# with 504 bonds authored and surviving the restart. Counting USD children
# proves authoring, not that the solver is honouring them. Pinning the hoops
# separates the two: if the fabric still pulls off a STATIC hoop, the bond is
# not doing anything; if it holds there and cinches between, the mechanism
# works and the free hoops were simply being dragged along.
CASES = [("off", False, 0.0, args.clearance, False),
         ("taut", True, 0.0, args.clearance, False),
         ("shrink", False, args.shrink, -0.004, False),
         ("shrink+pin", False, args.shrink, -0.004, True),
         # Pinning changed nothing, so the bonds are not surviving the restart
         # in the solver even though their prims survive in USD. Re-authoring
         # them AFTER the restart is the remaining option: stop resets the
         # fabric to its authored radius, so seams rebuilt there capture it ON
         # the hoops, and only then does it try to shrink away from them.
         ("shrink+rebond", False, args.shrink, -0.004, False),
         # CONTROL, and it matters beyond this feature: stop/play with NO rest
         # change. Bake (R) restarts the sim too, so if a plain restart is what
         # drops the attachments then bake is broken in the same way and the
         # duct comes apart after every bake.
         ("restart-only", False, -1.0, -0.004, False)]

for label, taut, shrink, clearance, pin in CASES:
    rebond = label.endswith("rebond")
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr().Set(9.81)
    sp = stage.GetPrimAtPath("/World/physicsScene")
    sp.ApplyAPI("PhysxSceneAPI")
    sp.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    sp.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
    sp.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
    sp.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                       Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(g)
    gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    MAT = "/World/cloth_material"
    deformableUtils.add_surface_deformable_material(
        stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
        surface_stretch_stiffness=3e3, surface_shear_stiffness=1.0,
        surface_bend_stiffness=1e-3)

    root, rings, _, n_bound = spawn_duct_path(
        stage, 0, stations, spec, MAT, clearance=clearance,
        n_circ=args.n_circ, taut=taut, verbose=False)
    drawn_r = (spec.radius - spec.ring_thickness + clearance)

    if pin:
        for rp_path in rings:
            mp = UsdPhysics.MassAPI.Get(stage, rp_path)
            if not mp:
                mp = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(rp_path))
            mp.CreateDensityAttr(0.0)      # static: nothing drags it inward

    sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
    sim.initialize_physics()
    sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
    sim.play()
    if shrink < 0:            # restart with no rest change: the control
        for _ in range(120):
            sim.step(render=False)
        sim.stop()
        sim.play()
    if shrink > 0:
        # the rest arrays only exist once the body has been cooked, so let it
        # run briefly, narrow the rest shape, then stop/play so PhysX re-cooks
        for _ in range(120):
            sim.step(render=False)
        from duct_sim.plastic import shrink_rest_shape
        shrink_rest_shape(stage, factor=shrink, axis=0, verbose=False)
        sim.stop()
        if rebond:
            from duct_sim.builder import rebuild_seams
            rebuild_seams(stage, -1, verbose=False)
        sim.play()
        _after = 0
        for c in stage.GetPrimAtPath(f"{root}/skin").GetChildren():
            if c.GetName().startswith("seam"):
                _after += len(c.GetChildren())
        print(f"       bonds before restart {n_bound}, after {_after}", flush=True)
    for _ in range(args.steps):
        sim.step(render=False)

    from duct_sim.plastic import _fabric_points  # noqa: E402
    p = _fabric_points(f"{root}/skin/simMesh")
    if p is None or not len(p):
        print(f"{str(taut):>6}   (no fabric points)", flush=True)
        sim.stop(); sim.clear_instance(); continue

    # radius about the duct axis, at a hoop station and midway between two
    def radius_near(x_target, tol):
        sel = p[np.abs(p[:, 0] - x_target) < tol]
        if not len(sel):
            return float("nan")
        c = sel.mean(axis=0)
        return float(np.linalg.norm(sel[:, 1:] - c[1:], axis=1).mean())

    mid_i = n // 2
    x_hoop = stations[mid_i][0]
    x_between = x_hoop + args.spacing * 0.5
    r_hoop = radius_near(x_hoop, args.spacing * 0.25)
    r_between = radius_near(x_between, args.spacing * 0.25)

    moved = r_hoop - drawn_r
    if shrink < 0:
        verdict = ("bonds HELD across restart" if abs(r_hoop - drawn_r) < 0.005
                   else "bonds LOST across restart")
    elif shrink > 0:
        verdict = ("CINCHED between hoops" if (r_between < r_hoop - 0.002)
                   else "no waist formed")
    else:
        verdict = ("PULLED OUT to the hoop"
                   if abs(r_hoop - hoop_inner) < abs(r_hoop - drawn_r)
                   else "stayed where it was drawn")
    # n_bound is the decisive diagnostic: the fabric contracting AT the hoops
    # means the hoops are not holding it, and "held but overpowered" looks
    # nothing like "never bound in the first place"
    print(f"{label:>10} {r_hoop*1e3:10.1f} mm {r_between*1e3:10.1f} mm "
          f"{verdict:>28}   (drawn {drawn_r*1e3:.1f}, {n_bound} bonds)",
          flush=True)

    sim.stop()
    sim.clear_instance()

print(f"\nauthored {authored_r*1e3:.1f} mm vs hoop {hoop_inner*1e3:.1f} mm — "
      f"whichever the fabric ends up near is the answer.", flush=True)
app.close()
