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
print(f"\n{'taut':>6} {'r at hoop':>12} {'r between':>12} {'verdict':>28}", flush=True)
print("-" * 62, flush=True)

for taut in (False, True):
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
        stage, 0, stations, spec, MAT, clearance=args.clearance,
        n_circ=args.n_circ, taut=taut, verbose=False)

    sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
    sim.initialize_physics()
    sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
    sim.play()
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
    r_hoop = radius_near(x_hoop, args.spacing * 0.15)
    r_between = radius_near(x_between, args.spacing * 0.15)

    pulled = abs(r_hoop - hoop_inner) < abs(r_hoop - authored_r)
    verdict = "PULLED OUT to the hoop" if pulled else "stayed where it was drawn"
    print(f"{str(taut):>6} {r_hoop*1e3:10.1f} mm {r_between*1e3:10.1f} mm "
          f"{verdict:>28}", flush=True)

    sim.stop()
    sim.clear_instance()

print(f"\nauthored {authored_r*1e3:.1f} mm vs hoop {hoop_inner*1e3:.1f} mm — "
      f"whichever the fabric ends up near is the answer.", flush=True)
app.close()
