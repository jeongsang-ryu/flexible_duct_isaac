"""Is the fabric trying to UNROLL the tube? Measure the cross-section over time.

restBendAnglesDefault is 'flatDefault': any adjacent triangle pair without an
explicit rest angle rests at ZERO degrees, i.e. flat. Our sleeve is a cylinder,
so every circumferential edge is being pulled toward flat -- the fabric wants to
become a sheet. That would show up as the tube collapsing between the hoops
(the "empty gap" look) and as a large restoring force fighting every bend.

Test: track the mean radius of the sleeve's points about the duct axis.

    holds ~0.20 m   -> the tube is stable, unrolling is not the problem
    shrinks         -> the fabric is flattening the tube

--bake-at N runs bake_rest_shape once at step N, which replaces those flat rest
angles with the cylinder's real ones. If the collapse stops after that, the
diagnosis is confirmed and the fix is to bake at build time.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--n-rings", type=int, default=12)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--report-every", type=int, default=150)
ap.add_argument("--bake-at", type=int, default=0, help="0 = never bake")
# These were hardcoded, which made the first run a test of the DEFAULT cloth
# rather than of the soft cloth being complained about. The measurement said
# "no collapse" while the soft settings were visibly collapsing.
ap.add_argument("--stretch", type=float, default=2.0e4)
ap.add_argument("--shear", type=float, default=2.0)
ap.add_argument("--bend", type=float, default=1.0e-3)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.builder import spawn_duct  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr().Set(9.81)
scene_prim = stage.GetPrimAtPath("/World/physicsScene")
scene_prim.ApplyAPI("PhysxSceneAPI")
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
scene_prim.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                           Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

spec = DuctSpec()
MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=args.stretch, surface_shear_stiffness=args.shear,
    surface_bend_stiffness=args.bend)
print(f"[unroll] cloth: stretch {args.stretch:.3g}, shear {args.shear:.3g}, "
      f"bend {args.bend:.3g}", flush=True)

root, rings, n_ok, n_bound = spawn_duct(
    stage, 0, args.n_rings, args.spacing, spec, MAT, origin=(0, 0, 0.30))
print(f"[unroll] seam elements: {n_bound}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

from duct_sim.plastic import _fabric_points  # noqa: E402

SLEEVE = f"{root}/sleeve_000/simMesh"


def cross_section():
    """Mean and min distance of the sleeve points from the duct axis (the x axis)."""
    p = _fabric_points(SLEEVE)
    if p is None or not len(p):
        return None
    c = p.mean(axis=0)
    r = np.linalg.norm(p[:, 1:] - c[1:], axis=1)   # radial distance in the y-z plane
    return float(r.mean()), float(r.min()), float(r.max())


print(f"[unroll] nominal tube radius: {spec.radius:.3f} m", flush=True)
baked = False
for step in range(args.steps + 1):
    if args.bake_at and step == args.bake_at and not baked:
        from duct_sim.plastic import bake_rest_shape
        bake_rest_shape(stage, verbose=False)
        sim.stop()
        sim.play()
        baked = True
        print(f"[unroll] --- baked rest shape at step {step} and restarted ---",
              flush=True)
    if step % args.report_every == 0:
        cs = cross_section()
        if cs:
            m, lo, hi = cs
            print(f"[unroll] step {step:5d}  radius mean {m:.4f}  "
                  f"min {lo:.4f}  max {hi:.4f}  "
                  f"({m/spec.radius*100:.0f}% of nominal)", flush=True)
    sim.step(render=False)

print("[unroll] END", flush=True)
app.close()
