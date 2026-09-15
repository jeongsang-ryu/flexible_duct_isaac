"""Does Bake actually make a bend stay? Measure it, do not eyeball it.

Protocol, run twice in one process is not possible (one physics scene), so the
control is the BEFORE window of the same run:

    1. push the middle hoops sideways for a while  -> duct bends
    2. release, let it settle                      -> record deflection D_bent
    3. bake_and_restart()
    4. run on with NO force                        -> record deflection D_after

    no bake  => D_after collapses toward 0 (springs back straight)
    bake     => D_after stays close to D_bent

--no-bake runs the same script without step 3, which is the control.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--n-rings", type=int, default=16)
ap.add_argument("--spacing", type=float, default=0.1)
ap.add_argument("--push", type=float, default=2.0, help="lateral force [N]")
ap.add_argument("--push-steps", type=int, default=300)
ap.add_argument("--settle-steps", type=int, default=240)
ap.add_argument("--pre-settle", type=int, default=400)
ap.add_argument("--after-steps", type=int, default=600)
ap.add_argument("--no-bake", action="store_true", help="control run")
ap.add_argument("--ring-thickness", type=float, default=0.0)
ap.add_argument("--ring-density", type=float, default=0.0)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.builder import min_segments, spawn_duct  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sc.CreateGravityMagnitudeAttr().Set(9.81)
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

import dataclasses  # noqa: E402

spec = DuctSpec()
_over = {}
if args.ring_thickness:
    _over["ring_thickness"] = args.ring_thickness
if args.ring_density:
    _over["ring_density"] = args.ring_density
if _over:
    spec = dataclasses.replace(spec, **_over)
_need = min_segments(spec.ring_thickness, spec.radius)
if spec.ring_segments < _need:
    spec = dataclasses.replace(spec, ring_segments=_need)
print(f"[test] hoop tube {spec.ring_thickness*1e3:.2f} mm, "
      f"{spec.ring_segments} segments, density {spec.ring_density:.0f}",
      flush=True)
MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=2.0e4, surface_shear_stiffness=2.0,
    surface_bend_stiffness=1.0e-3)

root, rings, n_ok, n_bound = spawn_duct(
    stage, 0, args.n_rings, args.spacing, spec, MAT,
    origin=(0.0, 0.0, 0.30), anchor_ends=True)
print(f"[test] seam elements bound: {n_bound}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
# SETTLE FIRST. Measuring the baseline after only 20 steps caught the duct
# still moving from spawn: it read 0.077 m of "deflection" on an authored-
# straight duct, and the push then appeared to REDUCE the bend. The baseline
# has to be a duct that has stopped moving.
for _ in range(args.pre_settle):
    sim.step(render=False)

from duct_sim.mouse_drag import _to_backend  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

view = RigidPrim(rings)


def deflection():
    p, _ = view.get_world_poses()
    p = np.asarray(p.cpu() if hasattr(p, "cpu") else p)
    return float(p[:, 1].max() - p[:, 1].min()), p


d0, _ = deflection()
print(f"[test] deflection at rest after {args.pre_settle} settle steps: "
      f"{d0:.4f} m", flush=True)

mid = args.n_rings // 2
forces = np.zeros((len(rings), 3), dtype=np.float32)
for j in (mid - 1, mid, mid + 1):
    if 0 <= j < len(rings):
        forces[j, 1] = args.push

for _ in range(args.push_steps):
    view.apply_forces(_to_backend(forces), is_global=True)
    sim.step(render=False)
for _ in range(args.settle_steps):
    sim.step(render=False)

d_bent, _ = deflection()
print(f"[test] deflection after push+settle: {d_bent:.4f} m", flush=True)

if not args.no_bake:
    from duct_sim.plastic import bake_and_restart
    bake_and_restart(stage, sim)
    view = RigidPrim(rings)      # the restart invalidates the old view
    for _ in range(20):
        sim.step(render=False)

for _ in range(args.after_steps):
    sim.step(render=False)

d_after, _ = deflection()
print(f"[test] deflection after {'BAKE + ' if not args.no_bake else 'NO BAKE, '}"
      f"{args.after_steps} free steps: {d_after:.4f} m", flush=True)

kept = (d_after - d0) / (d_bent - d0) * 100 if d_bent - d0 > 1e-6 else 0.0
print(f"[test] RESULT bend retained: {kept:.1f}%   "
      f"(start {d0:.4f} -> bent {d_bent:.4f} -> after {d_after:.4f})", flush=True)
app.close()
