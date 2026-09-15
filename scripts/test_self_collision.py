"""Does selfCollision stop the tube folding in on itself, and what does it cost?

With selfCollision off the fabric can pass through itself, so a fold is
permanent: once the tube creases inward nothing pushes it back out. That is the
mechanism behind the crumpled-fan failures. Turning it on should keep the tube
open, at the price of solver time.

Measured, under a hard sideways bend:
    cross-section radius   does the tube stay round, or collapse?
    ms/step                what the protection costs
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--n-rings", type=int, default=20)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--self-collision", action="store_true")
ap.add_argument("--n-circ", type=int, default=28)
ap.add_argument("--bend-force", type=float, default=3.0)
ap.add_argument("--steps", type=int, default=900)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import dataclasses  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.builder import min_segments, spawn_duct_single  # noqa: E402
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
spec = dataclasses.replace(spec, ring_thickness=0.002, ring_density=7800.0)
spec = dataclasses.replace(
    spec, ring_segments=max(spec.ring_segments,
                            min_segments(spec.ring_thickness, spec.radius)))

MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=2e4, surface_shear_stiffness=2.0,
    surface_bend_stiffness=1e-3)

root, rings, _, n_bound = spawn_duct_single(
    stage, 0, args.n_rings, args.spacing, spec, MAT, origin=(0, 0, 0.30),
    n_circ=args.n_circ, self_collision=args.self_collision)
print(f"[sc] selfCollision={args.self_collision}  seam elements={n_bound}",
      flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(60):
    sim.step(render=False)

from duct_sim.mouse_drag import _to_backend  # noqa: E402
from duct_sim.plastic import _fabric_points  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

view = RigidPrim(rings)
f = np.zeros((len(rings), 3), dtype=np.float32)
mid = len(rings) // 2
for j in range(max(0, mid - 3), min(len(rings), mid + 4)):
    f[j, 1] = args.bend_force

SLEEVE = f"{root}/skin/simMesh"


def radius():
    p = _fabric_points(SLEEVE)
    if p is None or not len(p):
        return None
    # radius about the local duct axis, measured per axial station
    order = np.argsort(p[:, 0])
    p = p[order]
    n = len(p)
    chunk = max(8, n // 20)
    rs = []
    for i in range(0, n - chunk, chunk):
        seg = p[i:i + chunk]
        c = seg.mean(axis=0)
        rs.append(np.linalg.norm(seg[:, 1:] - c[1:], axis=1).mean())
    return float(np.mean(rs)), float(np.min(rs))


t0 = time.perf_counter()
for step in range(args.steps + 1):
    if step < args.steps * 0.6:
        view.apply_forces(_to_backend(f), is_global=True)
    if step % 150 == 0:
        r = radius()
        if r:
            print(f"[sc] step {step:4d}  section radius mean {r[0]:.4f} "
                  f"min {r[1]:.4f}  ({r[0]/spec.radius*100:.0f}% of nominal)",
                  flush=True)
    sim.step(render=False)
wall = time.perf_counter() - t0

pos, _ = view.get_world_poses()
pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos)
print(f"[sc] RESULT selfCollision={args.self_collision}  "
      f"{wall / (args.steps + 1) * 1e3:.1f} ms/step  "
      f"final deflection {pos[:,1].max()-pos[:,1].min():.3f} m", flush=True)
print("[sc] END", flush=True)
app.close()
