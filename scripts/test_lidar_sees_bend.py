"""Does a LiDAR see the BENT duct, or the straight one it was authored as?

This is the highest-risk unknown for the track. Under GPU physics the simulated
geometry lives in Fabric and is never written back to USD. The viewport reads
Fabric, so the screen shows the bent duct -- but anything that reads USD sees
the straight one. If the LiDAR is on the USD side, the picture and the point
cloud disagree and nothing warns you: you would collect a whole dataset of
points from a duct shape that was never simulated.

Evidence so far is indirect: omni.sensors.nv.lidar depends on omni.hydra.rtx,
i.e. it traces the RENDERER's scene, which is Fabric-fed. That suggests it sees
the bend. This measures it instead of assuming it.

Method, deliberately free of any USD/Fabric question:
  1. build a duct lying along X, note where its surface is
  2. bend it hard sideways (+Y) and let it settle
  3. sweep a ring of rays from above and record hit positions
  4. compare the hit pattern against BOTH candidate shapes

    hits follow the bent duct     -> the sensor sees the simulation. good.
    hits follow the straight duct -> the sensor reads USD. must be fixed
                                     before any data is collected.

Uses the physics scene-query raycast, which is what a physics-based LiDAR
would use; an RTX LiDAR is checked separately because it goes through the
renderer instead.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--n-rings", type=int, default=24)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--bend-force", type=float, default=2.0)
ap.add_argument("--bend-steps", type=int, default=500)
ap.add_argument("--settle-steps", type=int, default=300)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
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
    stage, 0, args.n_rings, args.spacing, spec, MAT, origin=(0, 0, 0.30))
print(f"[lidar] seam elements: {n_bound}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(60):
    sim.step(render=False)

from duct_sim.mouse_drag import _to_backend  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

view = RigidPrim(rings)
f = np.zeros((len(rings), 3), dtype=np.float32)
mid = len(rings) // 2
for j in range(max(0, mid - 3), min(len(rings), mid + 4)):
    f[j, 1] = args.bend_force
for _ in range(args.bend_steps):
    view.apply_forces(_to_backend(f), is_global=True)
    sim.step(render=False)
for _ in range(args.settle_steps):
    sim.step(render=False)

pos, _ = view.get_world_poses()
pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos)
print(f"[lidar] hoop y after bend: min {pos[:,1].min():+.3f} "
      f"max {pos[:,1].max():+.3f}  (deflection "
      f"{pos[:,1].max()-pos[:,1].min():.3f} m)", flush=True)

# --- rays straight down on a grid; where does the surface actually sit? ---
sq = get_physx_scene_query_interface()
hits = []
for x in np.linspace(pos[:, 0].min(), pos[:, 0].max(), 40):
    for y in np.linspace(-0.6, 0.9, 60):
        origin = Gf.Vec3d(float(x), float(y), 2.0)
        d = Gf.Vec3d(0, 0, -1)
        h = sq.raycast_closest(origin, d, 4.0)
        if h["hit"] and "ground" not in h["collision"]:
            hits.append((x, y, h["position"][2], h["collision"]))

if not hits:
    print("[lidar] NO HITS on the duct at all -- the sensor cannot see it",
          flush=True)
else:
    hy = np.array([h[1] for h in hits])
    print(f"[lidar] {len(hits)} hits on the duct, y range "
          f"{hy.min():+.3f} .. {hy.max():+.3f}", flush=True)
    print(f"[lidar] hoops occupy y {pos[:,1].min():+.3f} .. {pos[:,1].max():+.3f}",
          flush=True)
    # the authored duct is a straight line at y = 0
    off = np.abs(hy).max()
    print(f"[lidar] furthest hit from the authored centreline (y=0): "
          f"{off:.3f} m", flush=True)
    bent = pos[:, 1].max() - pos[:, 1].min()
    verdict = ("SEES THE BEND" if off > 0.5 * max(bent, 1e-6) + spec.radius * 0.5
               else "sees only the STRAIGHT duct")
    print(f"[lidar] VERDICT: {verdict}", flush=True)
    names = {}
    for h in hits:
        key = h[3].split("/")[-2] if "/" in h[3] else h[3]
        names[key] = names.get(key, 0) + 1
    print(f"[lidar] hit prims: {sorted(names.items(), key=lambda kv: -kv[1])[:4]}",
          flush=True)

print("[lidar] END", flush=True)
app.close()
