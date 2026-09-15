"""Build the same duct five ways and measure them on equal terms.

The five are the standard options for a flexible duct in Isaac:

  static   one swept tube mesh, capsule colliders, nothing moves
  rigid    disc/sleeve chain hinged by compliant D6 joints
  hybrid   the rigid chain, with a ribbed tube re-swept onto it each step
  cloth    PhysX surface deformable sewn to hoops (the original build)
  fem      a solid low-resolution cylinder as a volume deformable

Same length, same spacing, same scene, same number of steps. Reports build
time, step time (marginal, not the running average -- the first hundreds of
steps are start-up and never leave a cumulative figure), GPU memory, and shape
counts, then writes docs/approaches.md.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True,
                choices=["static", "rigid", "hybrid", "cloth", "fem"])
ap.add_argument("--length", type=float, default=6.0)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--n-circ", type=int, default=28,
                help="fabric segments around the tube -- the cloth's resolution, "
                     "varied independently of hoop count")
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--measure", type=int, default=600)
ap.add_argument("--out", default="/tmp/bench")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

import duct_sim.builder as B  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402


def gpu_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        me = str(os.getpid())
        for line in out.splitlines():
            pid, mb = [t.strip() for t in line.split(",")[:2]]
            if pid == me:
                return int(mb)
    except Exception:
        pass
    return -1


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
gx.AddScaleOp().Set(Gf.Vec3f(30, 30, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

spec = DuctSpec()
spec_ring = B.min_segments(spec.ring_thickness, spec.radius)
import dataclasses  # noqa: E402
spec = dataclasses.replace(spec, ring_segments=max(spec.ring_segments, spec_ring))

n = max(2, int(round(args.length / args.spacing)) + 1)
stations = [(-args.length / 2 + args.spacing * k, 0.0, 0.0) for k in range(n)]

MAT = "/World/duct_material"
if args.mode in ("cloth", "fem"):
    if args.mode == "cloth":
        deformableUtils.add_surface_deformable_material(
            stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
            surface_stretch_stiffness=2e4, surface_shear_stiffness=2.0,
            surface_bend_stiffness=1e-3)
    else:
        deformableUtils.add_deformable_material(
            stage, MAT, dynamic_friction=0.5, youngs_modulus=5.0e5,
            poissons_ratio=0.45, density=120.0)

info = {"mode": args.mode, "length_m": args.length, "spacing_m": args.spacing,
        "stations": len(stations), "n_circ": args.n_circ}
skin = None

t0 = time.perf_counter()
if args.mode == "static":
    _, _, _, n_col = B.spawn_duct_static(stage, 0, stations, spec)
    info.update(bodies=0, joints=0, shapes=n_col)
elif args.mode == "rigid":
    _, paths, nb, nj = B.spawn_duct_rigid(stage, 0, stations, spec)
    info.update(bodies=nb, joints=nj, shapes=nb)
elif args.mode == "hybrid":
    _, paths, skin, nb = B.spawn_duct_hybrid(stage, 0, stations, spec)
    info.update(bodies=nb, joints=nb - 1, shapes=nb)
elif args.mode == "cloth":
    _, rings, _, n_bound = B.spawn_duct_path(stage, 0, stations, spec, MAT,
                                             clearance=-0.004, n_circ=args.n_circ)
    info.update(bodies=len(rings), joints=0, n_circ=args.n_circ,
                shapes=len(rings) * spec.ring_segments, seam_elements=n_bound)
else:
    _, _, ok, _ = B.spawn_duct_fem(stage, 0, stations, spec, MAT)
    info.update(bodies=1 if ok else 0, joints=0, shapes=1 if ok else 0)
info["build_s"] = round(time.perf_counter() - t0, 2)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

for _ in range(args.warmup):
    if skin:
        skin.update()
    sim.step(render=False)

t0 = time.perf_counter()
for _ in range(args.measure):
    if skin:
        skin.update()
    sim.step(render=False)
elapsed = time.perf_counter() - t0

info["ms_per_step"] = round(elapsed / args.measure * 1e3, 3)
info["gpu_mb"] = gpu_mb()
info["real_time_ratio"] = round(info["ms_per_step"] / (1000 / 120), 2)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"{args.mode}.json"), "w") as fh:
    json.dump(info, fh, indent=1)
print("BENCH " + json.dumps(info), flush=True)
app.close()
