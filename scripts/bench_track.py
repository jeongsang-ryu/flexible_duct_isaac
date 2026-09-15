"""What costs what on a full track: physics vs render, hoops vs fabric.

Counting triangles says the hoop VISUALS are 6x the fabric, but a triangle
count is not a cost -- it has to be measured. This builds the same track with
one thing changed at a time and reports marginal step time with rendering off
(pure physics) and on (physics + render), so the two are separated instead of
being blamed on each other.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True)
ap.add_argument("--n-circ", type=int, default=40)
ap.add_argument("--ring-segments", type=int, default=16)
ap.add_argument("--rib-visual", type=int, default=1)
ap.add_argument("--static", action="store_true",
                help="frozen swept tube + capsule colliders, no bodies -- what a track becomes after freeze, and the only variant that could carry thousands of parallel cars")
ap.add_argument("--solid-hoop", action="store_true",
                help="one convex disc per hoop instead of n capsules")
ap.add_argument("--no-fabric", action="store_true",
                help="hoops only -- isolates what the deformable costs")
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--measure", type=int, default=300)
ap.add_argument("--tag", default="full")
ap.add_argument("--out", default="/tmp/bench_track")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

import duct_sim.builder as B  # noqa: E402
from duct_sim.layout import load, resample  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402


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
    return 0


ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sp = stage.GetPrimAtPath("/World/physicsScene")
sp.ApplyAPI("PhysxSceneAPI")
UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)
g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(44, 44, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=3e3, surface_shear_stiffness=1.0,
    surface_bend_stiffness=1e-3)

doc = load(args.layout)
# DuctSpec is a frozen dataclass -- replace, do not assign
import dataclasses  # noqa: E402
spec = dataclasses.replace(DuctSpec(), ring_segments=args.ring_segments)
spacing = doc.get("duct", {}).get("hoop_spacing", 0.05)

t0 = time.perf_counter()
n_hoops = 0
if args.static:
    from duct_sim.builder import spawn_duct_static
    n_col = 0
    for i, run in enumerate(doc["runs"]):
        stations = resample(run["points"], spacing)
        _, _, _, c = spawn_duct_static(stage, i, stations, spec, n_circ=args.n_circ)
        n_col += c
        n_hoops += len(stations)
    print(f"[bench] static: {n_col} capsule colliders, 0 rigid bodies", flush=True)
elif args.no_fabric:
    # hoops alone, placed exactly where the full build would put them
    from duct_sim.geometry import create_ring
    R = spec.diameter * 0.5
    TUBE = getattr(spec, "ring_tube", 0.012)
    for i, run in enumerate(doc["runs"]):
        for k, (x, y, h) in enumerate(resample(run["points"], spacing)):
            p = f"/World/hoops_{i:02d}/ring_{k:04d}"
            create_ring(stage, p, R, TUBE, n_seg=spec.ring_segments,
                        solid=args.solid_hoop)
            prim = stage.GetPrimAtPath(p)
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.25))
            q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
                 * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(h)))
            xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(spec.ring_density)
            n_hoops += 1
else:
    for i, run in enumerate(doc["runs"]):
        stations = resample(run["points"], spacing)
        _, rings, _, _ = B.spawn_duct_path(
            stage, i, stations, spec, MAT, n_circ=args.n_circ,
            clearance=-0.004, rib_visual=bool(args.rib_visual),
            solid_hoop=args.solid_hoop, verbose=False)
        n_hoops += len(rings)
build_s = time.perf_counter() - t0

# count what is actually on the stage
meshes = tris = 0
for p in stage.Traverse():
    if p.IsA(UsdGeom.Mesh):
        meshes += 1
        c = UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get()
        tris += len(c) if c else 0

from isaacsim.core.api import SimulationContext  # noqa: E402

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

for _ in range(args.warmup):
    sim.step(render=False)

t = time.perf_counter()
for _ in range(args.measure):
    sim.step(render=False)
phys_ms = (time.perf_counter() - t) / args.measure * 1e3

for _ in range(30):
    sim.step(render=True)
t = time.perf_counter()
for _ in range(args.measure):
    sim.step(render=True)
both_ms = (time.perf_counter() - t) / args.measure * 1e3

info = dict(tag=args.tag, hoops=n_hoops, n_circ=args.n_circ,
            ring_segments=args.ring_segments, rib_visual=bool(args.rib_visual),
            no_fabric=args.no_fabric, solid_hoop=args.solid_hoop,
            static=args.static,
            meshes=meshes, tris=tris,
            build_s=round(build_s, 1), phys_ms=round(phys_ms, 2),
            both_ms=round(both_ms, 2), render_ms=round(both_ms - phys_ms, 2),
            gpu_mb=gpu_mb())
os.makedirs(args.out, exist_ok=True)
json.dump(info, open(os.path.join(args.out, f"{args.tag}.json"), "w"), indent=2)
print("[bench] " + json.dumps(info), flush=True)
app.close()
