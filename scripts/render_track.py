"""Render the whole track from a framed camera, headless.

The GUI viewport is not a reliable way to see this. `get_active_viewport()`
reports the camera was retargeted to /World/cam and the window header still
reads Perspective, so three straight screenshots came back as empty floor while
the physics was running correctly the whole time. A Replicator render product
is bound to the camera prim itself and does not depend on which viewport is
focused, so it shows what is actually there.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True)
ap.add_argument("--out", default="/tmp/track_render")
ap.add_argument("--settle", type=int, default=200)
ap.add_argument("--shrink-rest", type=float, default=0.0)
ap.add_argument("--n-circ", type=int, default=40)
ap.add_argument("--spacing", type=float, default=0.0,
                help="hoop spacing override; 0 = take it from the layout")
ap.add_argument("--clearance", type=float, default=-0.004)
ap.add_argument("--res", type=int, default=1600)
ap.add_argument("--views", default="top,iso",
                help="comma-separated: top, iso, corner")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res, "height": args.res})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.builder import spawn_duct_path  # noqa: E402
from duct_sim.layout import load, resample  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
sp = stage.GetPrimAtPath("/World/physicsScene")
sp.ApplyAPI("PhysxSceneAPI")

key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 20))
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)

ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(44, 44, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=3e3, surface_shear_stiffness=1.0,
    surface_bend_stiffness=1e-3)

doc = load(args.layout)
spec = DuctSpec()
xs, ys = [], []
for i, run in enumerate(doc["runs"]):
    stations = resample(run["points"],
                        args.spacing or doc.get("duct", {}).get("hoop_spacing", 0.05))
    xs += [s[0] for s in stations]
    ys += [s[1] for s in stations]
    spawn_duct_path(stage, i, stations, spec, MAT,
                    n_circ=args.n_circ, clearance=args.clearance,
                    rib_visual=True)

W = max(max(xs) - min(xs), 1.0)
D = max(max(ys) - min(ys), 1.0)
cxm, cym = (max(xs) + min(xs)) * 0.5, (max(ys) + min(ys)) * 0.5
print(f"[render] track spans {W:.1f} x {D:.1f} m about ({cxm:.1f}, {cym:.1f})",
      flush=True)

FOCAL, APER = 24.0, 36.0
fov = 2.0 * math.atan(APER * 0.5 / FOCAL)
# 1.15 leaves a margin so the outermost duct is not clipped by the frame edge
dist = (max(W, D) * 1.15) * 0.5 / math.tan(fov * 0.5)

VIEWS = {
    "top":    (Gf.Vec3d(0.02, -0.05, 1.0), 1.0),
    "iso":    (Gf.Vec3d(0.05, -0.75, 0.66), 1.0),
    "corner": (Gf.Vec3d(-0.72, -0.62, 0.32), 0.85),
}
tgt = Gf.Vec3d(cxm, cym, 0.2)
cams = []
for name in [v.strip() for v in args.views.split(",") if v.strip()]:
    if name not in VIEWS:
        continue
    d, k = VIEWS[name]
    eye = tgt + d.GetNormalized() * dist * k
    path = f"/World/cam_{name}"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(FOCAL)
    cam.CreateHorizontalApertureAttr(APER)
    cam.CreateVerticalApertureAttr(APER)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
    cams.append((name, path))
    print(f"[render] {name}: eye ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f})",
          flush=True)

from isaacsim.core.api import SimulationContext  # noqa: E402

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

for _ in range(args.settle):
    sim.step(render=False)

if args.shrink_rest > 0:
    from duct_sim.plastic import shrink_rest_shape  # noqa: E402
    n = shrink_rest_shape(stage, factor=args.shrink_rest)
    sim.stop()
    sim.play()
    print(f"[render] rest narrowed on {n} sleeve(s), restarted", flush=True)
    for _ in range(args.settle):
        sim.step(render=False)

os.makedirs(args.out, exist_ok=True)
for name, path in cams:
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    rp = rep.create.render_product(path, (args.res, args.res))
    w = rep.WriterRegistry.get("BasicWriter")
    w.initialize(output_dir=d, rgb=True)
    w.attach([rp])

for _ in range(6):
    sim.step(render=True)
rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
rep.orchestrator.wait_until_complete()

print(f"[render] wrote to {args.out}", flush=True)
app.close()
