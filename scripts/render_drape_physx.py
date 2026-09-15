"""Drape render for approach B implemented with native PhysX surface deformable.

Unlike the Newton version there is nothing to step by hand: the cloth, the
hoops, the bar and the ground are all in one PhysX scene, so sim.step() moves
everything and the mesh updates itself.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/duct_physx")
ap.add_argument("--length", type=float, default=4.0)
ap.add_argument("--bar-height", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=420)
ap.add_argument("--every", type=int, default=14)
ap.add_argument("--headless-only", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.build_b_physx import build_duct_b_physx  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
sc.CreateGravityMagnitudeAttr().Set(9.81)

key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 35))
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(700.0)

g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
ggx = UsdGeom.Xformable(g)
ggx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
ggx.AddScaleOp().Set(Gf.Vec3f(14, 14, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

bar = UsdGeom.Capsule.Define(stage, "/World/bar")
bar.CreateAxisAttr("Y")
bar.CreateRadiusAttr(0.06)
bar.CreateHeightAttr(2.0)
UsdGeom.Xformable(bar).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.bar_height))
UsdPhysics.CollisionAPI.Apply(bar.GetPrim())

spec = DuctSpec(length=args.length, layout="floor")

# Position the hoops on the bar BEFORE building the cloth: the sleeve is
# generated from the hoops' real poses, so it has to see them where they will
# actually start.
from duct_sim.build_a import build_duct_a  # noqa: E402

probe = build_duct_a(stage, "/World/_probe", spec)
for rp in probe["rings"]:
    stage.RemovePrim(rp)
stage.RemovePrim("/World/_probe")

lift = args.bar_height + 0.06 + spec.radius + spec.ring_thickness


class _Shifted(DuctSpec):
    pass


shifted = DuctSpec(length=args.length, layout="floor")
info = build_duct_b_physx(stage, "/World/DuctB", shifted)
for i, rp in enumerate(info["rings"]):
    x, _, _ = shifted.ring_center(i)
    UsdGeom.Xformable(stage.GetPrimAtPath(rp)).GetOrderedXformOps()[0].Set(
        Gf.Vec3d(x, 0.0, lift))
# move the sleeve with them, since it was authored at floor height
# The sleeve is a hierarchy now (root Xform + cooked sim mesh + skin), so it is
# moved by translating its ROOT rather than by rewriting point arrays -- editing
# points would desync the skin from the cooked simulation mesh.
from pxr import Vt  # noqa: E402

cloth_root = info["cloth"].rsplit("/", 1)[0]
UsdGeom.Xformable(stage.GetPrimAtPath(cloth_root)).AddTranslateOp().Set(
    Gf.Vec3d(0.0, 0.0, lift - shifted.drop_to_floor))

print(f"[physx] {spec.describe()}", flush=True)
print(f"[physx] cloth: {info['n_vertices']} verts, {info['n_triangles']} tris, "
      f"{info['n_seams']} seams sewn to hoops", flush=True)

cam = UsdGeom.Camera.Define(stage, "/World/sidecam")
cx = UsdGeom.Xformable(cam)
cx.AddTranslateOp().Set(Gf.Vec3d(0.0, -7.5, 1.3))
cx.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
cam.CreateFocalLengthAttr(24.0)

# device="cuda" IS THE WHOLE POINT, not a performance preference.
# PhysicsContext decides GPU dynamics purely from the device string:
#     if "cuda" in self._device:  enable_gpu_dynamics(True)
#     else:                       enable_gpu_dynamics(False)
# SimulationContext defaults to CPU, so it was actively turning the flag back
# OFF after ensure_gpu_dynamics() set it -- which is why the USD attribute read
# True and PhysX still reported "Deformable Body feature is only supported on
# GPU". Deformables do not run on the CPU pipeline at all.
sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()

# RE-ASSERT GPU DYNAMICS AFTER SimulationContext.
# Measured: the flag reads True right after ensure_gpu_dynamics() and False
# again once initialize_physics() has run -- SimulationContext rewrites the
# scene (it also applies NewtonSceneAPI alongside PhysxSceneAPI). Deformables
# are GPU-only, so the cloth silently refuses to simulate and every seam
# reports "invalid actors", which is indistinguishable from an attachment bug.
from duct_sim.build_b_physx import ensure_gpu_dynamics  # noqa: E402

ensure_gpu_dynamics(stage)
_chk = stage.GetPrimAtPath("/World/physicsScene").GetAttribute(
    "physxScene:enableGPUDynamics").Get()
print(f"[physx] GPU dynamics after SimulationContext = {_chk}", flush=True)

sim.play()

if not args.headless_only:
    import omni.replicator.core as rep  # noqa: E402
    os.makedirs(args.out, exist_ok=True)
    rp_prod = rep.create.render_product("/World/sidecam", (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=args.out, rgb=True)
    writer.attach([rp_prod])

frames = 0
for i in range(args.steps):
    sim.step(render=not args.headless_only)
    if not args.headless_only and i % args.every == 0:
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        frames += 1

pts = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(info["cloth"])).GetPointsAttr().Get())
print(f"[physx] frames={frames}  cloth z {pts[:,2].min():.3f}..{pts[:,2].max():.3f}  "
      f"finite={np.isfinite(pts).all()}", flush=True)
sys.stdout.flush()
app.close()
