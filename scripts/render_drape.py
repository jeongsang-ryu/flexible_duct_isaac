"""Side-view frames of the duct draping over a bar -- the flexibility evidence.

This is the picture that answers "does it bend naturally", because gravity is
the only thing acting: no friction, no rolling, no ground contact to confound
it. Rendered from the side, since the sag is a vertical shape.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/duct_drape")
ap.add_argument("--length", type=float, default=4.0)
ap.add_argument("--bar-height", type=float, default=2.0)
ap.add_argument("--stiffness", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=420)
ap.add_argument("--every", type=int, default=14)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.build_a import build_duct_a  # noqa: E402
from duct_sim.skin import DuctSkin  # noqa: E402
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

# BAR RADIUS 0.06, NOT 0.04. The hoops are 0.1 m apart and 0.016 m thick, so
# the gap between consecutive hoop surfaces is 0.084 m. A 0.04 m-radius bar is
# 0.08 m across and fits through that gap with 4 mm to spare -- the duct could
# slide sideways and drop off the bar mid-test. A real duct's fabric spans the
# gap, but approach A's fabric is not a collider, so the geometry has to avoid
# the situation instead. 0.06 m radius = 0.12 m across cannot pass.
bar = UsdGeom.Capsule.Define(stage, "/World/bar")
bar.CreateAxisAttr("Y")
bar.CreateRadiusAttr(0.06)   # see note below
bar.CreateHeightAttr(2.0)
UsdGeom.Xformable(bar).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.bar_height))
UsdPhysics.CollisionAPI.Apply(bar.GetPrim())

spec = DuctSpec(length=args.length, layout="floor")
info = build_duct_a(stage, "/World/DuctA", spec,
                    bend_limit_deg=30.0, bend_stiffness=args.stiffness,
                    bend_damping=max(args.stiffness * 0.2, 0.05))
rings = info["rings"]
skin = DuctSkin(stage, "/World/DuctA/fabric_visual", rings, spec)

lift = args.bar_height + 0.06 + spec.radius + spec.ring_thickness
for i, rp in enumerate(rings):
    x, _, _ = spec.ring_center(i)
    UsdGeom.Xformable(stage.GetPrimAtPath(rp)).GetOrderedXformOps()[0].Set(
        Gf.Vec3d(x, 0.0, lift)
    )

# side camera: looking along -Y at the XZ plane, where the sag happens
cam = UsdGeom.Camera.Define(stage, "/World/sidecam")
cx = UsdGeom.Xformable(cam)
cx.AddTranslateOp().Set(Gf.Vec3d(0.0, -7.5, 1.3))
cx.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
cam.CreateFocalLengthAttr(24.0)

sim = SimulationContext(stage_units_in_meters=1.0)
sim.initialize_physics()
sim.play()

os.makedirs(args.out, exist_ok=True)
rp_prod = rep.create.render_product("/World/sidecam", (1280, 720))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=args.out, rgb=True)
writer.attach([rp_prod])

frames = 0
for i in range(args.steps):
    sim.step(render=True)
    skin.update()
    if i % args.every == 0:
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        frames += 1

c = UsdGeom.XformCache()
pos = np.array([
    list(c.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation())
    for p in rings
])
print(f"[drape] stiffness={args.stiffness} frames={frames} -> {args.out}", flush=True)
print(f"[drape] top z={pos[:,2].max():.3f}  lowest z={pos[:,2].min():.3f}  "
      f"sag={pos[:,2].max()-pos[:,2].min():.3f} m", flush=True)
sys.stdout.flush()
app.close()
