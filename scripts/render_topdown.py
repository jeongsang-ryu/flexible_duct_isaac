"""Top-down frames of the duct being grabbed by the middle and pulled sideways.

Renders straight to PNGs from a headless Kit, so the bending can be inspected
without sitting in front of the GUI. Top-down is the informative view for this
question: the duct lies along X and the grab pulls along Y, so the whole
deformation happens in the image plane -- in a perspective view the curve is
partly hidden by foreshortening.

The grab is a scripted kinematic pull on one middle hoop rather than a mouse
drag, because a mouse drag cannot be replayed identically and this needs to be
comparable between runs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/duct_topdown")
ap.add_argument("--settle", type=int, default=120)
ap.add_argument("--pull-steps", type=int, default=150)
ap.add_argument("--release-steps", type=int, default=150)
ap.add_argument("--pull-speed", type=float, default=0.5, help="m/s sideways")
ap.add_argument("--every", type=int, default=15)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 1280})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.build_a import build_duct_a  # noqa: E402
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

light = UsdLux.DistantLight.Define(stage, "/World/key")
light.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 20))
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(600.0)

ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

spec = DuctSpec(layout="floor")
info = build_duct_a(stage, "/World/DuctA", spec)
rings = info["rings"]

from duct_sim.skin import DuctSkin  # noqa: E402

skin = DuctSkin(stage, "/World/DuctA/fabric_visual", rings, spec)

# top-down camera
cam = UsdGeom.Camera.Define(stage, "/World/topcam")
cx = UsdGeom.Xformable(cam)
cx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 3.4))
cx.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))   # looking down -Z by default
cam.CreateFocalLengthAttr(18.0)

from isaacsim.core.api import SimulationContext  # noqa: E402

sim = SimulationContext(stage_units_in_meters=1.0)
sim.initialize_physics()
sim.play()

os.makedirs(args.out, exist_ok=True)
rp = rep.create.render_product("/World/topcam", (1280, 1280))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=args.out, rgb=True)
writer.attach([rp])

from isaacsim.core.prims import RigidPrim  # noqa: E402

mid = len(rings) // 2
grab = RigidPrim(rings[mid])
grab.initialize()

frame = 0


def advance(n, pull=False):
    global frame
    for i in range(n):
        if pull:
            grab.set_velocities(
                np.array([[0.0, args.pull_speed, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            )
        sim.step(render=True)
        skin.update()
        if i % args.every == 0:
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
            frame += 1


print("[topdown] settling...", flush=True)
advance(args.settle)
print("[topdown] pulling the middle hoop sideways...", flush=True)
advance(args.pull_steps, pull=True)
print("[topdown] released, letting it settle...", flush=True)
advance(args.release_steps)

cache = UsdGeom.XformCache()
pos = np.array([
    list(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation())
    for p in rings
])
print(f"[topdown] wrote ~{frame} frames to {args.out}", flush=True)
print(f"[topdown] final centreline y: {np.round(pos[:,1],3)}", flush=True)
sys.stdout.flush()
app.close()
