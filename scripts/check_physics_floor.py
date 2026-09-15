"""Does the FLOOR-layout duct rest on the ground and dent where you push it?

The hanging checks do not transfer: there the useful question was "does the
free end swing while the anchor holds", here both ends are free and the
questions are whether it settles ON the floor rather than through or above it,
and whether a push deforms the duct LOCALLY instead of shoving the whole thing.

That last one is the point of the floor layout, so it is measured explicitly:
push one middle hoop sideways, then compare how far that hoop moved against how
far the far ends moved. If everything moves together the duct is behaving like
a rigid pipe on ice, which is exactly the failure this layout exists to avoid.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.build_a import build_duct_a  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402

report = []


def positions(stage, paths):
    cache = UsdGeom.XformCache()
    return np.array([
        list(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation())
        for p in paths
    ])


ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)

ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

spec = DuctSpec(layout="floor")
info = build_duct_a(stage, "/World/DuctA", spec)
rings = info["rings"]

from isaacsim.core.api import SimulationContext  # noqa: E402

sim = SimulationContext(stage_units_in_meters=1.0)
sim.initialize_physics()
sim.play()

for _ in range(180):
    sim.step(render=False)
settled = positions(stage, rings)

# --- 1. resting ON the floor, not through it or floating ---
expect_z = spec.radius + spec.ring_thickness
dz = settled[:, 2] - expect_z
report.append((
    np.abs(dz).max() < 0.05 and settled[:, 2].min() > 0.0,
    "rests on the floor",
    f"ring centre z: mean {settled[:,2].mean():.4f} m (expect {expect_z:.4f}), "
    f"min {settled[:,2].min():.4f}, max |offset| {np.abs(dz).max():.4f}",
))

# --- 2. still a duct: spacing preserved ---
gaps = np.linalg.norm(np.diff(settled, axis=0), axis=1)
report.append((abs(gaps.mean() - spec.ring_spacing) < 0.02,
               "hoop spacing held on the floor",
               f"mean {gaps.mean():.4f} m, std {gaps.std():.4f}"))

# --- 3. a push deforms LOCALLY, not globally ---
from isaacsim.core.prims import RigidPrim  # noqa: E402

mid = len(rings) // 2
pusher = RigidPrim(rings[mid])
pusher.initialize()
for _ in range(30):
    pusher.set_velocities(np.array([[0.0, 1.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    sim.step(render=False)
for _ in range(60):
    sim.step(render=False)
pushed = positions(stage, rings)

d = np.linalg.norm(pushed - settled, axis=1)
local, ends = d[mid], max(d[0], d[-1])
report.append((
    local > 0.03 and local > 2.0 * max(ends, 1e-6),
    "push deforms locally, not as a rigid body",
    f"pushed hoop moved {local:.4f} m, far ends moved {ends:.4f} m "
    f"(ratio {local / max(ends, 1e-6):.1f}x)",
))

# --- 4. the deformation is a smooth bend, not a kink ---
# second difference along the chain: a hinge at one joint would spike
prof = d - d.min()
curv = np.abs(np.diff(prof, 2))
report.append((curv.max() < 0.5 * max(prof.max(), 1e-6),
               "bend is smooth, no single-joint kink",
               f"max |2nd diff| {curv.max():.4f} vs profile peak {prof.max():.4f}"))

print(flush=True)
ok = True
for good, label, detail in report:
    print(f"[{'OK' if good else 'FAIL'}] {label}: {detail}", flush=True)
    ok &= good
print(flush=True)
print("FLOOR_OK" if ok else "FLOOR_FAIL", flush=True)
sys.stdout.flush()
app.close()
sys.exit(0 if ok else 1)
