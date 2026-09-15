"""Does approach A actually behave like a duct? Checks EFFECT, not exit status.

`run_duct.py --steps N` finishing without an exception proves only that no API
call threw -- a chain that exploded to NaN, collapsed into a single point, or
sat frozen in mid-air would all "complete N steps without error" too. This
measures what the duct actually did:

  1. hoop spacing stays at the specified 0.10 m   (are the joints holding?)
  2. nothing is NaN / has flown away               (is the solver stable?)
  3. under a sideways shove the free end moves and the fixed end does not
                                                   (is it anchored and compliant?)
  4. after the shove stops, motion decays          (does the damping settle it?)
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


def ring_positions(stage, paths):
    cache = UsdGeom.XformCache()
    out = []
    for p in paths:
        t = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation()
        out.append([t[0], t[1], t[2]])
    return np.array(out)


ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)

spec = DuctSpec()
info = build_duct_a(stage, "/World/DuctA", spec)
rings = info["rings"]

from isaacsim.core.api import SimulationContext  # noqa: E402

sim = SimulationContext(stage_units_in_meters=1.0)
sim.initialize_physics()
sim.play()

# let it settle under gravity
for _ in range(120):
    sim.step(render=False)
settled = ring_positions(stage, rings)

# --- 1. hoop spacing ---
gaps = np.linalg.norm(np.diff(settled, axis=0), axis=1)
report.append((
    abs(gaps.mean() - spec.ring_spacing) < 0.02 and gaps.std() < 0.01,
    "hoop spacing held",
    f"mean {gaps.mean():.4f} m (spec {spec.ring_spacing:.2f}), std {gaps.std():.4f}",
))

# --- 2. stability ---
finite = np.isfinite(settled).all()
near = np.abs(settled).max() < 10.0
report.append((finite and near,
               "no NaN / no fly-away",
               f"finite={finite}, max|coord|={np.abs(settled).max():.3f} m"))

# --- 3. anchored + compliant: shove the free end sideways ---
from isaacsim.core.prims import RigidPrim  # noqa: E402

free = RigidPrim(rings[-1])
free.initialize()
for _ in range(40):
    free.set_velocities(np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    sim.step(render=False)
shoved = ring_positions(stage, rings)

free_moved = abs(shoved[-1, 0] - settled[-1, 0])
fixed_moved = np.linalg.norm(shoved[0] - settled[0])
report.append((free_moved > 0.05 and fixed_moved < 0.01,
               "free end moves, fixed end does not",
               f"free dx={free_moved:.4f} m, fixed drift={fixed_moved:.5f} m"))

# --- 4. settles after release ---
mid = len(rings) // 2
traj = []
for i in range(300):
    sim.step(render=False)
    if i % 10 == 0:
        traj.append(ring_positions(stage, rings)[mid])
traj = np.array(traj)
early = np.linalg.norm(np.diff(traj[:10], axis=0), axis=1).mean()
late = np.linalg.norm(np.diff(traj[-10:], axis=0), axis=1).mean()
report.append((late < early,
               "oscillation decays after release",
               f"mid-ring motion/sample: early {early:.5f} -> late {late:.5f} m"))

print(flush=True)
ok = True
for good, label, detail in report:
    print(f"[{'OK' if good else 'FAIL'}] {label}: {detail}", flush=True)
    ok &= good
print(flush=True)
print("PHYSICS_OK" if ok else "PHYSICS_FAIL", flush=True)
sys.stdout.flush()
app.close()
sys.exit(0 if ok else 1)
