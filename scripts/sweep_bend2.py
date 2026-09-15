"""Second sweep: pull with a FORCE, not a prescribed velocity.

The first sweep concluded "rigid slide" at every stiffness down to and
including ZERO, which cannot be a stiffness problem -- a chain with no
restoring torque is floppy by construction. The measurement was creating the
result: `set_velocities` every step pins the grabbed hoop to an exact velocity,
and a velocity constraint is effectively infinitely strong, so the
translation-locked joints drag the neighbours along faster than ground friction
(~0.78 N per hoop) can hold them back. Nothing can bend against an infinitely
stiff pull.

A mouse drag applies a FORCE toward the cursor, which is finite and therefore
lets friction resist -- so that is what is applied here, and the force
magnitude is swept alongside stiffness because the two interact: the question
is not "is the duct floppy" but "is the pull comparable to the friction that
holds the rest of it down".
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
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

spec_probe = DuctSpec(layout="floor")
print(f"\nhoops={spec_probe.num_rings}  (friction budget per hoop ~0.78 N)", flush=True)
print("\n stiff  force_N |  spread   grabbed_dy  ends_dy  ratio | verdict", flush=True)
print("-" * 68, flush=True)

CASES = [
    (20.0, 2.0), (20.0, 8.0),
    (1.0, 2.0), (1.0, 8.0),
    (0.2, 2.0), (0.2, 8.0), (0.2, 20.0),
    (0.05, 8.0),
]

for stiff, force in CASES:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(g)
    gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    spec = DuctSpec(layout="floor")
    info = build_duct_a(stage, "/World/DuctA", spec,
                        bend_limit_deg=30.0, bend_stiffness=stiff,
                        bend_damping=max(stiff * 0.2, 0.02))
    rings = info["rings"]

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()

    def pos():
        c = UsdGeom.XformCache()
        return np.array([
            list(c.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation())
            for p in rings
        ])

    for _ in range(120):
        sim.step(render=False)
    before = pos()

    mid = len(rings) // 2
    grab = RigidPrim(rings[mid])
    grab.initialize()
    f = np.array([[0.0, force, 0.0]], dtype=np.float32)
    for _ in range(240):
        grab.apply_forces(f, is_global=True)
        sim.step(render=False)
    for _ in range(120):
        sim.step(render=False)
    after = pos()

    dy = after[:, 1] - before[:, 1]
    spread = dy.max() - dy.min()
    grabbed = dy[mid]
    ends = 0.5 * (abs(dy[0]) + abs(dy[-1]))
    ratio = spread / max(abs(grabbed), 1e-6)
    verdict = ("no motion" if abs(grabbed) < 0.02
               else "RIGID SLIDE" if ratio < 0.15
               else "CURVES" if ratio > 0.45 else "partial")
    print(f"{stiff:6.2f} {force:8.1f} | {spread:7.4f}  {grabbed:9.4f}  {ends:7.4f}"
          f"  {ratio:5.2f} | {verdict}", flush=True)
    sim.stop()

sys.stdout.flush()
app.close()
