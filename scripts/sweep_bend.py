"""Find the joint stiffness that makes the duct BEND rather than slide.

The first floor run pulled one middle hoop 1.14 m sideways and every hoop
followed to within 2 mm -- the duct translated as a rigid rod. That is a
stiffness problem, not a friction one: with the default drive the chain resists
bending far more than the ground resists sliding, so the cheapest way for the
duct to satisfy the pull is to move as one piece.

Measured here is the CENTRELINE SPREAD -- max(y) - min(y) over the hoops after
the pull. A rigid slide gives ~0; a duct that curves gives a spread comparable
to how far the grabbed hoop was pulled. One Kit boot covers the whole sweep
because booting costs far more than simulating.
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

CASES = [
    # (stiffness, damping, limit_deg)
    (20.0, 4.0, 12.0),    # current default -- the rigid-slide baseline
    (5.0, 1.0, 20.0),
    (1.0, 0.3, 25.0),
    (0.2, 0.1, 30.0),
    (0.05, 0.03, 35.0),
    (0.0, 0.02, 40.0),    # no restoring torque, damping only
]

print("\n stiff  damp  limit |  spread   grabbed_dy  ends_dy | verdict", flush=True)
print("-" * 66, flush=True)

for stiff, damp, lim in CASES:
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
                        bend_limit_deg=lim, bend_stiffness=stiff, bend_damping=damp)
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
    for _ in range(150):
        grab.set_velocities(np.array([[0.0, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        sim.step(render=False)
    for _ in range(90):
        sim.step(render=False)
    after = pos()

    dy = after[:, 1] - before[:, 1]
    spread = dy.max() - dy.min()
    grabbed = dy[mid]
    ends = 0.5 * (abs(dy[0]) + abs(dy[-1]))
    verdict = ("RIGID SLIDE" if spread < 0.1 * max(abs(grabbed), 1e-6)
               else "curves" if spread > 0.4 * max(abs(grabbed), 1e-6)
               else "partial")
    print(f"{stiff:6.2f} {damp:5.2f} {lim:6.1f} | {spread:7.4f}  {grabbed:9.4f}  "
          f"{ends:7.4f} | {verdict}", flush=True)

    sim.stop()

sys.stdout.flush()
app.close()
