"""Drape the duct over a raised bar and measure how far it sags.

WHY THIS TEST IS BETTER THAN PUSHING IT ON THE FLOOR. The floor test kept
answering the wrong question: the hoops are wheels, so a sideways push makes
the duct ROLL, and every result was dominated by rolling resistance rather than
by how easily the duct bends. Suspending it removes friction, rolling and
ground contact from the measurement entirely -- gravity alone does the bending,
so what is left is purely the joint compliance.

The duct starts straight and horizontal, resting across a bar. If it is
compliant it drapes over the bar and the two ends fall; if the joints are too
stiff it stays a straight rod balanced on the bar.

    sag = (height of the bar contact) - (height of the lowest hoop)

A duct of length L draped over a bar has a maximum possible sag of about L/2
per side; reporting sag as a FRACTION of that makes the number comparable
across duct lengths instead of being a bare metre reading.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=float, default=4.0)
ap.add_argument("--bar-height", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--render", default="", help="directory for top-down/side frames")
ap.add_argument("--stiffness", type=float, nargs="*",
                default=[20.0, 5.0, 1.0, 0.2, 0.05])
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.build_a import build_duct_a  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

print(f"\nduct length {args.length} m over a bar at z={args.bar_height} m", flush=True)
print("\n stiff | sag_m  sag_frac | end_z  | verdict", flush=True)
print("-" * 52, flush=True)

for stiff in args.stiffness:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)

    UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(500.0)

    # floor far below, only so the ends have somewhere to land
    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    ggx = UsdGeom.Xformable(g)
    ggx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ggx.AddScaleOp().Set(Gf.Vec3f(12, 12, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    # the bar: a static cylinder across the duct's path (along Y)
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
    bx = UsdGeom.Xformable(bar)
    bx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.bar_height))
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())   # collider, no rigid body = static

    spec = DuctSpec(length=args.length, layout="floor")
    info = build_duct_a(stage, "/World/DuctA", spec,
                        bend_limit_deg=30.0, bend_stiffness=stiff,
                        bend_damping=max(stiff * 0.2, 0.05))
    rings = info["rings"]

    # lift the whole duct so it starts resting across the bar
    lift = args.bar_height + 0.06 + spec.radius + spec.ring_thickness
    for i, rp in enumerate(rings):
        x, _, _ = spec.ring_center(i)
        UsdGeom.Xformable(stage.GetPrimAtPath(rp)).GetOrderedXformOps()[0].Set(
            Gf.Vec3d(x, 0.0, lift)
        )

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()
    for _ in range(args.steps):
        sim.step(render=False)

    c = UsdGeom.XformCache()
    pos = np.array([
        list(c.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation())
        for p in rings
    ])
    top = pos[:, 2].max()
    sag = top - pos[:, 2].min()
    max_sag = args.length * 0.5
    frac = sag / max_sag
    verdict = ("RIGID ROD" if frac < 0.05
               else "drapes" if frac > 0.3 else "partial")
    print(f"{stiff:6.2f} | {sag:5.3f}  {frac:8.2f} | {pos[0,2]:6.3f} | {verdict}",
          flush=True)
    sim.stop()

sys.stdout.flush()
app.close()
