"""How much does the rigid duct keep moving after it should have settled?

"It heaves around like a worm" is a real complaint but not a measurable one, so
this turns it into a number: drop the duct, wait, then watch how much motion is
left. A settled duct has none.

    residual = mean |linear velocity| over all bodies, m/s

Sweeps the parameters that plausibly matter and prints a table, so the choice
is made on measurement rather than on which value felt better.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=float, default=6.0)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--settle", type=int, default=900)
ap.add_argument("--watch", type=int, default=300)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.builder import spawn_duct_rigid  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

spec = DuctSpec()
n = max(2, int(round(args.length / args.spacing)) + 1)

CASES = [
    # mass_per_m, stiffness, damping, body_damping, bend_limit
    (0.5,   8.0,  2.0, 0.5, 14.0),
    (0.5,  50.0, 10.0, 0.5, 14.0),
    (0.5,  50.0, 10.0, 2.0, 10.0),
    (0.5, 200.0, 20.0, 1.0, 10.0),
    (0.5, 200.0, 40.0, 2.0,  8.0),
    (0.5, 600.0, 60.0, 2.0,  8.0),
]

print(f"\n{'mass':>5} {'stiff':>7} {'damp':>6} {'body':>5} {'limit':>6} "
      f"{'residual m/s':>13} {'sag m':>7}", flush=True)
print("-" * 60, flush=True)

for ci, (mpm, stiff, damp, bdamp, limit) in enumerate(CASES):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr().Set(9.81)
    sp = stage.GetPrimAtPath("/World/physicsScene")
    sp.ApplyAPI("PhysxSceneAPI")
    sp.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    sp.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
    sp.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)

    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(g)
    gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(30, 30, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    st = [(-args.length / 2 + args.spacing * k, 0.0, 0.0) for k in range(n)]
    _, paths, _, _ = spawn_duct_rigid(
        stage, 0, st, spec, bend_limit_deg=limit, stiffness=stiff,
        damping=damp, mass_per_m=mpm, body_damping=bdamp, verbose=False)

    sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
    sim.initialize_physics()
    sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
    sim.play()
    for _ in range(args.settle):
        sim.step(render=False)

    from isaacsim.core.prims import RigidPrim  # noqa: E402
    view = RigidPrim(paths)
    speeds = []
    zs = []
    for _ in range(args.watch):
        sim.step(render=False)
        v = view.get_velocities()
        v = np.asarray(v.cpu() if hasattr(v, "cpu") else v)
        speeds.append(float(np.abs(v[:, :3]).mean()))
        p, _r = view.get_world_poses()
        p = np.asarray(p.cpu() if hasattr(p, "cpu") else p)
        zs.append(float(p[:, 2].max() - p[:, 2].min()))

    print(f"{mpm:5.2f} {stiff:7.1f} {damp:6.1f} {bdamp:5.1f} {limit:6.1f} "
          f"{np.mean(speeds):13.5f} {np.mean(zs):7.4f}", flush=True)

    sim.stop()
    sim.clear_instance()

print("\nlower residual = settles instead of heaving. sag is the z spread,\n"
      "i.e. how far the duct droops; too stiff shows up as ~0.", flush=True)
app.close()
