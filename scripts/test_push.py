"""Push one hoop and count how many fall over. Capsule ring vs solid disc.

"It dominoes when I push it lightly" needs a push that is the same every time.
A mouse drag is not -- and this build's drag is known to grab a hoop 23-51 cm
from the click, so a hand test cannot separate the duct's behaviour from the
dragger's aim. This applies a fixed impulse to the middle hoop and reports how
far the tilt spreads.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--solid-hoop", action="store_true")
ap.add_argument("--length", type=float, default=3.0)
ap.add_argument("--spacing", type=float, default=0.10)
ap.add_argument("--n-circ", type=int, default=40)
ap.add_argument("--surface-sampling", type=float, default=0.02)
ap.add_argument("--impulse", type=float, default=2.0, help="N*s sideways")
ap.add_argument("--settle", type=int, default=250)
ap.add_argument("--after", type=int, default=400)
ap.add_argument("--tag", default="?")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.builder import spawn_duct_single  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sp = stage.GetPrimAtPath("/World/physicsScene")
sp.ApplyAPI("PhysxSceneAPI")
UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)

g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=3e3, surface_shear_stiffness=1.0,
    surface_bend_stiffness=1e-3)

spec = DuctSpec()
n_rings = int(round(args.length / args.spacing)) + 1
_, rings, _, n_seams = spawn_duct_single(
    stage, 0, n_rings, args.spacing, spec, MAT,
    origin=(0.0, 0.0, 0.30), n_circ=args.n_circ, clearance=-0.004,
    rib_visual=True, solid_hoop=args.solid_hoop,
    surface_sampling=args.surface_sampling, verbose=False)

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

for _ in range(args.settle):
    sim.step(render=False)

# BUILD THE VIEW AFTER WARM-UP. A RigidPrim made before physics has run reports
# authored poses forever and poisons every later read.
view = RigidPrim("/World/duct_00/ring_.*")
view.initialize()


def quats():
    _, q = view.get_world_poses()
    return np.asarray(q.cpu() if hasattr(q, "cpu") else q, dtype=np.float64)


def tilt_deg(q0, q1):
    """Angle between the two orientations, per hoop, in degrees."""
    d = np.abs((q0 * q1).sum(axis=1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


q_before = quats()

mid = len(rings) // 2
one = RigidPrim(rings[mid])
one.initialize()
import torch  # noqa: E402
imp = torch.tensor([[0.0, args.impulse, 0.0]], dtype=torch.float32, device="cuda")
one.apply_forces(imp / (1.0 / 60.0), is_global=True)   # one step of force = impulse

for _ in range(args.after):
    sim.step(render=False)

t = tilt_deg(q_before, quats())
moved = int((t > 10.0).sum())
far = int((t > 30.0).sum())
print(f"[push] {args.tag:16s} solid={args.solid_hoop!s:5s} seams={n_seams:4d} "
      f"impulse={args.impulse} Ns -> pushed hoop {t[mid]:6.1f} deg, "
      f"{moved:3d}/{len(t)} hoops tilted >10deg, {far:3d} >30deg, "
      f"max {t.max():6.1f} deg", flush=True)
app.close()
