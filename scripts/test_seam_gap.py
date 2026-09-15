"""Does the fabric pop away from the hoop on play? Measure the seam gap.

Reported symptom: no gap while stopped, a gap between ring and cloth as soon as
play starts. That is depenetration -- the hoop's collider pushing out the very
vertices the seam is holding -- and it happens because
create_auto_deformable_attachment leaves collisionFilteringOffset at its
schema default of -inf, so no vertex ever qualifies for filtering.

Measure: for the cloth vertices nearest a hoop plane, how far are they from the
hoop's centreline radius (0.20 m)? At rest they are built exactly on it.

    --legacy   use the stock helper (broken defaults)
    default    use create_seam with real offsets
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--legacy", action="store_true")
ap.add_argument("--n-rings", type=int, default=8)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--ring-thickness", type=float, default=0.002)
ap.add_argument("--steps", type=int, default=600)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

import duct_sim.builder as B  # noqa: E402
from duct_sim.builder import min_segments, spawn_duct  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

if args.legacy:
    def _legacy(stage, seam_path, cloth_path, rigid_path, **kw):
        return deformableUtils.create_auto_deformable_attachment(
            stage, target_attachment_path=Sdf.Path(seam_path),
            attachable0_path=Sdf.Path(cloth_path),
            attachable1_path=Sdf.Path(rigid_path))
    B.create_seam = _legacy
    print("[gap] using the STOCK helper (defaults: filtering -inf, overlap 0)",
          flush=True)
else:
    print("[gap] using create_seam (filtering 0.012, overlap 0.006)", flush=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr().Set(9.81)
scene_prim = stage.GetPrimAtPath("/World/physicsScene")
scene_prim.ApplyAPI("PhysxSceneAPI")
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
scene_prim.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                           Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

spec = DuctSpec()
spec = dataclasses.replace(spec, ring_thickness=args.ring_thickness,
                           ring_density=7800.0)
spec = dataclasses.replace(spec,
                           ring_segments=max(spec.ring_segments,
                                             min_segments(spec.ring_thickness, spec.radius)))

MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=2e4, surface_shear_stiffness=2.0,
    surface_bend_stiffness=1e-3)

root, rings, n_ok, n_bound = spawn_duct(
    stage, 0, args.n_rings, args.spacing, spec, MAT, origin=(0, 0, 0.30))
print(f"[gap] seam elements: {n_bound}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

from duct_sim.plastic import _fabric_points  # noqa: E402

SLEEVE = f"{root}/sleeve_000/simMesh"
span = args.spacing * (args.n_rings - 1)
x_ring = -span / 2      # the hoop sleeve_000 starts on


def seam_radius():
    """Radius of the cloth ring that is SEWN to the hoop at x_ring."""
    p = _fabric_points(SLEEVE)
    if p is None or not len(p):
        return None
    sel = p[np.abs(p[:, 0] - x_ring) < args.spacing * 0.25]
    if not len(sel):
        return None
    c = sel.mean(axis=0)
    r = np.linalg.norm(sel[:, 1:] - c[1:], axis=1)
    return float(r.mean()), float(r.std()), len(sel)


marks = {0, 60, 150, 300, args.steps}
for step in range(args.steps + 1):
    if step in marks:
        sr = seam_radius()
        if sr:
            m, sd, n = sr
            print(f"[gap] t={step:4d}  seam radius {m:.4f} m  "
                  f"offset {(m-spec.radius)*1e3:+6.1f} mm  spread {sd*1e3:4.1f} mm",
                  flush=True)
    sim.step(render=False)

print("[gap] END", flush=True)
app.close()
