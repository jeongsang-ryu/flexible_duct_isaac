"""Horizontal cloth clamped along ONE edge -- does it behave like fabric?

The hanging test proved the attachment holds. This asks the next question:
does the material actually behave like cloth rather than like a stiff plate?
A horizontal sheet gripped at one edge is the clearest way to see it -- gravity
has to bend it along its whole length, so the answer is the SHAPE of the droop:

  * fabric   -> the free edge falls quickly and the sheet curls, often with
                the far corners drooping more than the middle
  * plate    -> it stays nearly flat and sags only slightly at the tip
  * no grip  -> the whole thing drops to the floor (attachment bound nothing)

Only the near edge is clamped, so everything else is free to move.

Run it with:

    conda activate hmclab6
    cd ~/hmcl_issac_project/duct_sim
    python scripts/cloth_cantilever_gui.py

    --stretch / --bend / --shear   override the fabric stiffnesses
    --size / --res                 sheet size [m] and grid resolution
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=float, default=1.6, help="sheet side length [m]")
ap.add_argument("--res", type=int, default=40, help="grid divisions per side")
ap.add_argument("--height", type=float, default=1.6)
ap.add_argument("--stretch", type=float, default=1.0e4)
ap.add_argument("--bend", type=float, default=1.0e-1)
ap.add_argument("--shear", type=float, default=1.0e2)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
sp = stage.GetPrimAtPath("/World/physicsScene")
sp.ApplyAPI("PhysxSceneAPI")
sp.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
sp.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
sp.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
sp.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                   Sdf.ValueTypeNames.UInt).Set(4 * 1048576)

UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)

ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

# --- HORIZONTAL sheet: clamped edge at x=0, free end reaching out to +x ---
N, SIZE, H = args.res, args.size, args.height
pts, tris = [], []
for j in range(N + 1):
    for i in range(N + 1):
        pts.append([SIZE * i / N, -SIZE / 2 + SIZE * j / N, H])
for j in range(N):
    for i in range(N):
        a = j * (N + 1) + i
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        tris += [[a, b, c], [a, c, d]]
pts = np.array(pts)
tris = np.array(tris, dtype=np.int32)

root = "/World/cloth"
skin = f"{root}/skin"
UsdGeom.Xform.Define(stage, root)
mesh = UsdGeom.Mesh.Define(stage, skin)
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.flatten().tolist()))
mesh.CreateDoubleSidedAttr(True)
mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.80, 0.10)])

ok = deformableUtils.create_auto_surface_deformable_hierarchy(
    stage,
    root_prim_path=root,
    simulation_mesh_path=f"{root}/simMesh",
    cooking_src_mesh_path=skin,
    cooking_src_simplification_enabled=False,
    set_visibility_with_guide_purpose=True,
)
print(f"[cantilever] deformable hierarchy: {ok}", flush=True)

rp = stage.GetPrimAtPath(root)
rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
for name, val in (("physxDeformableBody:selfCollision", True),
                  ("physxDeformableBody:solverPositionIterationCount", 16),
                  ("physxDeformableBody:collisionPairUpdateFrequency", 4),
                  ("physxDeformableBody:collisionIterationMultiplier", 4)):
    attr = rp.GetAttribute(name)
    if attr and attr.IsValid():
        attr.Set(val)

mat = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, mat,
    dynamic_friction=0.5,
    surface_thickness=0.005,
    surface_stretch_stiffness=args.stretch,
    surface_shear_stiffness=args.shear,
    surface_bend_stiffness=args.bend,
)
physicsUtils.add_physics_material_to_prim(stage, rp, mat)

# --- the clamp: static rod straddling ONLY the x=0 edge ---
# Thin in x (0.08 m) so it grips the first column of vertices and nothing
# further along the sheet -- a wider rod would clamp several columns and
# stiffen the root artificially, making the cloth look less floppy than it is.
rod = physicsUtils.add_rigid_box(
    stage, "/World/clamp",
    size=Gf.Vec3f(0.08, SIZE * 1.05, 0.08),
    position=Gf.Vec3f(0.0, 0.0, H),
    density=0.0,
    color=Gf.Vec3f(0.05, 0.05, 0.05),
)

made = deformableUtils.create_auto_deformable_attachment(
    stage,
    target_attachment_path=Sdf.Path(f"{root}/attachment"),
    attachable0_path=Sdf.Path(root),
    attachable1_path=rod.GetPath(),
)
print(f"[cantilever] clamp at x=0 (static), attachment -> {made}", flush=True)
print(f"[cantilever] sheet {SIZE} m x {SIZE} m at z={H}, free end at x={SIZE}",
      flush=True)
print(f"[cantilever] stretch={args.stretch:g} bend={args.bend:g} "
      f"shear={args.shear:g}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

print("[cantilever] running. The free end should droop and curl like fabric.",
      flush=True)
while app.is_running():
    sim.step(render=True)

app.close()
