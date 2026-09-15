"""Can cloth be attached to a rigid body? Built on the scene that already works.

The cloth-on-ball test drapes correctly, so PhysX cloth is fine on this install.
This extends THAT scene by exactly one thing -- a static rod through the top of
the sheet, with an attachment -- instead of returning to the duct code that has
never rendered. If the sheet hangs from the rod, attachment works and the duct
failure is somewhere else entirely; if it falls to the floor, attachment is the
missing piece and this is the smallest scene that shows it.

TWO THINGS TAKEN FROM NVIDIA's DeformableAttachmentsDemo, both of which the
duct code got wrong:

  * the rigid partner is created with density=0.0, which makes it a STATIC
    collider. A dynamic body would be dragged around by the cloth instead of
    holding it.
  * the deformable OVERLAPS the rigid body. In the demo the blob is centred at
    z=3.2 and the cube spans 4.5..5.5, so they intersect.
    create_auto_deformable_attachment binds the deformable vertices that lie
    inside the rigid body's collision shape -- with no overlap it binds nothing,
    returns True anyway, and the cloth simply drops.

Run it with:

    conda activate hmclab6
    cd ~/hmcl_issac_project/duct_sim
    python scripts/cloth_attach_gui.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

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
gx.AddScaleOp().Set(Gf.Vec3f(6, 6, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

# --- the sheet, hanging in the air ---
N, SIZE, H = 40, 1.6, 1.6
pts, tris = [], []
for j in range(N + 1):
    for i in range(N + 1):
        # vertical sheet in the XZ plane so it can hang from a rod along X
        pts.append([-SIZE / 2 + SIZE * i / N, 0.0, H - SIZE * j / N])
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
print(f"[attach] deformable hierarchy: {ok}", flush=True)

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
    surface_stretch_stiffness=1.0e4,
    surface_shear_stiffness=1.0e2,
    surface_bend_stiffness=1.0e-1,
)
physicsUtils.add_physics_material_to_prim(stage, rp, mat)

# --- the rod: STATIC (density 0) and INTERSECTING the sheet's top edge ---
# Sized generously in Y and Z so the top row of cloth vertices is unambiguously
# inside it. An attachment that grabs nothing is silent, so the geometry has to
# make the overlap obvious rather than marginal.
rod = physicsUtils.add_rigid_box(
    stage, "/World/rod",
    size=Gf.Vec3f(SIZE * 1.05, 0.10, 0.10),
    position=Gf.Vec3f(0.0, 0.0, H),
    density=0.0,
    color=Gf.Vec3f(0.05, 0.05, 0.05),
)
print(f"[attach] rod is static (density 0) at z={H}, sheet top row at z={H}",
      flush=True)

made = deformableUtils.create_auto_deformable_attachment(
    stage,
    target_attachment_path=Sdf.Path(f"{root}/attachment"),
    attachable0_path=Sdf.Path(root),
    attachable1_path=rod.GetPath(),
)
print(f"[attach] create_auto_deformable_attachment -> {made}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

print("[attach] running. The sheet should HANG from the rod, not fall.",
      flush=True)
while app.is_running():
    sim.step(render=True)

app.close()
