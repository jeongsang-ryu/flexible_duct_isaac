"""Cloth dropped onto a ball, in the GUI. The simplest possible cloth scene.

WHY THIS SCENE. Everything harder has failed today, and the failures were never
where I said they were. A square sheet falling on a sphere removes the duct, the
hoops, the attachments and the seams: if this drapes, PhysX cloth works on this
install and the duct code is what is wrong; if it does not, nothing built on top
of it could ever have worked.

Built the way NVIDIA's own SurfaceDeformableDemo builds its cloth --
create_auto_surface_deformable_hierarchy for the body and
add_surface_deformable_material for the fabric -- rather than applying schemas
by hand, which is how the earlier attempts went wrong.

Run it with:

    conda activate hmclab6
    cd ~/hmcl_issac_project/duct_sim
    python scripts/cloth_on_ball_gui.py

The window takes a while on first launch (shader cache). Physics starts by
itself; the sheet should fall and wrap the ball.
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

# Deformables are GPU-only, and the demo also raises the surface-contact budget
# and the step rate. All three are set here because a deformable that is missing
# any of them does not simulate and does not report an error either.
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

ball = UsdGeom.Sphere.Define(stage, "/World/ball")
ball.CreateRadiusAttr(0.35)
UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.35))
ball.CreateDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.05, 0.05)])
UsdPhysics.CollisionAPI.Apply(ball.GetPrim())

# the sheet, starting above the ball
N, SIZE, H = 40, 1.6, 1.3
pts, tris = [], []
for j in range(N + 1):
    for i in range(N + 1):
        pts.append([-SIZE / 2 + SIZE * i / N, -SIZE / 2 + SIZE * j / N, H])
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
print(f"[ball] deformable hierarchy: {ok}", flush=True)

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

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
# SimulationContext rewrites the scene during init and turns GPU dynamics back
# off when the device is CPU; re-assert it after, and keep device="cuda" above.
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

print("[ball] running. The sheet should fall and wrap the ball.", flush=True)
while app.is_running():
    sim.step(render=True)

app.close()
