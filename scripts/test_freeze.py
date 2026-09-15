"""Verify the freeze round-trip: simulate, save, reload, confirm the shape survived."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import math  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from duct_sim.freeze import capture, freeze_to_usd  # noqa: E402
from duct_sim.geometry import create_ring  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

OUT = "/tmp/frozen_duct.usd"

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityMagnitudeAttr().Set(9.81)
sp = stage.GetPrimAtPath("/World/physicsScene")
sp.ApplyAPI("PhysxSceneAPI")
sp.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
sp.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
sp.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
sp.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                   Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
g = UsdGeom.Xformable(ground)
g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
g.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

spec = DuctSpec()
R, TUBE, H = spec.radius, spec.ring_thickness, 1.2
N, SPACING = 8, 0.1
xs = [-SPACING * (N - 1) / 2 + SPACING * k for k in range(N)]
rings = []
for i, x in enumerate(xs):
    p = f"/World/ring_{i:02d}"
    create_ring(stage, p, R, TUBE, n_seg=spec.ring_segments)
    prim = stage.GetPrimAtPath(p)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, H))
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0).GetQuat()))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    # DYNAMIC, so the duct actually falls and there is a new shape worth saving
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(spec.ring_density)
    rings.append(p)

mat = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, mat, dynamic_friction=0.5, surface_thickness=0.004,
    surface_stretch_stiffness=1e4, surface_shear_stiffness=1e1,
    surface_bend_stiffness=1e-2)

n_circ = 28
sleeves = []
for k in range(N - 1):
    x0, x1 = xs[k], xs[k + 1]
    L = max(2, round(SPACING / (2 * math.pi * R / n_circ)))
    pts, tris = [], []
    for j in range(L + 1):
        x = x0 + (x1 - x0) * j / L
        for i in range(n_circ):
            a = 2 * math.pi * i / n_circ
            pts.append([x, R * math.cos(a), H + R * math.sin(a)])
    for j in range(L):
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a, b = j * n_circ + i, j * n_circ + i2
            c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
            tris += [[a, b, c], [a, c, d]]
    root, skin = f"/World/sleeve_{k:02d}", f"/World/sleeve_{k:02d}/skin"
    UsdGeom.Xform.Define(stage, root)
    m = UsdGeom.Mesh.Define(stage, skin)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    m.CreateDoubleSidedAttr(True)
    deformableUtils.create_auto_surface_deformable_hierarchy(
        stage, root_prim_path=root, simulation_mesh_path=f"{root}/simMesh",
        cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True)
    rp = stage.GetPrimAtPath(root)
    rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    for nm, val in (("physxDeformableBody:selfCollision", False),
                    ("physxDeformableBody:solverPositionIterationCount", 16),
                    ("physxDeformableBody:collisionPairUpdateFrequency", 4),
                    ("physxDeformableBody:collisionIterationMultiplier", 4)):
        a = rp.GetAttribute(nm)
        if a and a.IsValid():
            a.Set(val)
    physicsUtils.add_physics_material_to_prim(stage, rp, mat)
    for side, ri in ((0, k), (1, k + 1)):
        deformableUtils.create_auto_deformable_attachment(
            stage, target_attachment_path=Sdf.Path(f"{root}/seam_{side}"),
            attachable0_path=Sdf.Path(root), attachable1_path=Sdf.Path(rings[ri]))
    sleeves.append(skin)

authored = {p: np.array(stage.GetPrimAtPath(p).GetAttribute("points").Get())
            for p in sleeves}
authored_ring0 = np.array(
    UsdGeom.Xformable(stage.GetPrimAtPath(rings[0])).GetLocalTransformation().ExtractTranslation())

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(360):          # 3 s: the duct drops from 1.2 m and settles
    sim.step(render=False)

print("\n===== FREEZE ROUND TRIP =====", flush=True)

live = capture(stage)
s0 = live.get(sleeves[0], {})
if "points" in s0:
    p = s0["points"]
    print(f"live cloth  : {p.shape[0]} pts, z range {p[:, 2].min():.3f}..{p[:, 2].max():.3f} m "
          f"(authored z {authored[sleeves[0]][:, 2].min():.3f}..{authored[sleeves[0]][:, 2].max():.3f})",
          flush=True)
r0 = live.get(rings[0], {})
if "position" in r0:
    print(f"live ring_00: {r0['position']}  (authored {authored_ring0})", flush=True)

freeze_to_usd(OUT, stage)

# --- reload the saved file in a clean layer and check it kept the settled shape ---
reloaded = Usd.Stage.Open(OUT)
ok = True
for p in sleeves:
    saved = np.array(reloaded.GetPrimAtPath(p).GetAttribute("points").Get())
    moved = np.abs(saved - authored[p]).max()
    if moved < 1e-4:
        ok = False
    print(f"reloaded {p}: moved {moved:.4f} m from authored", flush=True)
rr = UsdGeom.Xformable(reloaded.GetPrimAtPath(rings[0])).GetLocalTransformation().ExtractTranslation()
print(f"reloaded ring_00 at {np.array(rr)} (authored {authored_ring0})", flush=True)
print(f"RESULT: {'the saved USD holds the SIMULATED shape' if ok else 'FAILED - saved the authored shape'}",
      flush=True)
print("===== END =====", flush=True)
app.close()
