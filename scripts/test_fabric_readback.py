"""Can the SIMULATED duct state be read back and saved? Test before building.

This is the whole feasibility question for "drag it into a track layout, then
save that as the initial state". Plain USD cannot answer it: with GPU dynamics
the simulated points live in Fabric and the USD attribute keeps returning the
authored pose forever. Every sag measurement earlier in this project was wrong
for exactly this reason.

So: build a duct, let gravity deform it, then compare
    USD    stage.GetAttribute("points")          <- expected: unchanged
    Fabric usdrt equivalent                      <- expected: deformed
and report whether the Fabric values are actually retrievable.
"""

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
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt  # noqa: E402

from duct_sim.geometry import create_ring  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

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

spec = DuctSpec()
R, TUBE, H = spec.radius, spec.ring_thickness, 2.0
xs = [-0.2, 0.0, 0.2]
rings = []
for i, x in enumerate(xs):
    p = f"/World/ring_{i}"
    create_ring(stage, p, R, TUBE, n_seg=spec.ring_segments)
    prim = stage.GetPrimAtPath(p)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, H))
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0).GetQuat()))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(0.0)   # static ends
    rings.append(p)

mat = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, mat, dynamic_friction=0.5, surface_thickness=0.004,
    surface_stretch_stiffness=1e4, surface_shear_stiffness=1e1,
    surface_bend_stiffness=1e-2)

n_circ = 28
sleeves = []
for k in range(len(xs) - 1):
    x0, x1 = xs[k], xs[k + 1]
    L = 2
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
    root, skin = f"/World/sleeve_{k}", f"/World/sleeve_{k}/skin"
    UsdGeom.Xform.Define(stage, root)
    m = UsdGeom.Mesh.Define(stage, skin)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    deformableUtils.create_auto_surface_deformable_hierarchy(
        stage, root_prim_path=root, simulation_mesh_path=f"{root}/simMesh",
        cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True)
    rp = stage.GetPrimAtPath(root)
    rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    physicsUtils.add_physics_material_to_prim(stage, rp, mat)
    for side, ri in ((0, k), (1, k + 1)):
        deformableUtils.create_auto_deformable_attachment(
            stage, target_attachment_path=Sdf.Path(f"{root}/seam_{side}"),
            attachable0_path=Sdf.Path(root), attachable1_path=Sdf.Path(rings[ri]))
    sleeves.append(skin)

authored = {p: np.array(stage.GetPrimAtPath(p).GetAttribute("points").Get())
            for p in sleeves}

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(240):
    sim.step(render=False)

print("\n===== READBACK TEST =====", flush=True)
for p in sleeves:
    now = np.array(stage.GetPrimAtPath(p).GetAttribute("points").Get())
    print(f"USD    {p}: max move from authored = "
          f"{np.abs(now - authored[p]).max():.6f} m", flush=True)

# ---- probe every route to the simulated state ----
try:
    import usdrt
    from usdrt import Usd as RtUsd
    sid = ctx.get_stage_id()
    rt = RtUsd.Stage.Attach(sid)
    rp = rt.GetPrimAtPath(sleeves[0])
    print(f"\nusdrt prim members: "
          f"{[m for m in dir(rp) if not m.startswith('_')]}", flush=True)
    for nm in ("points", "_worldPosition", "Points"):
        try:
            a = rp.GetAttribute(nm)
            v = a.Get() if a else None
            if v is not None:
                arr = np.array(v)
                print(f"usdrt '{nm}': shape {arr.shape} max move "
                      f"{np.abs(arr - authored[sleeves[0]]).max():.6f} m", flush=True)
            else:
                print(f"usdrt '{nm}': None", flush=True)
        except Exception as e:
            print(f"usdrt '{nm}': {e}", flush=True)
except Exception as exc:
    print(f"usdrt route failed: {exc}", flush=True)

try:
    from omni.physx import get_physx_simulation_interface  # noqa
    import omni.physx.tensors as pxt
    print(f"\nomni.physx.tensors members: "
          f"{[m for m in dir(pxt) if not m.startswith('_')]}", flush=True)
except Exception as exc:
    print(f"tensors route failed: {exc}", flush=True)

try:
    import isaacsim.core.prims as icp
    print(f"\nisaacsim.core.prims: "
          f"{[m for m in dir(icp) if 'eform' in m or 'loth' in m or 'article' in m.lower()]}",
          flush=True)
except Exception as exc:
    print(f"core.prims route failed: {exc}", flush=True)

print("===== END =====", flush=True)
app.close()
