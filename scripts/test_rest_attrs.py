"""What do the rest-shape arrays actually look like BEFORE we overwrite them?

bake_rest_shape writes restShapePoints = the current simulation points (84 of
them for one sleeve). But the schema says restShapePoints "describes the per
triangle planar shape", which hints the array may be PER-TRIANGLE (3 entries
per face, 336 for 112 triangles) rather than per-vertex. If so, writing 84
points silently breaks the indexing and the bake does nothing -- which is
exactly what the measurement showed.

So: build one sleeve and print the authored sizes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from duct_sim.builder import spawn_duct  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
UsdPhysics.Scene.Define(stage, "/World/physicsScene")

spec = DuctSpec()
MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=2e4, surface_shear_stiffness=2.0,
    surface_bend_stiffness=1e-3)

spawn_duct(stage, 0, 3, 0.05, spec, MAT, origin=(0, 0, 0.3))

print("\n===== AUTHORED REST ARRAYS =====", flush=True)
for prim in stage.Traverse():
    if not prim.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"):
        continue
    print(f"\n{prim.GetPath()}", flush=True)
    for name in ("omniphysics:restShapePoints", "omniphysics:restTriVtxIndices",
                 "omniphysics:restAdjTriPairs", "omniphysics:restBendAngles",
                 "omniphysics:restBendAnglesDefault", "points",
                 "faceVertexIndices"):
        a = prim.GetAttribute(name)
        if not a or not a.IsValid():
            print(f"   {name:38s} <absent>", flush=True)
            continue
        v = a.Get()
        if v is None:
            print(f"   {name:38s} None (declared, unauthored)", flush=True)
        elif hasattr(v, "__len__") and not isinstance(v, str):
            arr = np.array(v)
            extra = ""
            if name == "omniphysics:restTriVtxIndices" and len(arr):
                extra = f"  max index {arr.max()}"
            print(f"   {name:38s} len {len(v)}  shape {arr.shape}{extra}", flush=True)
        else:
            print(f"   {name:38s} = {v!r}", flush=True)
    break

print("\n===== END =====", flush=True)
app.close()
