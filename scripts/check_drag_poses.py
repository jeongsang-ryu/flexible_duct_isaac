"""Does the dragger read distinct hoop poses? Verify without a human clicking."""
import os, sys
sys.path.insert(0, "/home/js/hmcl_issac_project/duct_sim")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np, omni.usd
from omni.physx.scripts import deformableUtils
from pxr import Gf, UsdGeom, UsdPhysics
from duct_sim.builder import spawn_duct_single
from duct_sim.spec import DuctSpec

ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sp = stage.GetPrimAtPath("/World/physicsScene"); sp.ApplyAPI("PhysxSceneAPI")
g = UsdGeom.Cube.Define(stage, "/World/ground"); g.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(g); gx.AddTranslateOp().Set(Gf.Vec3d(0,0,-0.05))
gx.AddScaleOp().Set(Gf.Vec3f(20,20,0.1)); UsdPhysics.CollisionAPI.Apply(g.GetPrim())
MAT="/World/cloth_material"
deformableUtils.add_surface_deformable_material(stage, MAT, dynamic_friction=0.5,
    surface_thickness=0.003, surface_stretch_stiffness=3e3,
    surface_shear_stiffness=1.0, surface_bend_stiffness=1e-3)
_, rings, _, _ = spawn_duct_single(stage, 0, 31, 0.10, DuctSpec(), MAT,
    origin=(0,0,0.30), n_circ=40, clearance=-0.004, rib_visual=True, verbose=False)

from isaacsim.core.api import SimulationContext
sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics(); sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(120):
    sim.step(render=False)

from duct_sim.mouse_drag import HoopDragger
d = HoopDragger(rings)
p = d._positions()
print()
print(f"[check] hoops            {len(p)}")
print(f"[check] x span           {np.ptp(p[:,0]):.3f} m   (a 3 m duct should be ~3)")
print(f"[check] all identical?   {bool(np.ptp(p, axis=0).max() < 1e-9)}")
print(f"[check] first three      {np.round(p[:3], 3).tolist()}")
print()
print("PASS -- poses are distinct" if np.ptp(p[:,0]) > 1.0 else "FAIL -- still degenerate")
app.close()
