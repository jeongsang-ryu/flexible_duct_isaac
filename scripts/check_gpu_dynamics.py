"""Is the deformable solver actually on the GPU?

The scene sets enableGPUDynamics=True and GPU utilisation during an 81 m cloth
track still reads 1-17% while the CPU sits at ~600%. Setting a flag is not
evidence it took: today the viewport reported a camera retarget that never
applied, and an inertia tensor was authored without ever being read back. So
read the flags back from the stage, and check the broadphase and solver type
too -- GPU dynamics silently falls back to CPU when the broadphase is not GPU.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from omni.physx.scripts import deformableUtils
from pxr import Gf, UsdGeom, UsdPhysics
from duct_sim.builder import spawn_duct_single
from duct_sim.spec import DuctSpec

ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sp = stage.GetPrimAtPath("/World/physicsScene")
sp.ApplyAPI("PhysxSceneAPI")

MAT = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
    surface_stretch_stiffness=3e3, surface_shear_stiffness=1.0,
    surface_bend_stiffness=1e-3)
spawn_duct_single(stage, 0, 31, 0.10, DuctSpec(), MAT, origin=(0, 0, 0.30),
                  n_circ=40, clearance=-0.004, rib_visual=True, verbose=False)

from isaacsim.core.api import SimulationContext
sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(60):
    sim.step(render=False)

print("\n=== physics scene, read back AFTER play ===")
for name in sorted(a.GetName() for a in sp.GetAttributes()):
    if not name.startswith("physxScene:"):
        continue
    v = sp.GetAttribute(name).Get()
    if any(k in name.lower() for k in
           ("gpu", "broadphase", "solver", "collision", "enable")):
        print(f"  {name:52s} {v}")

print("\n=== the deformable body ===")
for prim in stage.Traverse():
    if prim.HasAPI("OmniPhysicsSurfaceDeformableSimAPI"):
        print(f"  {prim.GetPath()}")
        for a in prim.GetAttributes():
            n = a.GetName()
            if any(k in n.lower() for k in ("solver", "iteration", "gpu", "enable")):
                print(f"    {n:50s} {a.Get()}")
        break
app.close()
