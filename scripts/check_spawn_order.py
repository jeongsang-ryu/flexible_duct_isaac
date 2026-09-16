"""Does play-then-spawn break the pose view? That is the GUI's order.

The GUI calls sim.play() on an empty scene at line 329 and spawns the 819 hoops
afterwards. My passing repro built first and played second. If order is the
cause, the same code gives distinct poses one way and zeros the other.
"""
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
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import RigidPrim

def scene():
    ctx = omni.usd.get_context(); ctx.new_stage(); st = ctx.get_stage()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")
    UsdPhysics.Scene.Define(st, "/World/physicsScene")
    p = st.GetPrimAtPath("/World/physicsScene"); p.ApplyAPI("PhysxSceneAPI")
    g = UsdGeom.Cube.Define(st, "/World/ground"); g.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(g); gx.AddTranslateOp().Set(Gf.Vec3d(0,0,-0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(20,20,0.1)); UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    M="/World/cloth_material"
    deformableUtils.add_surface_deformable_material(st, M, dynamic_friction=0.5,
        surface_thickness=0.003, surface_stretch_stiffness=3e3,
        surface_shear_stiffness=1.0, surface_bend_stiffness=1e-3)
    return st, p, M

def build(st, M):
    _, rings, _, _ = spawn_duct_single(st, 0, 31, 0.10, DuctSpec(), M,
        origin=(0,0,0.30), n_circ=40, clearance=-0.004, rib_visual=True, verbose=False)
    return rings

def span(rings):
    v = RigidPrim(rings); v.initialize()
    p, _ = v.get_world_poses()
    p = np.asarray(p.cpu() if hasattr(p,"cpu") else p, dtype=np.float64)
    return float(np.ptp(p[:,0]))

# A: build, then play   (my passing test)
st, p, M = scene()
rings = build(st, M)
sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics(); p.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
for _ in range(120): sim.step(render=False)
print(f"[order] build THEN play : x span {span(rings):7.3f} m", flush=True)
sim.stop(); sim.clear_instance()

# B: play, then build   (what the GUI does)
st, p, M = scene()
sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics(); p.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()
rings = build(st, M)
for _ in range(120): sim.step(render=False)
print(f"[order] play THEN build : x span {span(rings):7.3f} m", flush=True)

# C: play, build, then stop/play  (the proposed fix)
sim.stop(); sim.play()
for _ in range(120): sim.step(render=False)
print(f"[order] + stop/play     : x span {span(rings):7.3f} m", flush=True)
app.close()
