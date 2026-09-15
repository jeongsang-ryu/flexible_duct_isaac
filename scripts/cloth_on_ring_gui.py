"""Cloth sewn to a RING instead of a bar -- the last step before the duct.

The bar test proved attachment works on a simple box. A duct hoop is a much
worse target: it is a thin circle assembled from 16 capsule segments, so the
volume a cloth vertex has to land inside is a 8 mm tube rather than a solid
block. That distinction is exactly what the duct code got wrong -- the sleeve
was drawn OUTSIDE the hoops, nothing overlapped, and
create_auto_deformable_attachment bound zero vertices while still returning
True.

So the sleeve here is built ON the hoop centreline: its surface passes through
the middle of the capsule tube, which is the most overlap a thin ring can
offer. A sleeve hanging from the ring means the duct's seams can work; a sleeve
on the floor means ring attachment needs something else.

The ring is static (density 0), like the demo's collider -- a dynamic hoop would
be dragged around by the cloth rather than holding it.

Run it with:

    conda activate hmclab6
    cd ~/hmcl_issac_project/duct_sim
    python scripts/cloth_on_ring_gui.py

    --offset   sleeve radius relative to the hoop CENTRELINE (0 = through the
               tube). Try +0.02 to see the failure mode: no overlap, no grip.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--offset", type=float, default=0.0,
                help="sleeve radius minus hoop centreline radius [m]")
ap.add_argument("--length", type=float, default=1.2, help="sleeve length [m]")
ap.add_argument("--height", type=float, default=1.8)
ap.add_argument("--n-circ", type=int, default=32)
ap.add_argument("--loops", type=int, default=30)
ap.add_argument("--stretch", type=float, default=1.0e4)
ap.add_argument("--bend", type=float, default=1.0e-2)
ap.add_argument("--shear", type=float, default=1.0e1)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

from duct_sim.geometry import create_ring  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
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

spec = DuctSpec()
R, TUBE = spec.radius, spec.ring_thickness
H = args.height

# --- the hoop: real duct geometry, held still ---
ring_path = "/World/ring"
create_ring(stage, ring_path, R, TUBE, n_seg=spec.ring_segments)
ring_prim = stage.GetPrimAtPath(ring_path)
UsdGeom.Xformable(ring_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, H))
for seg in ring_prim.GetChildren():
    UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(0.03, 0.03, 0.03)])
# density 0 => static collider, the same trick the attachment demo uses on its
# cube. Without it the hoop is dynamic and the cloth drags it down instead of
# hanging from it.
UsdPhysics.RigidBodyAPI.Apply(ring_prim)
UsdPhysics.MassAPI.Apply(ring_prim).CreateDensityAttr(0.0)

# --- the sleeve: a tube hanging from the hoop, built ON the centreline ---
cloth_r = R + args.offset
pts, tris = [], []
for j in range(args.loops + 1):
    z = H - args.length * j / args.loops
    for i in range(args.n_circ):
        a = 2 * math.pi * i / args.n_circ
        pts.append([cloth_r * math.cos(a), cloth_r * math.sin(a), z])
for j in range(args.loops):
    for i in range(args.n_circ):
        i2 = (i + 1) % args.n_circ
        a = j * args.n_circ + i
        b = j * args.n_circ + i2
        c = (j + 1) * args.n_circ + i2
        d = (j + 1) * args.n_circ + i
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
print(f"[ring] deformable hierarchy: {ok}", flush=True)

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
    surface_thickness=0.004,
    surface_stretch_stiffness=args.stretch,
    surface_shear_stiffness=args.shear,
    surface_bend_stiffness=args.bend,
)
physicsUtils.add_physics_material_to_prim(stage, rp, mat)

made = deformableUtils.create_auto_deformable_attachment(
    stage,
    target_attachment_path=Sdf.Path(f"{root}/attachment"),
    attachable0_path=Sdf.Path(root),
    attachable1_path=Sdf.Path(ring_path),
)
print(f"[ring] hoop centreline r={R:.3f}, tube {R-TUBE:.3f}..{R+TUBE:.3f}; "
      f"sleeve r={cloth_r:.3f} (offset {args.offset:+.3f})", flush=True)
print(f"[ring] overlaps hoop tube: {abs(args.offset) < TUBE}", flush=True)
print(f"[ring] create_auto_deformable_attachment -> {made}  "
      f"(True does NOT mean it bound anything)", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

print("[ring] running. The sleeve should HANG from the hoop like a windsock.",
      flush=True)
while app.is_running():
    sim.step(render=True)

app.close()
