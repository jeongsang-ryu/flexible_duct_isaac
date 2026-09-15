"""A cloth sleeve sewn to a hoop at EACH end -- one duct segment, in isolation.

This is the piece the whole duct is made of. The single-hoop test showed a
sleeve can hang from one ring; sewing both ends is different in kind, because
now the fabric is constrained at two places at once and gravity has to be
carried between them. What the sleeve does in the middle is the answer:

  * held at both ends -> it sags into a catenary between the hoops and stays
                         attached, which is what a duct gap does
  * held at one end   -> it swings down from that hoop only
  * held at neither   -> it drops to the floor

Both hoops are static (density 0) so nothing moves except the fabric.

Scaling this to the full duct is then just repeating the segment: N hoops,
N-1 sleeves, each sewn to the two hoops it spans.

Run it with:

    conda activate hmclab6
    cd ~/hmcl_issac_project/duct_sim
    python scripts/cloth_between_rings_gui.py

    --span      distance between the hoops [m]
    --n-rings   hoops across that span (2 = just the ends)
    --offset    sleeve radius minus hoop centreline radius; 0 = overlapping,
                which is what makes the attachment bind at all
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--span", type=float, default=2.0, help="distance between end hoops [m]")
ap.add_argument("--n-rings", type=int, default=2, help="hoops spread across the span")
ap.add_argument("--offset", type=float, default=0.0)
ap.add_argument("--height", type=float, default=1.8)
ap.add_argument("--n-circ", type=int, default=32)
ap.add_argument("--loops-per-m", type=int, default=24)
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
gx.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

spec = DuctSpec()
R, TUBE, H = spec.radius, spec.ring_thickness, args.height

# --- hoops, all static, spread along X ---
xs = ([-args.span / 2, args.span / 2] if args.n_rings <= 2
      else [-args.span / 2 + args.span * k / (args.n_rings - 1)
            for k in range(args.n_rings)])
ring_paths = []
for i, x in enumerate(xs):
    path = f"/World/ring_{i}"
    create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments)
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, H))
    # hoop plane is YZ so the sleeve can run along X through it
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0).GetQuat()))
    for seg in prim.GetChildren():
        UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(0.03, 0.03, 0.03)])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(0.0)   # static
    ring_paths.append(path)

# --- the sleeve spanning them, built on the hoop centreline ---
cloth_r = R + args.offset
loops = max(4, int(args.span * args.loops_per_m))
pts, tris = [], []
for j in range(loops + 1):
    x = xs[0] + (xs[-1] - xs[0]) * j / loops
    for i in range(args.n_circ):
        a = 2 * math.pi * i / args.n_circ
        pts.append([x, cloth_r * math.cos(a), H + cloth_r * math.sin(a)])
for j in range(loops):
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
print(f"[seg] deformable hierarchy: {ok}", flush=True)

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

# --- one attachment per hoop ---
for i, path in enumerate(ring_paths):
    made = deformableUtils.create_auto_deformable_attachment(
        stage,
        target_attachment_path=Sdf.Path(f"{root}/seam_{i}"),
        attachable0_path=Sdf.Path(root),
        attachable1_path=Sdf.Path(path),
    )
    print(f"[seg] seam {i} at x={xs[i]:+.2f} -> {made}", flush=True)

print(f"[seg] {len(ring_paths)} static hoops over a {args.span} m span, "
      f"sleeve r={cloth_r:.3f} (hoop tube {R-TUBE:.3f}..{R+TUBE:.3f}), "
      f"{len(pts)} verts", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

print("[seg] running. The sleeve should stay on BOTH hoops and sag between.",
      flush=True)
while app.is_running():
    sim.step(render=True)

app.close()
