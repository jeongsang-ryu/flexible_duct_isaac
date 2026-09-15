"""How many seam elements each hoop shape actually binds, and where.

The fabric is drawn INSIDE the hoops (clearance < 0) so the hoops read as ribs.
That means it is sewn to the hoop's INNER surface. A capsule ring has one --
its tube spans 196..204 mm and the fabric at 192 mm sits 4 mm off it. A solid
disc does not: it is filled, its only radial surface is the rim at 204 mm, and
the fabric at 192 mm is 12 mm inside the solid with nothing to bind to.

Prediction this checks: put the fabric OUTSIDE (clearance > 0) and the disc's
rim becomes the sewing surface, so the seam count recovers.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=float, default=3.0)
ap.add_argument("--spacing", type=float, default=0.10)
ap.add_argument("--n-circ", type=int, default=40)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402
from duct_sim.builder import spawn_duct_single  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402

# seams look like one per COLLISION SHAPE, not per unit of surface: 16
# capsules gave 17 per hoop and 1 disc gave 2. If that is the rule, dialling
# ring_segments should move the seam count with it, one for one.
CASES = [
    ("capsule x16", False, -0.004, 16),
    ("capsule x8 ", False, -0.004, 8),
    ("capsule x4 ", False, -0.004, 4),
    ("capsule x2 ", False, -0.004, 2),
    ("disc       ", True,  -0.004, 16),
]

print()
print(f"{'case':14s}{'shapes/hoop':>13s}{'children':>8s}{'per hoop':>12s}"
      f"{'BOUND PTS':>12s}{'per hoop':>12s}")
print("-" * 71)
import dataclasses
for name, solid, clearance, nseg in CASES:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    stage.GetPrimAtPath("/World/physicsScene").ApplyAPI("PhysxSceneAPI")
    MAT = "/World/cloth_material"
    deformableUtils.add_surface_deformable_material(
        stage, MAT, dynamic_friction=0.5, surface_thickness=0.003,
        surface_stretch_stiffness=3e3, surface_shear_stiffness=1.0,
        surface_bend_stiffness=1e-3)
    spec = dataclasses.replace(DuctSpec(), ring_segments=nseg)
    n_rings = int(round(args.length / args.spacing)) + 1
    try:
        _, rings, _, n_seams = spawn_duct_single(
            stage, 0, n_rings, args.spacing, spec, MAT,
            origin=(0.0, 0.0, 0.30), n_circ=args.n_circ, clearance=clearance,
            rib_visual=True, solid_hoop=solid, verbose=False)
        shp = 1 if solid else nseg
        # n_seams counts CHILD PRIMS of the seam scope. Each child holds a
        # localPositionsSrc1 array, so one child can bind hundreds of vertices.
        # Counting children therefore says nothing about grip on its own --
        # count the actual attached points.
        pts_bound = 0
        for prim in stage.Traverse():
            a = prim.GetAttribute("omniphysics:localPositionsSrc1")
            if a and a.Get() is not None:
                pts_bound += len(a.Get())
        print(f"{name:14s}{shp:13d}{n_seams:8d}"
              f"{n_seams/max(len(rings),1):12.1f}"
              f"{pts_bound:12d}{pts_bound/max(len(rings),1):12.1f}", flush=True)
    except Exception as exc:
        print(f"{name:14s}   FAILED {exc}", flush=True)
print()
app.close()
