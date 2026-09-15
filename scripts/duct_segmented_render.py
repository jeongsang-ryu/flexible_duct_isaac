"""N hoops, N-1 SEPARATE cloth sleeves, each sewn to the two hoops it spans.

This is the segmented duct the lab asked for, built on the configuration that
was just verified piece by piece: cloth drapes (ball), cloth attaches to a
rigid body (bar), cloth attaches to a hoop (ring), cloth spans two hoops
(segment). This only repeats the last one.

An earlier attempt at exactly this shape produced scattered debris. Two things
were wrong then, and both are fixed here:

  * the hoops were DYNAMIC. A dynamic hoop is dragged around by the fabric
    instead of holding it. NVIDIA's own attachment demo makes its collider
    static with density=0.0, and so does this.
  * the sleeves were drawn OUTSIDE the hoops, to stop the hoops showing
    through. create_auto_deformable_attachment binds the cloth vertices that
    lie INSIDE the rigid body's collision shape, so that cleared the overlap
    and every seam bound nothing -- while still returning True. The sleeves
    here sit on the hoop centreline, where the overlap is greatest.

Run it with:

    conda activate hmclab6
    cd ~/hmcl_issac_project/duct_sim
    python scripts/duct_segmented_gui.py

    --n-rings / --spacing   hoop count and gap [m]
    --dynamic-hoops         let the hoops move (they fall; useful to see why
                            static matters)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--n-rings", type=int, default=10)
ap.add_argument("--spacing", type=float, default=1.0, help="hoop spacing [m]")
ap.add_argument("--height", type=float, default=2.2)
ap.add_argument("--offset", type=float, default=0.0,
                help="sleeve radius minus hoop centreline radius")
ap.add_argument("--n-circ", type=int, default=28)
ap.add_argument("--loops-per-gap", type=int, default=16)
ap.add_argument("--stretch", type=float, default=1.0e4)
ap.add_argument("--bend", type=float, default=1.0e-2)
ap.add_argument("--shear", type=float, default=1.0e1)
ap.add_argument("--dynamic-hoops", action="store_true")
ap.add_argument("--out", default="/tmp/duct_seg_render")
ap.add_argument("--steps", type=int, default=420)
ap.add_argument("--every", type=int, default=12)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

from duct_sim.geometry import create_ring  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import Usd  # noqa: E402

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
# Nine sleeves is a lot more deformable surface than one. The contact budget is
# raised accordingly -- overrunning it makes surfaces stop simulating silently.
sp.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                   Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)

span = args.spacing * (args.n_rings - 1)
ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(span * 2 + 4, span * 2 + 4, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

spec = DuctSpec()
R, TUBE, H = spec.radius, spec.ring_thickness, args.height
xs = [-span / 2 + args.spacing * k for k in range(args.n_rings)]

ring_paths = []
for i, x in enumerate(xs):
    path = f"/World/ring_{i:02d}"
    create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments)
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, H))
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0).GetQuat()))
    for seg in prim.GetChildren():
        UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(0.03, 0.03, 0.03)])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(
        spec.ring_density if args.dynamic_hoops else 0.0)
    ring_paths.append(path)

cloth_r = R + args.offset
mat = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, mat,
    dynamic_friction=0.5,
    surface_thickness=0.004,
    surface_stretch_stiffness=args.stretch,
    surface_shear_stiffness=args.shear,
    surface_bend_stiffness=args.bend,
)

n_ok, n_seams = 0, 0
for k in range(args.n_rings - 1):
    x0, x1 = xs[k], xs[k + 1]
    L = args.loops_per_gap
    pts, tris = [], []
    for j in range(L + 1):
        x = x0 + (x1 - x0) * j / L
        for i in range(args.n_circ):
            a = 2 * math.pi * i / args.n_circ
            pts.append([x, cloth_r * math.cos(a), H + cloth_r * math.sin(a)])
    for j in range(L):
        for i in range(args.n_circ):
            i2 = (i + 1) % args.n_circ
            a = j * args.n_circ + i
            b = j * args.n_circ + i2
            c = (j + 1) * args.n_circ + i2
            d = (j + 1) * args.n_circ + i
            tris += [[a, b, c], [a, c, d]]
    pts = np.array(pts)
    tris = np.array(tris, dtype=np.int32)

    root = f"/World/sleeve_{k:02d}"
    skin = f"{root}/skin"
    UsdGeom.Xform.Define(stage, root)
    mesh = UsdGeom.Mesh.Define(stage, skin)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.flatten().tolist()))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.80, 0.10)])

    if not deformableUtils.create_auto_surface_deformable_hierarchy(
            stage, root_prim_path=root, simulation_mesh_path=f"{root}/simMesh",
            cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True):
        print(f"[duct] sleeve {k}: hierarchy FAILED", flush=True)
        continue
    n_ok += 1

    rp = stage.GetPrimAtPath(root)
    rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    for name, val in (("physxDeformableBody:selfCollision", False),
                      ("physxDeformableBody:solverPositionIterationCount", 16),
                      ("physxDeformableBody:collisionPairUpdateFrequency", 4),
                      ("physxDeformableBody:collisionIterationMultiplier", 4)):
        attr = rp.GetAttribute(name)
        if attr and attr.IsValid():
            attr.Set(val)
    physicsUtils.add_physics_material_to_prim(stage, rp, mat)

    for side, ring_idx in ((0, k), (1, k + 1)):
        if deformableUtils.create_auto_deformable_attachment(
                stage,
                target_attachment_path=Sdf.Path(f"{root}/seam_{side}"),
                attachable0_path=Sdf.Path(root),
                attachable1_path=Sdf.Path(ring_paths[ring_idx])):
            n_seams += 1

print(f"[duct] {args.n_rings} hoops @ {args.spacing} m "
      f"({'dynamic' if args.dynamic_hoops else 'static'}), span {span} m",
      flush=True)
print(f"[duct] {n_ok}/{args.n_rings - 1} sleeves built, {n_seams} seams sewn "
      f"(expect {2 * (args.n_rings - 1)})", flush=True)
print(f"[duct] sleeve r={cloth_r:.3f}, hoop tube {R - TUBE:.3f}..{R + TUBE:.3f}, "
      f"overlapping={abs(args.offset) < TUBE}", flush=True)

# Camera from the measured bounding box via SetLookAt. Reasoning about Euler
# angles produced three empty renders earlier today; a marker sphere placed at
# the aim point proved this construction lands dead centre, so it is reused
# verbatim rather than re-derived.
_bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
_rng = Gf.Range3d()
for _p in ring_paths + [f"/World/sleeve_{k:02d}" for k in range(args.n_rings - 1)]:
    _pr = stage.GetPrimAtPath(_p)
    if _pr and _pr.IsValid():
        _rng = Gf.Range3d.GetUnion(_rng, _bb.ComputeWorldBound(_pr).ComputeAlignedRange())
# For a drop, frame the whole fall -- the duct starts high and ends on the
# ground, so a camera framed on the starting bbox alone would lose it halfway
# down. Extending the range to the floor keeps both ends of the motion in shot.
if args.dynamic_hoops:
    _rng = Gf.Range3d.GetUnion(
        _rng, Gf.Range3d(Gf.Vec3d(_rng.GetMin()[0], _rng.GetMin()[1], 0.0),
                         Gf.Vec3d(_rng.GetMax()[0], _rng.GetMax()[1], 0.05)))
_c = _rng.GetMidpoint()
_sz = _rng.GetSize()
# Distance is sized from the LARGEST dimension and the NARROWER (vertical)
# field of view, not from a single fudge factor. The drop render framed
# z = 0.28..2.12 while the duct started at 2.41, so the first half of the fall
# happened above the top edge and the video looked empty -- a framing failure
# that is indistinguishable from a physics failure unless the numbers are
# checked. 28 mm gives a 39.8 deg vertical field; the 1.35 margin keeps the
# subject inside it with room to spare.
_extent = max(_sz[0], _sz[1], _sz[2], 0.5)
_vfov = 2.0 * math.atan(10.125 / 28.0)
_dist = (_extent * 0.5) / math.tan(_vfov * 0.5) * 1.35
_eye = Gf.Vec3d(_c[0] + _dist * 0.30, _c[1] - _dist * 0.90, _c[2] + _dist * 0.22)
print(f"[duct] framing: extent {_extent:.2f} m -> camera {_dist:.2f} m away, "
      f"vertical coverage {2 * _dist * math.tan(_vfov / 2):.2f} m", flush=True)
_cam = UsdGeom.Camera.Define(stage, "/World/rendercam")
UsdGeom.Xformable(_cam).AddTransformOp().Set(
    Gf.Matrix4d().SetLookAt(_eye, Gf.Vec3d(_c[0], _c[1], _c[2]),
                            Gf.Vec3d(0, 0, 1)).GetInverse())
_cam.CreateFocalLengthAttr(28.0)
_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
print(f"[duct] bbox {tuple(round(v, 2) for v in _sz)} centre "
      f"{tuple(round(v, 2) for v in _c)}, camera at "
      f"{tuple(round(v, 2) for v in _eye)}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)

# TURN FABRIC OFF FOR HEADLESS CAPTURE.
# The GUI shows this duct correctly, but the same scene rendered headless comes
# out empty except for the ground -- and the ground is the one thing physics
# never touches. PhysicsContext enables Fabric whenever the device is CUDA, so
# simulated transforms and deformed points live in Fabric and are never written
# back to USD; the replicator capture path reads USD and therefore sees the
# scene as it was authored, or not at all. Disabling Fabric costs performance
# but puts the results back on the stage where the renderer can find them.
try:
    sim.get_physics_context().enable_fabric(False)
    print("[duct] fabric disabled for USD-based capture", flush=True)
except Exception as exc:
    print(f"[duct] could not disable fabric: {type(exc).__name__}: {exc}", flush=True)

sim.play()

os.makedirs(args.out, exist_ok=True)
_prod = rep.create.render_product("/World/rendercam", (1280, 720))
_writer = rep.WriterRegistry.get("BasicWriter")
_writer.initialize(output_dir=args.out, rgb=True)
_writer.attach([_prod])

_frames = 0
for _i in range(args.steps):
    sim.step(render=True)
    if _i % args.every == 0:
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        _frames += 1
print(f"[duct] wrote {_frames} frames to {args.out}", flush=True)
sys.stdout.flush()
app.close()
