"""Minimal isolation test: TWO hoops held 4 m apart, one cloth tube between.

The full duct has 41 hoops, 40 gaps, 80 attachments and a bar to drape over --
far too many interacting parts to tell which one is broken. This strips it to
the single question that has to work before any of that can: does one cloth
tube attach to two hoops and hang between them?

Both hoops are held in place (fixed joints to the world), so anything that
moves is the cloth. If the cloth hangs in a catenary between them, the
attachment mechanism is sound and the problem is elsewhere. If it falls, drifts
or explodes, the failure is right here and every larger result was built on it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/duct_two")
ap.add_argument("--gap", type=float, default=4.0, help="distance between hoops [m]")
ap.add_argument("--loops", type=int, default=40, help="cloth rings along the tube")
ap.add_argument("--n-circ", type=int, default=24)
ap.add_argument("--no-physics", action="store_true",
                help="render without ever calling play(): separates 'the scene "
                     "is not described right' from 'physics destroys it'")
ap.add_argument("--n-rings", type=int, default=2,
                help="hoops spread evenly across the span, all held aloft")
ap.add_argument("--height", type=float, default=3.0)
ap.add_argument("--cloth-offset", type=float, default=0.0,
                help="cloth radius relative to the hoop CENTRELINE; 0 = "
                     "passes through the hoop tube (overlap, attachable)")
ap.add_argument("--steps", type=int, default=420)
ap.add_argument("--every", type=int, default=14)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import math  # noqa: E402

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

from duct_sim.build_b_physx import ensure_gpu_dynamics  # noqa: E402
from duct_sim.geometry import create_ring  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
sc.CreateGravityMagnitudeAttr().Set(9.81)
ensure_gpu_dynamics(stage)
# Two scene settings the official SurfaceDeformableDemo sets and this script did
# not. The GPU deformable-surface contact buffer in particular has a default
# that a 4 m sleeve overruns immediately, and when it does the surface simply
# does not simulate -- without an error, which is the state this test was in.
_scene_prim = stage.GetPrimAtPath("/World/physicsScene")
_scene_prim.CreateAttribute("physxScene:timeStepsPerSecond",
                            Sdf.ValueTypeNames.UInt).Set(120)
_scene_prim.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                            Sdf.ValueTypeNames.UInt).Set(4 * 1048576)

key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 35))
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(700.0)

g = UsdGeom.Cube.Define(stage, "/World/ground")
g.CreateSizeAttr(1.0)
ggx = UsdGeom.Xformable(g)
ggx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
ggx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())

spec = DuctSpec()
R = spec.radius

# --- N hoops, ALL held in the air by fixed joints, spread across the span ---
# Nothing here is allowed to rest on the ground: the point of the test is what
# the CLOTH does between hoops, and a hoop touching the floor would carry part
# of the load and muddy that.
ring_paths = []
_xs = ([-args.gap / 2, args.gap / 2] if args.n_rings <= 2
       else [-args.gap / 2 + args.gap * k / (args.n_rings - 1)
             for k in range(args.n_rings)])
for i, x in enumerate(_xs):
    rp = f"/World/ring_{i}"
    create_ring(stage, rp, R, spec.ring_thickness, n_seg=spec.ring_segments)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(rp))
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, args.height))
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0).GetQuat()))
    prim = stage.GetPrimAtPath(rp)
    for seg in prim.GetChildren():
        UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(0.02, 0.02, 0.02)])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(spec.ring_density)
    fj = UsdPhysics.FixedJoint.Define(stage, f"/World/anchor_{i}")
    fj.CreateBody1Rel().SetTargets([rp])
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    ring_paths.append(rp)

# --- one cloth tube spanning the gap ---
# THE CLOTH MUST OVERLAP THE HOOP, not clear it.
# create_auto_deformable_attachment binds the cloth vertices that lie INSIDE
# the rigid body's collision geometry. Put the sleeve outside the hoop and it
# finds none: the attachment prim is still created and still returns True, but
# it holds nothing, and the cloth simply falls off. That is what every previous
# run was doing -- and it is why increasing the clearance to hide the hoops
# made things worse, not better. The sleeve is therefore placed ON the hoop
# centreline, so its surface passes through the hoop's tube.
cloth_r = R + args.cloth_offset
pts = []
for j in range(args.loops + 1):
    t = j / args.loops
    x = -args.gap / 2 + args.gap * t
    for i in range(args.n_circ):
        a = 2 * math.pi * i / args.n_circ
        pts.append([x, cloth_r * math.cos(a), args.height + cloth_r * math.sin(a)])
pts = np.array(pts)
tris = []
for j in range(args.loops):
    for i in range(args.n_circ):
        i2 = (i + 1) % args.n_circ
        a = j * args.n_circ + i
        b = j * args.n_circ + i2
        c = (j + 1) * args.n_circ + i2
        d = (j + 1) * args.n_circ + i
        tris.append([a, b, c])
        tris.append([a, c, d])
tris = np.array(tris, dtype=np.int32)

root = "/World/cloth"
skin = f"{root}/skin"
UsdGeom.Xform.Define(stage, root)
m = UsdGeom.Mesh.Define(stage, skin)
m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
m.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.flatten().tolist()))
m.CreateDoubleSidedAttr(True)
# The demo does this before cooking; without the standard op set the cooker can
# see a transform it cannot decompose.
physicsUtils.setup_transform_as_scale_orient_translate(m)
m.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.80, 0.10)])   # yellow fabric

ok = deformableUtils.create_auto_surface_deformable_hierarchy(
    stage, root_prim_path=root, simulation_mesh_path=f"{root}/simMesh",
    cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
    set_visibility_with_guide_purpose=True)
print(f"[two] deformable hierarchy created: {ok}", flush=True)
rp = stage.GetPrimAtPath(root)
rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
for name, val in (("physxDeformableBody:selfCollision", True),
                  ("physxDeformableBody:solverPositionIterationCount", 16)):
    a = rp.GetAttribute(name)
    if a and a.IsValid():
        a.Set(val)

mat = "/World/cloth_material"
deformableUtils.add_surface_deformable_material(
    stage, mat, dynamic_friction=0.5, surface_thickness=0.004,
    surface_stretch_stiffness=1.0e3, surface_shear_stiffness=1.0e1,
    surface_bend_stiffness=1.0e-3)
physicsUtils.add_physics_material_to_prim(stage, rp, mat)

for i, rpth in enumerate(ring_paths):
    made = deformableUtils.create_auto_deformable_attachment(
        stage, Sdf.Path(f"/World/seam_{i}"), Sdf.Path(root), Sdf.Path(rpth))
    print(f"[two] attachment to ring {i}: {made}", flush=True)

cam = UsdGeom.Camera.Define(stage, "/World/sidecam")
cx = UsdGeom.Xformable(cam)
# Camera built with an explicit look-at matrix instead of Euler angles.
# The previous version used AddRotateXYZOp(90,0,0) and reasoned that this
# points a USD camera along +Y. That reasoning is easy to get wrong (rotation
# order, which axis the camera looks down, where "up" ends up), and a camera
# aimed at empty space is indistinguishable from a scene that failed to build
# -- which is the confusion that cost several runs today. SetLookAt removes the
# guesswork: give it an eye, a target and an up vector, and invert it because a
# prim's transform maps camera space to world while SetLookAt returns
# world-to-camera.
_bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
_range = Gf.Range3d()
for _p in [stage.GetPrimAtPath(x) for x in ring_paths] + [stage.GetPrimAtPath(skin)]:
    if _p and _p.IsValid():
        _range = Gf.Range3d.GetUnion(_range, _bb.ComputeWorldBound(_p).ComputeAlignedRange())
_c = _range.GetMidpoint()
_size = _range.GetSize()
_span = max(_size[0], _size[1], _size[2], 0.5)

# three-quarter view: off to the side, above, and back -- so the duct reads as
# a tube in 3D rather than as an ambiguous silhouette
_eye = Gf.Vec3d(_c[0] + _span * 0.55,
                _c[1] - _span * 1.05,
                _c[2] + _span * 0.45)
_view = Gf.Matrix4d().SetLookAt(_eye, Gf.Vec3d(_c[0], _c[1], _c[2]),
                                Gf.Vec3d(0, 0, 1))
cx.AddTransformOp().Set(_view.GetInverse())
cam.CreateFocalLengthAttr(30.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))

# REFERENCE MARKER at the exact bbox centre. The ground renders but the duct
# does not, and those two facts cannot both be explained by the camera OR by
# the geometry alone. A bright sphere placed where the camera is aimed settles
# it: if the sphere lands in the middle of the frame, the camera is correct and
# the duct prims are genuinely not being drawn; if it is missing or off-centre,
# the camera transform is still wrong.
_mark = UsdGeom.Sphere.Define(stage, "/World/_camera_target_marker")
_mark.CreateRadiusAttr(0.25)
_mark.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
UsdGeom.Xformable(_mark).AddTranslateOp().Set(Gf.Vec3d(_c[0], _c[1], _c[2]))
print(f"[two] bbox size {tuple(round(v,2) for v in _size)} centre "
      f"{tuple(round(v,2) for v in _c)}", flush=True)
print(f"[two] camera eye {tuple(round(v,2) for v in _eye)} looking at centre",
      flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
ensure_gpu_dynamics(stage)
# The USD state has been verified correct prim by prim -- rings and cloth are
# active, visible, purpose=default, with the right world bounds -- and still
# nothing draws. The remaining fork is whether the scene is fine until physics
# touches it. Rendering with play() never called answers that directly.
if not args.no_physics:
    sim.play()
else:
    print("[two] PHYSICS DISABLED -- rendering the authored scene only", flush=True)

os.makedirs(args.out, exist_ok=True)
prod = rep.create.render_product("/World/sidecam", (1280, 720))
w = rep.WriterRegistry.get("BasicWriter")
w.initialize(output_dir=args.out, rgb=True)
w.attach([prod])

# Report BOTH meshes and their visibility. The skin is authored geometry and
# may never be rewritten; the simMesh is what PhysX actually solves. Measuring
# only the skin cannot tell "the cloth did not move" from "the cloth moved and
# the skin was not updated", and those need completely different fixes.
sim_mesh_path = f"{root}/simMesh"
for label, path in (("skin", skin), ("simMesh", sim_mesh_path)):
    pr = stage.GetPrimAtPath(path)
    if not pr or not pr.IsValid():
        print(f"[two] {label}: MISSING at {path}", flush=True)
        continue
    img = UsdGeom.Imageable(pr)
    vis = img.ComputeVisibility()
    purpose = img.ComputePurpose()
    npts = len(UsdGeom.Mesh(pr).GetPointsAttr().Get() or [])
    print(f"[two] {label}: visibility={vis} purpose={purpose} points={npts}",
          flush=True)

z0 = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(skin)).GetPointsAttr().Get())[:, 2]
sm = stage.GetPrimAtPath(sim_mesh_path)
zs0 = np.array(UsdGeom.Mesh(sm).GetPointsAttr().Get())[:, 2] if sm and sm.IsValid() else None
frames = 0
for i in range(args.steps):
    sim.step(render=True)
    if i % args.every == 0:
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        frames += 1
z1 = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(skin)).GetPointsAttr().Get())[:, 2]
if zs0 is not None:
    zs1 = np.array(UsdGeom.Mesh(sm).GetPointsAttr().Get())[:, 2]
    print(f"[two] simMesh z {zs0.min():.3f}..{zs0.max():.3f} -> "
          f"{zs1.min():.3f}..{zs1.max():.3f}  sag={zs0.min()-zs1.min():+.3f} m",
          flush=True)
print(f"[two] cloth_offset={args.cloth_offset:+.3f} -> cloth_r={cloth_r:.4f}, "
      f"hoop tube spans {R-spec.ring_thickness:.4f}..{R+spec.ring_thickness:.4f}",
      flush=True)
print(f"[two] frames={frames}  cloth z {z0.min():.3f}..{z0.max():.3f} -> "
      f"{z1.min():.3f}..{z1.max():.3f}  sag={z0.min()-z1.min():+.3f} m", flush=True)
print(f"[two] {len(ring_paths)} hoops held aloft at z={args.height}, span "
      f"{args.gap} m; cloth should hang BELOW them and stay attached",
      flush=True)
sys.stdout.flush()
app.close()
