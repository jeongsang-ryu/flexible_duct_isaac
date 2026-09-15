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
ap.add_argument("--loops-per-gap", type=int, default=0,
                help="axial mesh loops per gap; 0 = pick so the triangles come "
                     "out roughly square")
ap.add_argument("--stretch", type=float, default=1.0e4)
ap.add_argument("--bend", type=float, default=1.0e-2)
ap.add_argument("--shear", type=float, default=1.0e1)
ap.add_argument("--dynamic-hoops", action="store_true")
ap.add_argument("--bar", action="store_true",
                help="put a horizontal bar under the duct so it drapes over it")
ap.add_argument("--bar-height", type=float, default=1.2)
ap.add_argument("--bar-radius", type=float, default=0.12)
ap.add_argument("--bar-length", type=float, default=1.5)
ap.add_argument("--cam-dist", type=float, default=0.0,
                help="override the computed camera distance [m]")
ap.add_argument("--cam-aspect", type=float, default=1000.0 / 640.0,
                help="viewport aspect; must match the recording crop")
ap.add_argument("--frame", action="store_true",
                help="aim the viewport camera at the scene instead of leaving "
                     "it at Kit's default perspective")
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
# Nine sleeves is a lot more deformable surface than one. The contact budget is
# raised accordingly -- overrunning it makes surfaces stop simulating silently.
# The budget has to scale with the number of sleeves. Overrunning it does NOT
# raise an error -- surfaces just stop simulating -- so it is sized from the
# actual sleeve count rather than left at a figure that happened to work for 9.
# Do NOT scale this linearly. Raising it to 38M slots for 38 sleeves exhausted
# GPU memory -- the log filled with "allocDeviceBuffer failed with error code
# 2" and the solver then ran on garbage contact data. 8M is the value that has
# actually been verified, and the budget is per-scene, not per-sleeve.
_contact_mb = 8
sp.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                   Sdf.ValueTypeNames.UInt).Set(_contact_mb * 1048576)

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

if args.bar:
    # BAR THICKNESS IS NOT FREE. Hoops 0.2 m apart with a 0.016 m cross-section
    # leave a 0.184 m gap between them, so a bar thinner than that can pass
    # straight between two hoops instead of carrying the duct. The fabric now
    # spans those gaps for real, which helps, but sizing the bar above the gap
    # removes the question entirely.
    gap_between_hoops = args.spacing - 2 * TUBE
    if 2 * args.bar_radius <= gap_between_hoops:
        print(f"[duct] WARNING bar diameter {2*args.bar_radius:.3f} m fits the "
              f"{gap_between_hoops:.3f} m hoop gap -- it may slip through",
              flush=True)
    bar = UsdGeom.Capsule.Define(stage, "/World/bar")
    bar.CreateAxisAttr("Y")
    bar.CreateRadiusAttr(float(args.bar_radius))
    bar.CreateHeightAttr(float(args.bar_length))
    bar.CreateDisplayColorAttr().Set([Gf.Vec3f(0.25, 0.25, 0.28)])
    UsdGeom.Xformable(bar).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, args.bar_height))
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())   # collider only = static
    # A bare collider gets the default physics material, and the duct slid
    # straight off it. Worse, hoops in the YZ plane are WHEELS about the bar,
    # so friction alone does not stop them rolling -- the fabric bridging the
    # hoops is what actually has to grip. Give the bar a high-friction material
    # and let the yellow sleeves do the holding.
    from pxr import UsdShade  # noqa: E402
    bmat = UsdShade.Material.Define(stage, "/World/bar_material")
    UsdPhysics.MaterialAPI.Apply(bmat.GetPrim()).CreateStaticFrictionAttr(1.2)
    UsdPhysics.MaterialAPI(bmat.GetPrim()).CreateDynamicFrictionAttr(1.0)
    UsdPhysics.MaterialAPI(bmat.GetPrim()).CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(bar.GetPrim())
    UsdShade.MaterialBindingAPI(bar.GetPrim()).Bind(
        bmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    print(f"[duct] bar at z={args.bar_height}, radius {args.bar_radius}, "
          f"hoop gap {gap_between_hoops:.3f} m", flush=True)
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
    # SLIVER TRIANGLES ARE THE OTHER HALF OF THE 0.1 m FAILURE. The mesh is
    # n_circ segments around a 1.26 m circumference, i.e. ~45 mm spacing
    # circumferentially. Holding loops-per-gap at 16 over a 0.1 m gap gives
    # 6 mm x 45 mm slivers -- a 7:1 aspect ratio the cloth solver handles
    # badly. Pick the axial resolution from the circumferential one instead,
    # so the elements stay near square at any spacing.
    L = args.loops_per_gap or max(
        2, round(args.spacing / (2 * math.pi * cloth_r / args.n_circ)))
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
print(f"[duct] mesh {args.n_circ} circumferential x "
      f"{args.loops_per_gap or max(2, round(args.spacing / (2 * math.pi * cloth_r / args.n_circ)))}"
      f" axial per gap "
      f"(element {2 * math.pi * cloth_r / args.n_circ * 1e3:.0f} mm x "
      f"{args.spacing / (args.loops_per_gap or max(2, round(args.spacing / (2 * math.pi * cloth_r / args.n_circ)))) * 1e3:.0f} mm)",
      flush=True)
print(f"[duct] {n_ok}/{args.n_rings - 1} sleeves built, {n_seams} seams sewn "
      f"(expect {2 * (args.n_rings - 1)})", flush=True)
print(f"[duct] sleeve r={cloth_r:.3f}, hoop tube {R - TUBE:.3f}..{R + TUBE:.3f}, "
      f"overlapping={abs(args.offset) < TUBE}", flush=True)

if args.frame:
    # Kit's default perspective is fixed and does not know the scene grew.
    # SetLookAt builds the WORLD-to-camera matrix, so the camera transform is
    # its inverse.
    #
    # THE LENS IS NOT 28 mm. UsdGeom.Camera defaults to focalLength 50 with a
    # 15.29 mm vertical aperture; assuming 28 made the first framing 2.3x too
    # close and cropped the hanging ends off. Author the lens, then derive the
    # angles from what was actually authored.
    import math as _m
    FOCAL, HAP = 20.0, 36.0
    VAP = HAP / args.cam_aspect
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(FOCAL)
    cam.CreateHorizontalApertureAttr(HAP)
    cam.CreateVerticalApertureAttr(VAP)

    bar_z = args.bar_height if args.bar else H
    lo_z = 0.0
    hi_z = max(H, bar_z) + 0.4
    target = Gf.Vec3d(0.0, 0.0, (lo_z + hi_z) * 0.5)
    need_w = span + 1.2          # the duct also swings out sideways as it drapes
    need_h = (hi_z - lo_z) * 1.1
    hfov = 2.0 * _m.atan(HAP * 0.5 / FOCAL)
    vfov = 2.0 * _m.atan(VAP * 0.5 / FOCAL)
    dist = args.cam_dist or max(need_w * 0.5 / _m.tan(hfov * 0.5),
                                need_h * 0.5 / _m.tan(vfov * 0.5)) * 1.10

    eye = target + Gf.Vec3d(0.35, -0.90, 0.22).GetNormalized() * dist
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/cam"
        print(f"[duct] cam dist {dist:.2f} m, fits {need_w:.1f} x {need_h:.1f} m "
              f"(hfov {_m.degrees(hfov):.0f}, vfov {_m.degrees(vfov):.0f})",
              flush=True)
    except Exception as exc:                      # GUI-only; never fatal
        print(f"[duct] could not retarget viewport: {exc}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
sp.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

print("[duct] running. Each sleeve should stay on its two hoops and sag "
      "between them.", flush=True)
import time  # noqa: E402

_n, _t0, _acc = 0, time.perf_counter(), 0.0
while app.is_running():
    _a = time.perf_counter()
    sim.step(render=True)
    _acc += time.perf_counter() - _a
    _n += 1
    if _n % 300 == 0:
        _wall = time.perf_counter() - _t0
        print(f"[perf] {_n} steps  {_acc / _n * 1e3:.1f} ms/step  "
              f"{_n / _wall:.1f} fps  (sim+render, {args.n_rings - 1} sleeves)",
              flush=True)

app.close()
