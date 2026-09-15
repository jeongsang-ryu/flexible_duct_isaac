"""Spawn a duct, drag it into a track layout by hand, save that layout as USD.

THE WORKFLOW THIS SUPPORTS
    1. spawn a straight duct of some length
    2. drag it with the mouse until it follows the track you want
    3. save -- and what lands on disk is the DRAGGED shape, not the straight one
    4. reload it later as the initial state

WHY STEP 3 IS NOT JUST stage.Export(). Under GPU dynamics the simulated state
lives in Fabric, and the USD attributes keep returning the authored pose
forever. Exporting after dragging therefore saves the straight duct. Measured
on this scene: USD reported 0.000000 m of movement while the cloth had actually
fallen 1.0 m. duct_sim.freeze reads the live state (cloth points through usdrt,
rigid poses straight from PhysX, which is the half Fabric did NOT have) and
bakes it onto the stage before exporting.

    python scripts/duct_layout_gui.py --n-rings 40 --spacing 0.1
        drag things around, then press S to save

    python scripts/duct_layout_gui.py --load /tmp/track_layout.usd
        starts from the saved layout instead of a straight duct

    --save-after N   save automatically N steps in (for unattended capture)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--n-rings", type=int, default=20)
ap.add_argument("--spacing", type=float, default=0.05)
ap.add_argument("--height", type=float, default=0.6)
ap.add_argument("--n-circ", type=int, default=28)
ap.add_argument("--out", default="/tmp/track_layout.usd")
ap.add_argument("--load", default="", help="start from a previously saved layout")
ap.add_argument("--save-after", type=int, default=0,
                help="save automatically after N physics steps, then keep running")
ap.add_argument("--obstacles", action="store_true",
                help="drop a couple of posts in so the duct settles into a curve")
ap.add_argument("--frame", action="store_true")
ap.add_argument("--cam-dist", type=float, default=0.0)
ap.add_argument("--grab-force", type=float, default=5.0,
                help="/physics/pickingForce; UI slider tops out at 10")
ap.add_argument("--pull", action="append", default=[], metavar="IDX:FX,FY,FZ",
                help="apply a constant force [N] to hoop IDX, e.g. 10:0,40,0 . "
                     "Repeatable. This is the scripted equivalent of grabbing "
                     "a hoop with the mouse, and does not depend on any GUI "
                     "extension being present.")
# Softer than the duct scenes. Stretch is what stops the sleeve elongating;
# shear and bend are what make it feel limp. Lowering shear by 5x and bend by
# 10x is what "more floppy" actually means here -- dropping stretch instead
# just makes the duct grow longer, which is not the same thing.
ap.add_argument("--stretch", type=float, default=5.0e3)
ap.add_argument("--shear", type=float, default=2.0)
ap.add_argument("--bend", type=float, default=1.0e-3)
ap.add_argument("--thickness", type=float, default=0.003)
ap.add_argument("--friction", type=float, default=0.5)
# THINNER HOOPS NEED MORE SEGMENTS. The collider is a chain of capsules whose
# axes are CHORDS, so mid-chord the collider sits one sagitta (R*(1-cos(pi/n)))
# inside the circle the cloth vertices lie on. If the tube radius is smaller
# than that sagitta the seam vertices fall OUTSIDE the collider and bind
# nothing -- and create_auto_deformable_attachment still returns True. At the
# default 16 segments the floor is 3.84 mm; 32 segments drops it to 0.96 mm.
ap.add_argument("--ring-thickness", type=float, default=0.0,
                help="hoop tube radius [m]; 0 = spec default (0.008)")
ap.add_argument("--ring-segments", type=int, default=0,
                help="capsules per hoop; 0 = spec default (16). Raise this "
                     "whenever you lower --ring-thickness")
ap.add_argument("--ring-density", type=float, default=0.0,
                help="hoop density [kg/m^3]; 0 = spec default (400). A thin "
                     "hoop at plastic density weighs grams and gets shoved "
                     "around by the fabric; steel wire is ~7800")
ap.add_argument("--report-every", type=int, default=0,
                help="print hoop positions every N steps (bend measurement)")
ap.add_argument("--anchor-ends", action="store_true",
                help="make the first and last hoops static, so a pull BENDS the "
                     "duct instead of sliding the whole thing sideways")
ap.add_argument("--top-down", action="store_true",
                help="look from above; lateral bending is invisible from the side")
ap.add_argument("--no-drag", action="store_true",
                help="disable the custom shift+drag hoop grab")
ap.add_argument("--drag-stiffness", type=float, default=60.0)
ap.add_argument("--drag-max-force", type=float, default=40.0)
ap.add_argument("--pull-steps", type=int, default=0,
                help="stop applying --pull after this many steps (0 = forever)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False})

import carb  # noqa: E402
import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

from duct_sim.freeze import freeze_to_usd  # noqa: E402
from duct_sim.geometry import create_ring  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()

if args.load:
    ctx.open_stage(args.load)
    stage = ctx.get_stage()
    print(f"[layout] loaded {args.load}", flush=True)
else:
    ctx.new_stage()
    stage = ctx.get_stage()

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
if not stage.GetPrimAtPath("/World"):
    UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

# DuctSpec is a frozen dataclass -- assigning to its fields raises
# FrozenInstanceError, so overrides go through dataclasses.replace.
import dataclasses  # noqa: E402

spec = DuctSpec()
_over = {}
if args.ring_thickness:
    _over["ring_thickness"] = args.ring_thickness
if args.ring_segments:
    _over["ring_segments"] = args.ring_segments
if args.ring_density:
    _over["ring_density"] = args.ring_density
if _over:
    spec = dataclasses.replace(spec, **_over)
R, TUBE, H = spec.radius, spec.ring_thickness, args.height

# AUTO-RAISE, don't just warn. A warning scrolls past and the failure is
# silent: the seams bind nothing, the sleeves drop away, and the duct looks
# like bare hoops with empty gaps between them. Since the required segment
# count is exactly computable, just use it.
def _min_segments(tube_r, hoop_r, margin=2.0):
    """Segments needed so the chord sagitta stays under tube_r / margin."""
    want = tube_r / margin
    if want >= hoop_r:
        return 8
    return max(8, int(math.ceil(math.pi / math.acos(1.0 - want / hoop_r))))


_need = _min_segments(TUBE, R)
if spec.ring_segments < _need:
    print(f"[layout] tube radius {TUBE*1e3:.2f} mm needs >= {_need} segments "
          f"(had {spec.ring_segments}); raising it. Below that the seam "
          f"vertices fall outside the collider, bind nothing, and the fabric "
          f"silently drops away leaving empty gaps between the hoops.",
          flush=True)
    spec = dataclasses.replace(spec, ring_segments=_need)

_sagitta = R * (1.0 - math.cos(math.pi / spec.ring_segments))
print(f"[layout] hoop: tube r {TUBE*1e3:.2f} mm, {spec.ring_segments} segments, "
      f"density {spec.ring_density:.0f} -- chord sagitta {_sagitta*1e3:.2f} mm "
      f"(margin {TUBE/_sagitta:.1f}x)", flush=True)

if not stage.GetPrimAtPath("/World/physicsScene"):
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
scene_prim = stage.GetPrimAtPath("/World/physicsScene")
scene_prim.ApplyAPI("PhysxSceneAPI")
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
scene_prim.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                   Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

if not args.load:
    UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)

    span = args.spacing * (args.n_rings - 1)
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr(1.0)
    g = UsdGeom.Xformable(ground)
    g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    g.AddScaleOp().Set(Gf.Vec3f(span * 2 + 6, span * 2 + 6, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    if args.obstacles:
        # Posts stand in for whatever shapes the track boundary. Dragging by
        # mouse does the same job; these just make an unattended run settle
        # into something that is visibly NOT a straight duct.
        # Posts of alternating height along the duct axis. The duct drapes over
        # them into a wave -- a shape nothing about the authored geometry
        # contains, so a reload that starts wavy can only have come from the
        # saved SIMULATED state.
        for i, (px, ht) in enumerate(((-0.62, 0.85), (0.0, 0.42), (0.62, 0.85))):
            post = UsdGeom.Cylinder.Define(stage, f"/World/post_{i}")
            post.CreateAxisAttr("Z")
            post.CreateRadiusAttr(0.10)
            post.CreateHeightAttr(ht)
            post.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.35, 0.7)])
            UsdGeom.Xformable(post).AddTranslateOp().Set(Gf.Vec3d(px, 0.0, ht * 0.5))
            UsdPhysics.CollisionAPI.Apply(post.GetPrim())

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
        # DYNAMIC -- the whole point is that the mouse can move them. The end
        # hoops can be pinned instead: without an anchor a sideways pull just
        # accelerates the whole duct across the floor and nothing bends.
        is_end = args.anchor_ends and i in (0, args.n_rings - 1)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(
            0.0 if is_end else spec.ring_density)
        ring_paths.append(path)

    mat = "/World/cloth_material"
    deformableUtils.add_surface_deformable_material(
        stage, mat, dynamic_friction=args.friction,
        surface_thickness=args.thickness,
        surface_stretch_stiffness=args.stretch,
        surface_shear_stiffness=args.shear,
        surface_bend_stiffness=args.bend)
    print(f"[layout] cloth: stretch {args.stretch:.3g}, shear {args.shear:.3g}, "
          f"bend {args.bend:.3g}, thickness {args.thickness}", flush=True)

    n_ok = n_seams = n_bound = 0
    for k in range(args.n_rings - 1):
        x0, x1 = xs[k], xs[k + 1]
        L = max(2, round(args.spacing / (2 * math.pi * R / args.n_circ)))
        pts, tris = [], []
        for j in range(L + 1):
            x = x0 + (x1 - x0) * j / L
            for i in range(args.n_circ):
                a = 2 * math.pi * i / args.n_circ
                pts.append([x, R * math.cos(a), H + R * math.sin(a)])
        for j in range(L):
            for i in range(args.n_circ):
                i2 = (i + 1) % args.n_circ
                a, b = j * args.n_circ + i, j * args.n_circ + i2
                c, d = (j + 1) * args.n_circ + i2, (j + 1) * args.n_circ + i
                tris += [[a, b, c], [a, c, d]]
        root, skin = f"/World/sleeve_{k:02d}", f"/World/sleeve_{k:02d}/skin"
        UsdGeom.Xform.Define(stage, root)
        m = UsdGeom.Mesh.Define(stage, skin)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
        m.CreateDoubleSidedAttr(True)
        m.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.80, 0.10)])
        if not deformableUtils.create_auto_surface_deformable_hierarchy(
                stage, root_prim_path=root, simulation_mesh_path=f"{root}/simMesh",
                cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
                set_visibility_with_guide_purpose=True):
            continue
        n_ok += 1
        rp = stage.GetPrimAtPath(root)
        rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        for nm, val in (("physxDeformableBody:selfCollision", False),
                        ("physxDeformableBody:solverPositionIterationCount", 16),
                        ("physxDeformableBody:collisionPairUpdateFrequency", 4),
                        ("physxDeformableBody:collisionIterationMultiplier", 4)):
            a = rp.GetAttribute(nm)
            if a and a.IsValid():
                a.Set(val)
        physicsUtils.add_physics_material_to_prim(stage, rp, mat)
        for side, ri in ((0, k), (1, k + 1)):
            seam = f"{root}/seam_{side}"
            if deformableUtils.create_auto_deformable_attachment(
                    stage, target_attachment_path=Sdf.Path(seam),
                    attachable0_path=Sdf.Path(root),
                    attachable1_path=Sdf.Path(ring_paths[ri])):
                n_seams += 1
                # The return value is NOT evidence. It is True even when zero
                # vertices were bound. Count the children it actually authored.
                # NOT `sp` -- that name already holds the physics scene prim,
                # and shadowing it made the later
                # scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
                # resolve against a seam prim and kill the app with
                # "Empty typeName for .../seam_1.physxScene:enableGPUDynamics".
                seam_prim = stage.GetPrimAtPath(seam)
                n_bound += (len(seam_prim.GetChildren())
                            if seam_prim and seam_prim.IsValid() else 0)
    print(f"[layout] {args.n_rings} hoops @ {args.spacing} m, {n_ok} sleeves, "
          f"{n_seams} seams, span {span:.2f} m", flush=True)
    print(f"[layout] seam attachment elements authored: {n_bound} "
          f"({'OK' if n_bound else 'ZERO -- the fabric is NOT attached'})",
          flush=True)

if args.frame:
    import math as _m
    FOCAL, HAP = 20.0, 36.0
    VAP = HAP / (1280.0 / 800.0)
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(FOCAL)
    cam.CreateHorizontalApertureAttr(HAP)
    cam.CreateVerticalApertureAttr(VAP)
    # Frame the duct where it ENDS UP, not where the aim happened to land. The
    # first attempt aimed at z=0.25 while the duct rested near z=0.9, so it sat
    # against the top edge with the posts filling the frame.
    span_m = args.spacing * (args.n_rings - 1)
    extent = max(span_m, args.height) + 1.6
    target = Gf.Vec3d(0, 0, max(0.45, args.height * 0.35))
    vfov = 2.0 * _m.atan(VAP * 0.5 / FOCAL)
    hfov = 2.0 * _m.atan(HAP * 0.5 / FOCAL)
    dist = args.cam_dist or max(extent * 0.5 / _m.tan(hfov * 0.5),
                                extent * 0.5 / _m.tan(vfov * 0.5)) * 1.15
    # NOT straight down. SetLookAt degenerates when the view direction is
    # nearly antiparallel to the up vector, and a (0.02, -0.45, 0.89) eye
    # offset rendered an empty frame. ~40 degrees of elevation still shows
    # lateral bending clearly and keeps the basis well conditioned.
    offset = (Gf.Vec3d(0.02, -0.76, 0.65) if args.top_down
              else Gf.Vec3d(0.30, -0.86, 0.40))
    eye = target + offset.GetNormalized() * dist
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/cam"
    except Exception as exc:
        print(f"[layout] viewport retarget failed: {exc}", flush=True)

# --- KEEP THE BUILT-IN GRAB OFF ---
# Enabling omni.physx.ui was a mistake. Its grab calls
# PxRigidDynamic::addForce, which PhysX refuses under direct-GPU API (which
# cloth forces on), so every drag spewed
#     addForce(): it is illegal to call this method if
#     PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled!
# and the window died. The gesture is re-implemented in duct_sim/mouse_drag.py
# on RigidPrim.apply_forces, which IS legal, so the built-in one must stay off.
try:
    _s = carb.settings.get_settings()
    _s.set("/physics/mouseInteractionEnabled", False)
    _s.set("/physics/mouseGrab", False)
    _s.set("/physics/forceGrab", False)
    import omni.kit.app
    _mgr = omni.kit.app.get_app().get_extension_manager()
    if _mgr.is_extension_enabled("omni.physx.ui"):
        _mgr.set_extension_enabled_immediate("omni.physx.ui", False)
    print("[layout] built-in physx grab disabled (it crashes under GPU physics)",
          flush=True)
except Exception as exc:
    print(f"[layout] could not disable built-in grab: {exc}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

SENTINEL = "/tmp/duct_save_now"


def do_save(reason):
    n_pts, n_xf = freeze_to_usd(args.out, stage)
    print(f"[layout] SAVED ({reason}): {n_pts} meshes, {n_xf} transforms -> {args.out}",
          flush=True)


# Press S in the viewport to save. The sentinel file does the same thing, which
# is what an unattended recording uses.
try:
    import carb.input
    import omni.appwindow
    _inp = carb.input.acquire_input_interface()
    _kb = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_key(event, *_):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input == carb.input.KeyboardInput.S:
            do_save("S key")
        elif event.input == carb.input.KeyboardInput.R:
            # "stays where I bent it". The duct springs back because the solver
            # keeps pulling the fabric toward the shape it was BUILT as. There
            # is no plasticity setting on PhysX surface deformables, so instead
            # the rest shape is rewritten to whatever the duct looks like right
            # now -- after this the bend IS the relaxed state.
            try:
                from duct_sim.plastic import bake_and_restart
                n = bake_and_restart(stage, sim)
                print(f"[layout] REST SHAPE BAKED on {n} sleeves -- the current "
                      f"bend is now the shape it returns to", flush=True)
            except Exception as exc:
                print(f"[layout] bake failed: {exc}", flush=True)
        return True

    _sub = _inp.subscribe_to_keyboard_events(_kb, _on_key)
    print("[layout] keys: S = save layout to USD,  R = make the current bend "
          "permanent (rewrites the cloth rest shape)", flush=True)
except Exception as exc:
    print(f"[layout] keyboard hook unavailable ({exc}); use: touch {SENTINEL}", flush=True)

print("[layout] HOW TO BEND IT: hold SHIFT and left-drag one of the BLACK "
      "HOOPS (not the yellow fabric). This uses a custom grab -- the built-in "
      "one cannot work in a cloth scene, see duct_sim/mouse_drag.py.",
      flush=True)
print(f"[layout] running. drag, then press S (or touch {SENTINEL}) "
      f"to write {args.out}", flush=True)

# --- scripted pulls: the same thing the mouse grab does, without the GUI ---
# Forces, NOT velocities. Setting a velocity is effectively infinite strength,
# so the duct cannot bend against it and every stiffness reads the same -- that
# mistake made an entire earlier round of results meaningless.
pulls = []
for spec_str in args.pull:
    try:
        idx_s, vec_s = spec_str.split(":")
        fx, fy, fz = (float(v) for v in vec_s.split(","))
        pulls.append((int(idx_s), np.array([[fx, fy, fz]], dtype=np.float32)))
    except Exception as exc:
        print(f"[layout] bad --pull {spec_str!r}: {exc}", flush=True)

# ONE view over every pulled hoop, not one view per hoop. Each RigidPrim
# allocates GPU tensors, and building them in a loop (on top of another Isaac
# instance already holding 7 GB) hit "CUDA error: out of memory" before the
# first step.
# LAZY. Building a RigidPrim before the physics warm-up does not just fail to
# read -- it makes every later view read the AUTHORED pose too. Symptom: the
# bend report printed z=0.300 and y=0.000 on every sample while the recording
# showed the duct plainly moving. With no early view the same reader correctly
# showed z falling 0.60 -> 0.21.
pull_view, pull_forces, pull_specs = None, None, None
if pulls:
    paths, forces = [], []
    for idx, force in pulls:
        path = f"/World/ring_{idx:02d}"
        if not stage.GetPrimAtPath(path):
            print(f"[layout] --pull: no {path}", flush=True)
            continue
        paths.append(path)
        forces.append(force[0])
    if paths:
        pull_specs = (paths, np.array(forces, dtype=np.float32))
        print(f"[layout] will pull {paths} with {forces} N "
              f"(view built after warm-up)", flush=True)


def _build_pull_view():
    global pull_view, pull_forces
    paths, forces = pull_specs
    try:
        from isaacsim.core.prims import RigidPrim
        pull_view = RigidPrim(paths)
        pull_forces = forces
        # Print the mass. Forces were being picked by guesswork: 6 N on a hoop
        # that turned out to weigh ~0.09 kg is 64 m/s^2, which threw the duct
        # out of frame instead of bending it. The useful scale is comparable to
        # the hoop's own weight (m*g ~ 0.92 N).
        try:
            m = pull_view.get_masses()
            m = np.asarray(m.cpu() if hasattr(m, "cpu") else m).reshape(-1)
            print(f"[layout] pulled hoop masses {m.tolist()} kg "
                  f"(weight {(m * 9.81).tolist()} N)", flush=True)
        except Exception as exc:
            print(f"[layout] mass query failed: {exc}", flush=True)
        print(f"[layout] pulling {paths} with {pull_forces.tolist()} N "
              f"for {args.pull_steps or 'all'} steps", flush=True)
    except Exception as exc:
        print(f"[layout] could not build pull view: {exc}", flush=True)

# --- shift+drag, re-implemented on the legal force path ---
# The built-in omni.physx.ui grab is useless here: it calls
# PxRigidDynamic::addForce, which PhysX refuses under direct-GPU API -- and
# cloth forces cuda, which forces that flag on. So the gesture is rebuilt on
# RigidPrim.apply_forces instead.
dragger = None
if not args.no_drag:
    all_rings = sorted(str(pm.GetPath()) for pm in stage.Traverse()
                       if pm.GetName().startswith("ring_")
                       and pm.HasAPI(UsdPhysics.RigidBodyAPI))
    if all_rings:
        try:
            from duct_sim.mouse_drag import HoopDragger
            dragger = HoopDragger(all_rings,
                                  stiffness=args.drag_stiffness,
                                  max_force=args.drag_max_force)
            print(f"[layout] shift+drag active over {len(all_rings)} hoops "
                  f"(stiffness {args.drag_stiffness}, max {args.drag_max_force} N)",
                  flush=True)
        except Exception as exc:
            print(f"[layout] dragger unavailable: {exc}", flush=True)
    else:
        print("[layout] no hoops found for dragging", flush=True)

# BUILD THIS LATE. A RigidPrim created before the physics warm-up reports the
# AUTHORED pose forever: the first attempt printed z=0.30 and y=0.000 on every
# sample while the recording plainly showed the duct moving. Building it a few
# steps in (and initialising it) puts it on the tensor backend, which reads the
# real simulated state.
_report_view = None
_report_ring_paths = sorted(str(pm.GetPath()) for pm in stage.Traverse()
                            if pm.GetName().startswith("ring_")
                            and pm.HasAPI(UsdPhysics.RigidBodyAPI))


def _build_report_view():
    global _report_view
    try:
        from isaacsim.core.prims import RigidPrim
        v = RigidPrim(_report_ring_paths)
        try:
            v.initialize()
        except Exception:
            pass
        _report_view = v
        print(f"[bend] measuring {len(_report_ring_paths)} hoops every "
              f"{args.report_every} steps", flush=True)
    except Exception as exc:
        print(f"[bend] could not build report view: {exc}", flush=True)


step = 0
while app.is_running():
    if dragger is not None:
        try:
            dragger.update()
        except Exception as exc:
            print(f"[layout] drag error, disabling: {exc}", flush=True)
            dragger = None
    if pull_specs is not None and pull_view is None and step == 12:
        _build_pull_view()
    if pull_view is not None and (not args.pull_steps or step < args.pull_steps):
        try:
            # apply_forces goes through the tensor API, which is the path that
            # IS legal under eENABLE_DIRECT_GPU_API. PxRigidDynamic::addForce
            # -- what the built-in mouse grab uses -- is not.
            from duct_sim.mouse_drag import _to_backend
            pull_view.apply_forces(_to_backend(pull_forces), is_global=True)
        except Exception as exc:
            print(f"[layout] apply_forces failed: {exc}", flush=True)
            pull_view = None
    sim.step(render=True)
    step += 1
    if args.save_after and step == args.save_after:
        do_save(f"--save-after {args.save_after}")
    # Measure the bend instead of trusting a screenshot. Two recordings came
    # back as empty grey frames and it was not clear whether the duct had been
    # flung away, had never built, or was simply off camera -- numbers settle
    # that. Lateral deflection is what "bending" means for a track layout.
    if args.report_every and _report_view is None and step == 10:
        _build_report_view()
    if _report_view is not None and args.report_every and step % args.report_every == 0:
        try:
            ys = _report_view.get_world_poses()[0]
            ys = np.asarray(ys.cpu() if hasattr(ys, "cpu") else ys)
            lat = ys[:, 1]
            print(f"[bend] step {step:5d}  lateral y: "
                  f"min {lat.min():+.3f}  max {lat.max():+.3f}  "
                  f"deflection {lat.max() - lat.min():.3f} m   "
                  f"x span {ys[:, 0].min():+.2f}..{ys[:, 0].max():+.2f}  "
                  f"z {ys[:, 2].min():.2f}..{ys[:, 2].max():.2f}", flush=True)
        except Exception as exc:
            print(f"[bend] report failed: {exc}", flush=True)

    if step % 30 == 0 and os.path.exists(SENTINEL):
        os.remove(SENTINEL)
        do_save("sentinel")

app.close()
