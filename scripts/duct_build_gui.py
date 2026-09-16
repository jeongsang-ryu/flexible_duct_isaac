"""Track builder: spawn ducts on demand, drag them into place, save the layout.

The three steps, in one window:

    SPAWN   type a length in metres, press Spawn. Repeat as often as you like;
            each duct is independent and lands in front of the camera.
    PLACE   hold SHIFT and left-drag any black hoop.
    SAVE    writes a plain USD holding the ARRANGED shape.

Two things make this harder than it sounds, and both are handled here.

1. Under GPU physics the simulated state lives in Fabric and never returns to
   USD -- stage.Export() after arranging would silently save the straight
   ducts. duct_sim.freeze reads the live state (cloth points via usdrt, rigid
   poses straight from PhysX) and bakes it before writing.

2. PhysX will not necessarily accept a new deformable while the simulation is
   running. If it does not, spawning has to stop and restart the sim -- which
   would snap every already-placed duct back to where it was authored. So the
   spawn path freezes the live state onto the stage FIRST, and the restart then
   resumes from the arranged shape instead of the original one.

Bake (R) rewrites each sleeve's rest shape to its current shape, so a bend
stops springing back -- PhysX surface deformables have no plasticity setting,
so moving the target is the only way to get "stays where I put it".
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=float, default=2.0, help="default duct length [m]")
ap.add_argument("--spacing", type=float, default=0.0,
                help="hoop spacing in m. 0 = take it from the layout file, else 0.05. Hoop count is the dominant cost: 0.05 -> 0.10 halves the hoops and halves the step time, and 0.10 is what real flexible duct uses anyway.")
ap.add_argument("--height", type=float, default=0.5, help="spawn height [m]")
ap.add_argument("--n-circ", type=int, default=28)
ap.add_argument("--bend-in", type=int, default=0, metavar="STEPS",
                help="build the duct STRAIGHT and drive the hoops to the "
                     "layout over STEPS. Without this the duct is born curved, "
                     "so the corners carry zero strain and the creasing there "
                     "is whatever the mesh generator produced rather than what "
                     "the fabric would do. 600 is a reasonable start.")
ap.add_argument("--solid-hoop", action="store_true",
                help="one convex disc per hoop instead of 16 capsules. The "
                     "capsule chain is the most expensive thing in a track: "
                     "1,629 hoops x 16 = 26,064 shapes cost 17.4 ms/step with "
                     "no fabric at all, vs 8.4 ms for the whole deformable.")
ap.add_argument("--cloth-roughness", type=float, default=0.97,
                help="fabric surface roughness: 1.0 = pure matte cloth, "
                     "0.3 = glossy rubber. Prims with no bound material get "
                     "the renderer default, which looks like rubber.")
ap.add_argument("--ring-roughness", type=float, default=0.55,
                help="hoop surface roughness (0.55 = painted metal)")
ap.add_argument("--out", default="/home/js/hmcl_issac_project/duct_sim/layouts/track.usd")
ap.add_argument("--load", default="")
ap.add_argument("--layout", default="",
                help="track JSON exported from the planner: builds one duct "
                     "per drawn run, following the centreline")
ap.add_argument("--ground", type=float, default=40.0, help="ground plane size [m]")
ap.add_argument("--fem", action="store_true",
                help="volume (FEM) deformable: a solid low-res cylinder that "
                     "squashes and springs back. No hollow interior.")
ap.add_argument("--static", action="store_true",
                help="one swept tube mesh with capsule colliders; nothing moves")
ap.add_argument("--pusher", type=float, default=0.0,
                help="drop a draggable ball of this radius [m] to prod the duct "
                     "with. Needed for --fem and --static, which have no rigid "
                     "bodies for SHIFT+drag to grab. 0 = auto.")
ap.add_argument("--pusher-mass", type=float, default=8.0,
                help="mass of the pusher ball [kg]; it has to outweigh the duct "
                     "to shove it, the way a car does")
ap.add_argument("--fem-youngs", type=float, default=8.0e3,
                help="--fem: stiffness of the material [Pa]; lower = softer")
ap.add_argument("--fem-poisson", type=float, default=0.15,
                help="--fem: Poisson's ratio. THE BIG LEVER for a solid body "
                     "standing in for a hollow tube: near 0.5 the material is "
                     "incompressible and physically cannot squash, however low "
                     "the modulus. Low values let it give way.")
ap.add_argument("--taut", action="store_true",
                help="MEASURED TO DO NOTHING. Was meant to let hoop surfaces "
                     "pull fabric drawn inside them outwards; attachments "
                     "always preserve the gap they were built with, so the "
                     "fabric does not move. Kept so the result stays "
                     "reproducible (scripts/test_taut.py).")
ap.add_argument("--shrink-rest", type=float, default=0.0,
                help="cloth build: narrow the fabric's REST cross-section to "
                     "this factor a few steps in, so it pulls itself tight "
                     "against the hoops. 0.8 is a strong pull; 0 = off.")
ap.add_argument("--surface-sampling", type=float, default=0.02,
                help="--taut: spacing of attachment points sampled on the hoop")
ap.add_argument("--fem-rings", type=float, default=0.0,
                help="--fem: bond a hoop to the body every N metres so the duct "
                     "reads as ribbed. They deform with it. 0 = none.")
ap.add_argument("--fem-ring-tube", type=float, default=0.010)
ap.add_argument("--fem-damping", type=float, default=0.5,
                help="--fem: elasticity damping; raise it to stop the body "
                     "springing back like rubber")
ap.add_argument("--rigid", action="store_true",
                help="build the duct from black discs and yellow sleeves hinged "
                     "by compliant joints, with no cloth at all. ~16x fewer "
                     "collision shapes; cannot crumple or drape.")
ap.add_argument("--bend-limit", type=float, default=14.0,
                help="--rigid: bend allowed per joint [deg]")
ap.add_argument("--stiffness", type=float, default=30.0,
                help="--rigid: joint drive stiffness (the compliance)")
ap.add_argument("--damping", type=float, default=30.0)
ap.add_argument("--mass-per-m", type=float, default=0.5,
                help="--rigid: duct mass per metre [kg]. A real 400 mm flexible "
                     "duct is ~0.5; setting a density instead made it 23 kg/m.")
ap.add_argument("--body-damping", type=float, default=2.0,
                help="--rigid: linear/angular damping on each body; raise it if "
                     "the chain shivers instead of settling")
ap.add_argument("--solver-iters", type=int, default=32,
                help="--rigid: position iterations per body")
ap.add_argument("--posts", action="store_true",
                help="also build the planner's posts as static colliders. Off "
                     "by default: they shape the layout in 2-D and the drawn "
                     "centreline already carries the result.")
ap.add_argument("--perf-every", type=int, default=0,
                help="print ms/step and GPU memory every N steps")
ap.add_argument("--frame", action="store_true",
                help="aim the viewport at the whole arena")
ap.add_argument("--cam-dist", type=float, default=0.0)
ap.add_argument("--stretch", type=float, default=2.0e4)
ap.add_argument("--shear", type=float, default=2.0)
ap.add_argument("--bend", type=float, default=1.0e-3)
ap.add_argument("--thickness", type=float, default=0.003)
ap.add_argument("--friction", type=float, default=0.5)
ap.add_argument("--ring-thickness", type=float, default=0.0)
ap.add_argument("--ring-segments", type=int, default=0)
ap.add_argument("--ring-density", type=float, default=0.0)
ap.add_argument("--drag-stiffness", type=float, default=60.0)
ap.add_argument("--drag-max-force", type=float, default=40.0)
ap.add_argument("--segmented", action="store_true",
                help="old construction: one sleeve per gap, fabric drawn ON the "
                     "hoop centreline. The default is a single continuous "
                     "sleeve with the hoops inside it.")
ap.add_argument("--rib-visual", action="store_true",
                help="hide the capsule chain and draw a real torus at the "
                     "fabric radius, so the hoop reads as a rib on the duct "
                     "instead of a fat tube standing off it. Display only -- "
                     "the collider is unchanged.")
ap.add_argument("--rib-inset", type=float, default=0.0015,
                help="tuck the visual rib this far under the fabric [m] so it "
                     "does not pop through at tight folds")
ap.add_argument("--no-smooth-render", action="store_true",
                help="disable catmullClark subdivision on the fabric (render only)")
ap.add_argument("--no-ccd", action="store_true",
                help="disable speculative CCD on the fabric")
ap.add_argument("--rib-tube", type=float, default=0.0,
                help="visual rib cross-section radius [m]; 0 = 1.6x the hoop tube")
ap.add_argument("--self-collision", action="store_true",
                help="stop the fabric passing through itself (prevents the "
                     "tube folding inward permanently). Costs solver time.")
ap.add_argument("--self-collision-distance", type=float, default=0.0,
                help="self-contact filter distance [m]; 0 = auto (0.3x the "
                     "fabric's own vertex spacing)")
ap.add_argument("--filtering", type=float, default=-1.0,
                help="ring<->cloth collision filtering offset [m]. "
                     "0 = collision ON everywhere (no filtering), "
                     "-1 = match the seam capture band (default), "
                     "or give a value in metres. Toggle it live with C.")
ap.add_argument("--clearance", type=float, default=0.004,
                help="gap between fabric and hoop tube [m]. POSITIVE = fabric "
                     "outside, hoops hidden inside. NEGATIVE = fabric inside, "
                     "hoops visible as outer ribs (e.g. -0.004).")
ap.add_argument("--toggle-at", type=int, default=0,
                help="fire the collision toggle at this step (for testing)")
ap.add_argument("--bend-force", type=float, default=0.0,
                help="lateral force [N] on the middle hoops, to test whether "
                     "the hoops punch through the skin when the duct bends")
ap.add_argument("--bake-at", type=int, default=0,
                help="trigger Bake automatically at this step (for testing)")
ap.add_argument("--spawn-on-start", type=int, default=1,
                help="how many ducts to create immediately")
args = ap.parse_args()


from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False})

import dataclasses  # noqa: E402

import carb  # noqa: E402
import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.builder import (min_segments, rebuild_seams, spawn_duct,  # noqa: E402
                               spawn_duct_fem, spawn_duct_path,
                               spawn_duct_rigid, spawn_duct_single,
                               spawn_duct_static)
from duct_sim.freeze import freeze_to_usd  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402
from duct_sim import geometry as _geom  # noqa: E402

# MUST be after SimulationApp: duct_sim.geometry imports pxr at module level,
# and importing pxr before Kit starts leaves its extensions half-registered
# ("extension class wrapper for base class ... has not been created yet") and
# then segfaults ~2.7 s in. Nothing about setting a float needs to happen early.
_geom.CLOTH_ROUGHNESS = args.cloth_roughness
_geom.RING_ROUGHNESS = args.ring_roughness
from isaacsim.core.api import SimulationContext  # noqa: E402

ctx = omni.usd.get_context()
if args.load:
    ctx.open_stage(args.load)
    print(f"[build] loaded {args.load}", flush=True)
else:
    ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
if not stage.GetPrimAtPath("/World"):
    UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

# ---- spec, with the segment count forced high enough for the tube radius ----
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

_need = min_segments(spec.ring_thickness, spec.radius)
if spec.ring_segments < _need:
    print(f"[build] tube radius {spec.ring_thickness*1e3:.2f} mm needs >= {_need} "
          f"segments (had {spec.ring_segments}); raising. Below that the seam "
          f"vertices fall outside the collider and bind nothing, silently.",
          flush=True)
    spec = dataclasses.replace(spec, ring_segments=_need)
_sag = spec.radius * (1 - math.cos(math.pi / spec.ring_segments))
print(f"[build] hoop: tube r {spec.ring_thickness*1e3:.2f} mm, "
      f"{spec.ring_segments} segments, density {spec.ring_density:.0f}, "
      f"sagitta {_sag*1e3:.2f} mm (margin {spec.ring_thickness/_sag:.1f}x)",
      flush=True)

# ---- scene ----
if not stage.GetPrimAtPath("/World/physicsScene"):
    sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
scene_prim = stage.GetPrimAtPath("/World/physicsScene")
scene_prim.ApplyAPI("PhysxSceneAPI")
scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")
scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
# per-scene, NOT per-sleeve: scaling this with the sleeve count exhausted GPU
# memory and the solver then ran on garbage contact data
scene_prim.CreateAttribute("physxScene:gpuMaxDeformableSurfaceContacts",
                   Sdf.ValueTypeNames.UInt).Set(8 * 1048576)

if not args.load:
    UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800.0)
    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(g)
    gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(args.ground, args.ground, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

FEM_MAT = "/World/fem_material"
if args.fem and not stage.GetPrimAtPath(FEM_MAT):
    # DERIVE THE DENSITY FROM THE TARGET MASS. The FEM body is a SOLID cylinder
    # standing in for a hollow duct, so a density that sounds reasonable is
    # wildly wrong: 120 kg/m3 over pi*0.2^2*6 m makes a 6 m duct weigh 90 kg --
    # 15 kg/m against a real duct's ~0.5, and 45x the 2 kg ball meant to shove
    # it. Same trap as setting a density on the rigid sleeves; solid volumes
    # standing in for thin walls must have their mass set, not their density.
    _vol = math.pi * spec.radius ** 2 * args.length
    _fem_density = max(0.5, args.mass_per_m * args.length / _vol)
    deformableUtils.add_deformable_material(
        stage, FEM_MAT, dynamic_friction=0.6,
        youngs_modulus=args.fem_youngs, poissons_ratio=args.fem_poisson,
        density=_fem_density)
    # damping lives on the PhysX side of the material, not the USD physics one
    _fm = stage.GetPrimAtPath(FEM_MAT)
    _fm.ApplyAPI("PhysxDeformableMaterialAPI")
    _da = _fm.GetAttribute("physxDeformableMaterial:elasticityDamping")
    if _da and _da.IsValid():
        _da.Set(float(args.fem_damping))
    print(f"[build] FEM material: youngs {args.fem_youngs:.3g} Pa, "
          f"poisson {args.fem_poisson}, damping {args.fem_damping}, "
          f"density {_fem_density:.2f} kg/m3 -> "
          f"{args.mass_per_m * args.length:.1f} kg for {args.length} m",
          flush=True)

MAT = "/World/cloth_material"
if not stage.GetPrimAtPath(MAT):
    deformableUtils.add_surface_deformable_material(
        stage, MAT, dynamic_friction=args.friction,
        surface_thickness=args.thickness,
        surface_stretch_stiffness=args.stretch,
        surface_shear_stiffness=args.shear,
        surface_bend_stiffness=args.bend)
print(f"[build] cloth: stretch {args.stretch:.3g}, shear {args.shear:.3g}, "
      f"bend {args.bend:.3g}", flush=True)

# the built-in grab calls PxRigidDynamic::addForce, illegal under direct-GPU
# API (which cloth forces on) -- it errors every frame and kills the window
try:
    _s = carb.settings.get_settings()
    _s.set("/physics/mouseInteractionEnabled", False)
    _s.set("/physics/mouseGrab", False)
    import omni.kit.app
    _mgr = omni.kit.app.get_app().get_extension_manager()
    if _mgr.is_extension_enabled("omni.physx.ui"):
        _mgr.set_extension_enabled_immediate("omni.physx.ui", False)
except Exception as exc:
    print(f"[build] could not disable built-in grab: {exc}", flush=True)

sim = SimulationContext(stage_units_in_meters=1.0, device="cuda")
sim.initialize_physics()
scene_prim.GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim.play()

state = {"n_ducts": 0, "dragger": None, "rings": [], "pending_spawn": [],
         "pending_bake": False, "pending_filtering": None,
         "filtering": args.filtering}


def _next_index():
    i = 0
    while stage.GetPrimAtPath(f"/World/duct_{i:02d}"):
        i += 1
    return i


def _make_pusher(radius):
    """A heavy ball you can grab and shove into the duct.

    SHIFT+drag works by pushing RIGID BODIES, and --fem and --static have none
    -- the duct is a single deformable, or scenery. Without this there is
    literally nothing for the gesture to take hold of, which reads as "it will
    not deform" when in fact nothing was ever touching it.
    """
    path = "/World/pusher"
    if stage.GetPrimAtPath(path):
        return path
    b = UsdGeom.Sphere.Define(stage, path)
    b.CreateRadiusAttr(float(radius))
    b.CreateDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.45, 0.85)])
    UsdGeom.Xformable(b).AddTranslateOp().Set(
        Gf.Vec3d(0.0, -spec.radius * 4.0, radius))
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(b.GetPrim())
    UsdPhysics.MassAPI.Apply(b.GetPrim()).CreateMassAttr(float(args.pusher_mass))
    print(f"[build] pusher ball r={radius:.2f} m, {args.pusher_mass} kg "
          f"— SHIFT+drag it into the duct",
          flush=True)
    return path


def _refresh_dragger():
    """Rebuild the drag target list so newly spawned ducts are grabbable too."""
    rings = sorted(str(p.GetPath()) for p in stage.Traverse()
                   # --rigid names its bodies disc_/sleeve_, not ring_; without
                   # this the dragger finds nothing to grab in a rigid scene
                   # every body kind any build makes: cloth hoops, chain discs
                   # and sleeves, FEM bonded hoops, and the pusher ball. Missing
                   # a prefix here reads as "the duct will not move".
                   if p.GetName().startswith(("ring_", "disc_", "sleeve_",
                                              "hoop_", "pusher"))
                   and p.HasAPI(UsdPhysics.RigidBodyAPI))
    state["rings"] = rings
    if not rings:
        state["dragger"] = None
        return
    try:
        from duct_sim.mouse_drag import HoopDragger
        state["dragger"] = HoopDragger(rings, stiffness=args.drag_stiffness,
                                       max_force=args.drag_max_force)
        print(f"[build] drag active over {len(rings)} hoops", flush=True)
    except Exception as exc:
        print(f"[build] dragger unavailable: {exc}", flush=True)


def do_spawn(length_m):
    """Queued, not immediate -- see the loop: spawning mid-step is unsafe."""
    state["pending_spawn"].append(float(length_m))
    print(f"[build] spawn queued: {length_m:.2f} m", flush=True)


def _do_spawn_now(length_m):
    n_rings = max(2, int(round(length_m / (args.spacing or 0.05))) + 1)
    idx = _next_index()
    # stagger new ducts sideways so they do not land inside an existing one
    origin = (0.0, idx * 0.9, args.height)
    if args.rigid or args.fem or args.static:
        # a straight line of stations: the path builder's special case
        n = max(2, int(round(length_m / (args.spacing or 0.05))) + 1)
        st = [(-length_m / 2 + (args.spacing or 0.05) * k, origin[1], 0.0)
              for k in range(n)]
        if args.static:
            spawn_duct_static(stage, idx, st, spec)
            state["n_ducts"] = idx + 1
            _make_pusher(args.pusher or spec.radius * 0.8)
            _refresh_dragger()
            return
        if args.fem:
            spawn_duct_fem(stage, idx, st, spec, FEM_MAT, n_circ=args.n_circ,
                           ring_every=args.fem_rings,
                           ring_tube=args.fem_ring_tube)
            state["n_ducts"] = idx + 1
            _make_pusher(args.pusher or spec.radius * 0.8)
            _refresh_dragger()
            return
        spawn_duct_rigid(stage, idx, st, spec,
                         bend_limit_deg=args.bend_limit,
                         stiffness=args.stiffness, damping=args.damping,
                         mass_per_m=args.mass_per_m,
                         body_damping=args.body_damping,
                         solver_pos_iters=args.solver_iters)
        state["n_ducts"] = idx + 1
        _refresh_dragger()
        return
    try:
        fn = spawn_duct if args.segmented else spawn_duct_single
        kw = ({} if args.segmented
              else {"clearance": args.clearance,
                    "taut": args.taut,
                    "surface_sampling": args.surface_sampling,
                    "rib_visual": args.rib_visual,
                    "rib_tube": args.rib_tube,
                    "rib_inset": args.rib_inset,
                    "smooth_render": not args.no_smooth_render,
                    "speculative_ccd": not args.no_ccd,
                    "self_collision": args.self_collision,
                    "self_collision_distance": args.self_collision_distance,
                    "filtering_offset": (None if state["filtering"] < 0
                                         else state["filtering"])})
        fn(stage, idx, n_rings, (args.spacing or 0.05), spec, MAT,
           origin=origin, n_circ=args.n_circ, solid_hoop=args.solid_hoop, **kw)
        state["n_ducts"] = idx + 1
        _refresh_dragger()
    except Exception as exc:
        print(f"[build] spawn failed: {exc}", flush=True)


def do_save():
    n_pts, n_xf = freeze_to_usd(args.out, stage)
    print(f"[build] SAVED {n_pts} meshes / {n_xf} transforms -> {args.out}",
          flush=True)


def set_filtering(value):
    """Queue a seam rebuild with a new collision-filtering offset."""
    state["pending_filtering"] = float(value)
    print(f"[build] collision change queued: "
          f"{'ON everywhere' if value == 0 else 'filtered at the seams'}",
          flush=True)


def _apply_filtering_now(value):
    # Same stale-GPU-view hazard as bake: stop() tears down the simulation view.
    state["dragger"] = None
    try:
        from duct_sim.freeze import apply_to_stage, capture
        # keep whatever has been arranged so far; a restart would otherwise
        # snap every duct back to where it was authored
        apply_to_stage(capture(stage), stage, verbose=False)
    except Exception as exc:
        print(f"[build] pre-toggle freeze failed: {exc}", flush=True)
    try:
        sim.stop()
        rebuild_seams(stage, value)
        state["filtering"] = value
    except Exception as exc:
        import traceback
        print(f"[build] collision toggle FAILED: {exc}", flush=True)
        traceback.print_exc()
    finally:
        try:
            sim.play()
        except Exception:
            pass
        _refresh_dragger()


def do_bake():
    """Queued, never run inside a UI callback -- see the main loop."""
    state["pending_bake"] = True
    print("[build] bake queued", flush=True)


def _do_bake_now():
    # DROP EVERY GPU VIEW FIRST. bake_and_restart calls sim.stop()/sim.play(),
    # which tears down the physics simulation view. Anything still holding a
    # RigidPrim across that -- the dragger, in particular -- is left pointing at
    # freed GPU handles, and the next access takes the whole app down. That is
    # why pressing Bake killed the simulator.
    state["dragger"] = None
    try:
        from duct_sim.plastic import bake_and_restart
        n = bake_and_restart(stage, sim)
        print(f"[build] BAKED {n} sleeves -- the current bend is now the "
              f"resting shape", flush=True)
    except Exception as exc:
        import traceback
        print(f"[build] bake FAILED: {exc}", flush=True)
        traceback.print_exc()
        # make sure the sim is running again even if the bake threw midway
        try:
            sim.play()
        except Exception:
            pass
    finally:
        _refresh_dragger()


# ---- on-screen panel ----
_ui_ok = False
try:
    import omni.ui as ui

    # bottom-left, out of the way. Centred by default it sat on top of the
    # duct, which is the one thing you need to see while placing it.
    _win = ui.Window("Duct Builder", width=300, height=240,
                     position_x=12, position_y=470)
    with _win.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("1. spawn", height=18)
            with ui.HStack(spacing=6, height=26):
                ui.Label("length (m)", width=78)
                _len = ui.FloatField()
                _len.model.set_value(args.length)
            ui.Button("Spawn duct", height=30,
                      clicked_fn=lambda: do_spawn(_len.model.get_value_as_float()))
            ui.Spacer(height=4)
            ui.Label("2. place:  SHIFT + left-drag a black hoop", height=18)
            ui.Spacer(height=4)
            ui.Label("3. keep / save", height=18)
            with ui.HStack(spacing=6, height=26):
                ui.Label("ring<->cloth", width=78)
                ui.Button("collision ON", width=78,
                          clicked_fn=lambda: set_filtering(0.0))
                ui.Button("filtered", clicked_fn=lambda: set_filtering(-1.0))
            ui.Button("Bake bend (make it stay)", height=28, clicked_fn=do_bake)
            ui.Button("Save layout to USD", height=30, clicked_fn=do_save)
    _ui_ok = True
    print("[build] panel ready", flush=True)
except Exception as exc:
    print(f"[build] panel unavailable ({exc}); use keys N / R / S", flush=True)

try:
    import carb.input
    import omni.appwindow
    _inp = carb.input.acquire_input_interface()
    _kb = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_key(event, *_):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input == carb.input.KeyboardInput.N:
            do_spawn(args.length)
        elif event.input == carb.input.KeyboardInput.R:
            do_bake()
        elif event.input == carb.input.KeyboardInput.S:
            do_save()
        elif event.input == carb.input.KeyboardInput.C:
            # toggle between "collision on everywhere" and "filtered at seams"
            set_filtering(0.0 if state["filtering"] != 0.0 else -1.0)
        return True

    _sub = _inp.subscribe_to_keyboard_events(_kb, _on_key)
except Exception as exc:
    print(f"[build] keyboard unavailable: {exc}", flush=True)

_benders = []          # (rings, goal stations) per duct, layout mode only
if args.layout:
    # a drawn layout replaces the default straight spawn entirely
    from duct_sim.layout import describe, load, posts, resample

    _doc = load(args.layout)
    # an explicit --spacing overrides the file; the file only fills the gap
    _sp = float(args.spacing or _doc.get("duct", {}).get("hoop_spacing", 0.05))
    print(f"[build] hoop spacing {_sp:.3f} m "
          f"({'--spacing' if args.spacing else 'from layout'})", flush=True)
    print(f"[build] layout {args.layout}", flush=True)
    print(describe(_doc), flush=True)

    # POSTS ARE A PLANNER DEVICE, NOT SCENE GEOMETRY. They exist to shape the
    # centreline while you drag it in 2-D; once the run is drawn, the hoop
    # positions already carry that shape, so nothing has to hold it up in the
    # simulation. Building them by default just added 27 colliders and a row of
    # obstacles nobody asked for. --posts puts them in if you do want them.
    for _pi, (_px, _py, _pr) in enumerate(posts(_doc) if args.posts else []):
        _pp = f"/World/post_{_pi:02d}"
        _cyl = UsdGeom.Cylinder.Define(stage, _pp)
        _cyl.CreateAxisAttr("Z")
        _cyl.CreateRadiusAttr(float(_pr))
        _cyl.CreateHeightAttr(0.8)
        _cyl.CreateDisplayColorAttr().Set([Gf.Vec3f(0.70, 0.25, 0.17)])
        UsdGeom.Xformable(_cyl).AddTranslateOp().Set(Gf.Vec3d(_px, _py, 0.4))
        UsdPhysics.CollisionAPI.Apply(_cyl.GetPrim())
    _np_posts = len(posts(_doc))
    if args.posts and _np_posts:
        print(f"[build] {_np_posts} static post(s) placed", flush=True)
    elif _np_posts:
        print(f"[build] {_np_posts} planner post(s) ignored (layout aid only; "
              f"--posts builds them)", flush=True)
    for _i, _run in enumerate(_doc["runs"]):
        _st = resample(_run.get("points", []), _sp)
        if len(_st) < 2:
            print(f"[build] run {_i} has too few points; skipped", flush=True)
            continue
        _goal = _st
        if args.bend_in > 0:
            from duct_sim.bend import straight_stations  # noqa: E402
            _st = straight_stations(_st)
        if args.rigid:
            spawn_duct_rigid(stage, _i, _st, spec,
                             bend_limit_deg=args.bend_limit,
                             stiffness=args.stiffness, damping=args.damping,
                             mass_per_m=args.mass_per_m,
                             body_damping=args.body_damping,
                             solver_pos_iters=args.solver_iters)
            state["n_ducts"] = _i + 1
            continue
        _root, _rings, _, _ = spawn_duct_path(
            stage, _i, _st, spec, MAT,
            n_circ=args.n_circ, clearance=args.clearance,
            self_collision=args.self_collision,
            self_collision_distance=args.self_collision_distance,
            rib_visual=args.rib_visual, rib_tube=args.rib_tube,
            rib_inset=args.rib_inset,
            smooth_render=not args.no_smooth_render,
            speculative_ccd=not args.no_ccd,
            taut=args.taut, surface_sampling=args.surface_sampling,
            solid_hoop=args.solid_hoop,
            filtering_offset=(None if state["filtering"] < 0
                              else state["filtering"]))
        if args.bend_in > 0:
            _benders.append((_rings, _goal))
        state["n_ducts"] = _i + 1
    _refresh_dragger()
    if _benders:
        print(f"[bend] built straight; {len(_benders)} duct(s) will bend into "
              f"the layout over {args.bend_in} steps", flush=True)
else:
    for _ in range(int(args.spawn_on_start)):
        do_spawn(args.length)

if args.frame:
    # SetLookAt builds the WORLD-to-camera matrix, so the camera transform is
    # its inverse. Lens is authored explicitly: UsdGeom.Camera defaults to
    # 50 mm, and assuming 28 once put the subject 2.3x too close.
    _W, _D = 30.0, 20.0
    if args.layout:
        try:
            import json as _json
            _a = _json.load(open(args.layout)).get("arena", {})
            _W = float(_a.get("width", _W)); _D = float(_a.get("depth", _D))
        except Exception:
            pass
    FOCAL, HAP = 20.0, 36.0
    VAP = HAP / (820.0 / 460.0)
    _cam = UsdGeom.Camera.Define(stage, "/World/cam")
    _cam.CreateFocalLengthAttr(FOCAL)
    _cam.CreateHorizontalApertureAttr(HAP)
    _cam.CreateVerticalApertureAttr(VAP)
    _tgt = Gf.Vec3d(0, 0, 0.3)
    _hf = 2.0 * math.atan(HAP * 0.5 / FOCAL)
    _vf = 2.0 * math.atan(VAP * 0.5 / FOCAL)
    _dist = args.cam_dist or max((_W * 1.12) * 0.5 / math.tan(_hf * 0.5),
                                 (_D * 1.12) * 0.5 / math.tan(_vf * 0.5))
    _eye = _tgt + Gf.Vec3d(0.02, -0.62, 0.78).GetNormalized() * _dist
    _xf = UsdGeom.Xformable(_cam)
    _xf.ClearXformOpOrder()
    _xf.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(_eye, _tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
    _look = Gf.Matrix4d().SetLookAt(_eye, _tgt, Gf.Vec3d(0, 0, 1)).GetInverse()
    _shown = False
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/cam"
        _shown = True
    except Exception as exc:
        print(f"[build] viewport retarget failed: {exc}", flush=True)

    # Retargeting the viewport reports success and then silently does not stick
    # on a large stage -- the header still says Perspective and the shot comes
    # back as empty floor. Driving the persp camera itself always holds, so do
    # both. It lives on the session layer, which has to be the edit target or
    # the op is authored somewhere the viewport never reads.
    try:
        _persp = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if _persp:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                _pxf = UsdGeom.Xformable(_persp)
                for _op in _pxf.GetOrderedXformOps():
                    _persp.RemoveProperty(_op.GetOpName())
                _pxf.ClearXformOpOrder()
                _pxf.AddTransformOp().Set(_look)
                UsdGeom.Camera(_persp).CreateFocalLengthAttr(FOCAL)
                UsdGeom.Camera(_persp).CreateHorizontalApertureAttr(HAP)
            _shown = True
    except Exception as exc:
        print(f"[build] persp camera move failed: {exc}", flush=True)

    # READ IT BACK. Both the viewport retarget and the persp override report
    # success and then do not hold -- Kit's camera manipulator rewrites the
    # persp transform every frame from its own state. Believing the success
    # message put three empty-floor screenshots on the record.
    try:
        from omni.kit.viewport.utility import get_active_viewport
        _vp_now = str(get_active_viewport().camera_path)
    except Exception as exc:
        _vp_now = f"<unreadable: {exc}>"
    print(f"[build] camera {_dist:.1f} m back, framing {_W} x {_D} m; "
          f"eye ({_eye[0]:.1f}, {_eye[1]:.1f}, {_eye[2]:.1f}); "
          f"viewport is on {_vp_now}", flush=True)
    _camera_wanted = "/World/cam"

print(f"[build] running. N/panel = spawn, SHIFT+drag = place, "
      f"R = bake, S = save -> {args.out}", flush=True)

def _gpu_mb():
    """Resident GPU memory for this process, or -1. nvidia-smi is the only
    source that sees the renderer as well as the solver."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4).stdout
        mine = str(os.getpid())
        for line in out.splitlines():
            pid, mb = [t.strip() for t in line.split(",")[:2]]
            if pid == mine:
                return int(mb)
    except Exception:
        pass
    return -1


import time  # noqa: E402

_t0 = time.perf_counter()
_acc = 0.0
_perf_last_acc, _perf_last_step = 0.0, 0
step = 0
while app.is_running():
    if state["pending_spawn"]:
        length = state["pending_spawn"].pop(0)
        # FREEZE FIRST. If PhysX refuses a new deformable mid-run we have to
        # stop and restart, and a restart snaps every placed duct back to its
        # authored pose -- unless the arranged pose has been written onto the
        # stage beforehand. Freezing costs nothing when no restart happens.
        try:
            if state["n_ducts"]:
                from duct_sim.freeze import apply_to_stage, capture
                apply_to_stage(capture(stage), stage, verbose=False)
        except Exception as exc:
            print(f"[build] pre-spawn freeze failed: {exc}", flush=True)
        state["dragger"] = None      # same stale-GPU-view hazard as bake
        sim.stop()
        _do_spawn_now(length)
        sim.play()
        print(f"[build] duct count now {state['n_ducts']}", flush=True)

    if args.bend_force and step == 120 and state["rings"]:
        try:
            from isaacsim.core.prims import RigidPrim
            from duct_sim.mouse_drag import _to_backend
            state["bend_view"] = RigidPrim(state["rings"])
            f = np.zeros((len(state["rings"]), 3), dtype=np.float32)
            mid = len(state["rings"]) // 2
            for j in range(max(0, mid - 2), min(len(state["rings"]), mid + 3)):
                f[j, 1] = args.bend_force
            state["bend_forces"] = f
            state["_to_backend"] = _to_backend
            print(f"[build] bending: {args.bend_force} N on hoops around {mid}",
                  flush=True)
        except Exception as exc:
            print(f"[build] bend setup failed: {exc}", flush=True)
    if state.get("bend_view") is not None and 120 < step < 900:
        try:
            state["bend_view"].apply_forces(
                state["_to_backend"](state["bend_forces"]), is_global=True)
        except Exception as exc:
            print(f"[build] bend force failed: {exc}", flush=True)
            state["bend_view"] = None

    if args.bake_at and step == args.bake_at:
        print(f"[build] --bake-at {args.bake_at} firing", flush=True)
        state["pending_bake"] = True
    if args.toggle_at and step == args.toggle_at:
        print(f"[build] --toggle-at {args.toggle_at} firing", flush=True)
        set_filtering(0.0 if state["filtering"] != 0.0 else -1.0)

    if state["pending_filtering"] is not None:
        v = state["pending_filtering"]
        state["pending_filtering"] = None
        _apply_filtering_now(v)

    if args.shrink_rest > 0 and step == 150 and not state.get("shrunk"):
        state["shrunk"] = True
        state["dragger"] = None          # stale GPU views across stop/play
        try:
            from duct_sim.plastic import shrink_rest_shape
            n = shrink_rest_shape(stage, factor=args.shrink_rest, axis=0)
            sim.stop()
            sim.play()
            print(f"[build] rest cross-section narrowed to "
                  f"{args.shrink_rest:.2f} on {n} sleeve(s)", flush=True)
        except Exception as exc:
            import traceback
            print(f"[build] shrink failed: {exc}", flush=True)
            traceback.print_exc()
            try:
                sim.play()
            except Exception:
                pass
        finally:
            _refresh_dragger()

    if state["pending_bake"]:
        state["pending_bake"] = False
        _do_bake_now()

    d = state["dragger"]
    if d is not None:
        try:
            d.update()
        except Exception as exc:
            print(f"[build] drag error: {exc}", flush=True)
            state["dragger"] = None
    if args.frame and step < 120 and step % 10 == 0:
        # Re-assert every 10 frames through start-up: one set at build time is
        # overwritten while the stage is still loading.
        try:
            from omni.kit.viewport.utility import get_active_viewport as _gav
            _vp = _gav()
            if str(_vp.camera_path) != _camera_wanted:
                _vp.camera_path = _camera_wanted
                if step == 0:
                    print(f"[build] re-asserted camera -> {_vp.camera_path}",
                          flush=True)
        except Exception:
            pass
    if _benders and step == 30 and not state.get("bend_started"):
        # AFTER warm-up. A RigidPrim built before physics has run reports
        # authored poses forever and poisons every later read.
        state["bend_started"] = True
        from duct_sim.bend import Bender  # noqa: E402
        state["bend"] = [Bender(stage, r, g, steps=args.bend_in)
                         for r, g in _benders]
        state["dragger"] = None            # rebuilt once the bend finishes
    if state.get("bend"):
        _alive = [b for b in state["bend"] if b.update()]
        if not _alive:
            state["bend"] = None
            print("[bend] done; the corners are now solver output, "
                  "not mesh output", flush=True)
            _refresh_dragger()
        else:
            state["bend"] = _alive
    _a = time.perf_counter()
    sim.step(render=True)
    _acc += time.perf_counter() - _a
    step += 1
    if args.perf_every and step % args.perf_every == 0:
        _mb = _gpu_mb()
        # MARGINAL, not cumulative. A cumulative average never sheds start-up
        # -- cooking, the first contacts, a stop/play restart -- so it keeps
        # reporting a rate the sim is no longer running at. Only the interval
        # since the last report says how fast it is going NOW.
        _win = _acc - _perf_last_acc
        _wsteps = step - _perf_last_step
        print(f"[perf] {step:6d} steps  {_win / _wsteps * 1e3:7.1f} ms/step "
              f"marginal  ({_acc / step * 1e3:7.1f} cumulative)  "
              f"{_wsteps / _win:5.1f} fps  GPU {_mb} MiB", flush=True)
        _perf_last_acc, _perf_last_step = _acc, step

app.close()
