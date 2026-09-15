"""Approach B, done natively: PhysX surface deformable cloth on PhysX hoops.

WHY THIS REPLACES THE NEWTON VERSION. The Newton hybrid existed because a
probe of `pxr.PhysxSchema` reported no cloth or attachment classes in Isaac Sim
6.1. That probe was wrong: `pxr.PhysxSchema` is a DEPRECATED shim carrying only
a subset, and Kit says so at startup ("pxr.PhysxSchema is deprecated - please
use PhysxSchema instead"). Cloth was not removed in 6.1, it was renamed --
particle cloth became SURFACE DEFORMABLE:

    PhysxParticleClothAPI      ->  OmniPhysicsDeformableBodyAPI
                                   + OmniPhysicsSurfaceDeformableSimAPI
                                   + PhysxSurfaceDeformableBodyAPI
    PhysxPhysicsAttachment     ->  OmniPhysicsVtxXformAttachment (a PRIM)

(The migration is spelled out in omni.physx.asset_validator's
deformableSchemaChecker.convert_PhysxParticleCloth, which rewrites old assets.)

Everything the Newton version had to hand-build is native here: the cloth
collides with the hoops and the ground because they are all PhysX colliders in
one scene, the coupling is two-way, there is a single solver, and Kit's mouse
grab still works because nothing left PhysX.

The schemas have no Python bindings on this build, so they are applied by name
with Usd.Prim.ApplyAPI() and their attributes created explicitly -- which is
exactly how the validator does it.
"""

from __future__ import annotations

import math

import numpy as np
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .build_a import build_duct_a
from .spec import DuctSpec


def ensure_gpu_dynamics(stage: Usd.Stage, scene_path: str = "/World/physicsScene"):
    """Deformables are GPU-only in PhysX; without this they silently do nothing.

        "Deformable Body feature is only supported on GPU. Please enable GPU
         dynamics flag in Property/Scene of physics scene!"

    and then every attachment to the cloth reports invalid actors, so the
    symptom is a frozen sleeve with the hoops falling away from it -- identical
    to the symptom of a missing CollisionAPI, and equally unrelated to the
    attachments themselves. Both have to be right before the cloth moves at all.
    """
    prim = stage.GetPrimAtPath(scene_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"no physics scene at {scene_path}")
    prim.ApplyAPI("PhysxSceneAPI")
    prim.CreateAttribute("physxScene:enableGPUDynamics",
                         Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("physxScene:broadphaseType",
                         Sdf.ValueTypeNames.Token).Set("GPU")
    return prim


def _attr(prim: Usd.Prim, name: str, sdf_type, value):
    a = prim.CreateAttribute(name, sdf_type)
    a.Set(value)
    return a


def build_duct_b_physx_segmented(
    stage: Usd.Stage,
    root_path: str = "/World/DuctBSeg",
    spec: DuctSpec | None = None,
    n_circ: int = 20,
    loops: int = 5,
    bulge: float = 0.16,
    cloth_mass_per_gap: float = 0.004,
    stretch_stiffness: float = 1.0e3,
    bend_stiffness: float = 1.0e-3,
    shear_stiffness: float = 1.0e1,
    thickness: float = 0.004,
    radius_clearance: float = 0.004,
    bend_limit_deg: float = 30.0,
    hoop_bend_stiffness: float = 1.0,
    hoop_bend_damping: float = 0.3,
    hide_hoops: bool = False,
) -> dict:
    """One SEPARATE cloth patch per gap, each sewn to its two bounding hoops.

    The single-sleeve version cannot keep the hoops covered at a tight fold: the
    attachments filter cloth-hoop collision at every seam, so nothing pushes the
    fabric back out once a bend pulls it inside the hoop radius, and no amount
    of clearance or slack converges.

    A per-gap patch changes the geometry of the problem. Each patch spans only
    0.1 m and is pinned at BOTH of its ends, so its slack has nowhere to go but
    outward -- it bulges into a corrugation instead of being able to pull itself
    taut across a corner. That is also how a real flex duct looks: a row of
    barrel-shaped bulges between the wires, not a smooth tube.

    THE COST IS REAL: this is `num_rings - 1` separate deformable bodies, each
    cooked and solved independently, instead of one. For a 2 m duct that is 20
    bodies. Whether it is worth it against the single sleeve is what the
    comparison render is for.
    """
    spec = spec or DuctSpec()
    ensure_gpu_dynamics(stage)

    info = build_duct_a(stage, root_path, spec,
                        bend_limit_deg=bend_limit_deg,
                        bend_stiffness=hoop_bend_stiffness,
                        bend_damping=hoop_bend_damping)
    rings = info["rings"]

    cache = UsdGeom.XformCache()

    def frame(i):
        cache.Clear()
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(rings[i]))
        t = m.ExtractTranslation()
        r = m.ExtractRotationMatrix()
        return (np.array([t[0], t[1], t[2]]),
                np.array([[r[0][0], r[0][1], r[0][2]],
                          [r[1][0], r[1][1], r[1][2]],
                          [r[2][0], r[2][1], r[2][2]]]))

    base_r = spec.radius + spec.ring_thickness + radius_clearance
    unit = np.array([[math.cos(2 * math.pi * i / n_circ),
                      math.sin(2 * math.pi * i / n_circ), 0.0]
                     for i in range(n_circ)])

    mat_path = f"{root_path}/fabric_material"
    deformableUtils.add_surface_deformable_material(
        stage, mat_path,
        dynamic_friction=0.5,
        surface_thickness=thickness,
        surface_stretch_stiffness=stretch_stiffness,
        surface_shear_stiffness=shear_stiffness,
        surface_bend_stiffness=bend_stiffness,
    )

    patches, attachments = [], []
    for k in range(spec.num_rings - 1):
        (c0, b0), (c1, b1) = frame(k), frame(k + 1)
        pts = []
        for s in range(loops + 1):
            t = s / loops
            c = c0 * (1 - t) + c1 * t
            b = b0 if t < 0.5 else b1
            # zero bulge at both seams, maximum in the middle: a corrugation
            rr = base_r * (1.0 + bulge * math.sin(math.pi * t))
            for u in unit:
                pts.append(c + b @ (u * rr))
        pts = np.array(pts)

        tris = []
        for j in range(loops):
            for i in range(n_circ):
                i2 = (i + 1) % n_circ
                a = j * n_circ + i
                bb = j * n_circ + i2
                cc = (j + 1) * n_circ + i2
                dd = (j + 1) * n_circ + i
                tris.append([a, bb, cc])
                tris.append([a, cc, dd])
        tris = np.array(tris, dtype=np.int32)

        patch_root = f"{root_path}/patch_{k:03d}"
        skin = f"{patch_root}/skin"
        UsdGeom.Xform.Define(stage, patch_root)
        m = UsdGeom.Mesh.Define(stage, skin)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.flatten().tolist()))
        m.CreateDoubleSidedAttr(True)

        if not deformableUtils.create_auto_surface_deformable_hierarchy(
                stage, root_prim_path=patch_root,
                simulation_mesh_path=f"{patch_root}/simMesh",
                cooking_src_mesh_path=skin,
                cooking_src_simplification_enabled=False,
                set_visibility_with_guide_purpose=True):
            raise RuntimeError(f"patch {k}: deformable hierarchy failed")

        pp = stage.GetPrimAtPath(patch_root)
        pp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        for name, val in (("physxDeformableBody:selfCollision", False),
                          ("physxDeformableBody:solverPositionIterationCount", 12)):
            a = pp.GetAttribute(name)
            if a and a.IsValid():
                a.Set(val)
        physicsUtils.add_physics_material_to_prim(stage, pp, mat_path)
        patches.append(patch_root)

        # sew this patch to BOTH of its bounding hoops
        for j, ring_idx in ((0, k), (1, k + 1)):
            att = f"{root_path}/seam_{k:03d}_{j}"
            if deformableUtils.create_auto_deformable_attachment(
                    stage, Sdf.Path(att), Sdf.Path(patch_root),
                    Sdf.Path(rings[ring_idx])):
                attachments.append(att)

    if hide_hoops:
        for rp in rings:
            UsdGeom.Imageable(stage.GetPrimAtPath(rp)).MakeInvisible()

    return {
        "root": root_path,
        "rings": rings,
        "patches": patches,
        "material": mat_path,
        "attachments": attachments,
        "spec": spec,
        "n_patches": len(patches),
    }


def build_duct_b_physx(
    stage: Usd.Stage,
    root_path: str = "/World/DuctB",
    spec: DuctSpec | None = None,
    n_circ: int = 24,
    # More loops per gap = the sleeve can follow a tighter curve. At 3 the
    # fabric has only 4 rings of vertices per 0.1 m and cannot round the
    # 180-degree fold over the bar; it cuts the corner and the hoops emerge.
    rings_per_gap: int = 5,
    cloth_mass: float = 0.15,          # kg for the whole sleeve
    # The sleeve barely moved at 5e4: a stiffness that high makes the fabric
    # behave like sheet metal, and the duct stayed folded over the bar instead
    # of draping. Real duct skin resists stretching but not bending, so stretch
    # stays high-ish while bend and shear go low.
    stretch_stiffness: float = 1.0e3,
    bend_stiffness: float = 1.0e-3,    # low = limp fabric
    shear_stiffness: float = 1.0e1,
    thickness: float = 0.004,
    # Slack is the material the fabric spends on the OUTSIDE of a bend. Over the
    # bar the duct folds ~180 degrees, so the outer surface needs noticeably
    # more length than a straight duct: 6% was not enough and the fabric pulled
    # itself taut across the fold, which is exactly where the hoops showed.
    slack: float = 0.15,
    # THE HOOPS MUST NEVER SHOW. They sit at radius+tube = 0.208; the sleeve
    # is drawn `radius_clearance` beyond that. 10 mm was not enough, because
    # the attachment helper FILTERS cloth-hoop collision at every seam (that is
    # what the element_filter prims it creates are for), so nothing pushes the
    # fabric back out when a tight bend pulls it inward. Clearance has to be
    # large enough that the fabric never needs pushing.
    radius_clearance: float = 0.030,
    bend_limit_deg: float = 30.0,
    hoop_bend_stiffness: float = 1.0,
    hoop_bend_damping: float = 0.3,
) -> dict:
    spec = spec or DuctSpec()
    ensure_gpu_dynamics(stage)

    info = build_duct_a(stage, root_path, spec,
                        bend_limit_deg=bend_limit_deg,
                        bend_stiffness=hoop_bend_stiffness,
                        bend_damping=hoop_bend_damping)
    rings = info["rings"]

    # ---- sleeve geometry, built on the hoops' CURRENT poses ----
    cache = UsdGeom.XformCache()

    def frame(i):
        cache.Clear()
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(rings[i]))
        t = m.ExtractTranslation()
        r = m.ExtractRotationMatrix()
        return (np.array([t[0], t[1], t[2]]),
                np.array([[r[0][0], r[0][1], r[0][2]],
                          [r[1][0], r[1][1], r[1][2]],
                          [r[2][0], r[2][1], r[2][2]]]))

    radius = spec.radius + spec.ring_thickness + radius_clearance
    loops_per_gap = rings_per_gap + 1
    unit = np.array([[math.cos(2 * math.pi * i / n_circ),
                      math.sin(2 * math.pi * i / n_circ), 0.0]
                     for i in range(n_circ)])

    pts, seam_loops = [], []
    loop = 0
    for k in range(spec.num_rings - 1):
        (c0, b0), (c1, b1) = frame(k), frame(k + 1)
        for s in range(loops_per_gap):
            t = s / loops_per_gap
            c = c0 * (1 - t) + c1 * t
            b = b0 if t < 0.5 else b1
            if s == 0:
                seam_loops.append(loop)
            rr = radius * (1.0 + slack * math.sin(math.pi * t))
            for u in unit:
                pts.append(c + b @ (u * rr))
            loop += 1
    seam_loops.append(loop)
    c, b = frame(spec.num_rings - 1)
    for u in unit:
        pts.append(c + b @ (u * radius))
    pts = np.array(pts)

    tris = []
    n_loops = len(pts) // n_circ
    for j in range(n_loops - 1):
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a = j * n_circ + i
            bb = j * n_circ + i2
            cc = (j + 1) * n_circ + i2
            dd = (j + 1) * n_circ + i
            tris.append([a, bb, cc])
            tris.append([a, cc, dd])
    tris = np.array(tris, dtype=np.int32)

    # ---- author the sleeve through the OFFICIAL helpers ----
    # Hand-applying OmniPhysicsDeformableBodyAPI to a single mesh does not work:
    # a surface deformable is a HIERARCHY -- a root Xform carrying the body API
    # plus a cooked SIMULATION mesh underneath, separate from the visible skin.
    # omni.physx's own SurfaceDeformableDemo builds it with
    # create_auto_surface_deformable_hierarchy(), and that is what is used here
    # rather than re-deriving the structure from the schema definitions.
    cloth_root = f"{root_path}/fabric"
    skin_path = f"{cloth_root}/skin"
    sim_mesh_path = f"{cloth_root}/simMesh"

    UsdGeom.Xform.Define(stage, cloth_root)
    mesh = UsdGeom.Mesh.Define(stage, skin_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(tris.flatten().tolist()))
    mesh.CreateDoubleSidedAttr(True)

    ok = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=cloth_root,
        simulation_mesh_path=sim_mesh_path,
        cooking_src_mesh_path=skin_path,
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        raise RuntimeError("create_auto_surface_deformable_hierarchy failed")

    root_prim = stage.GetPrimAtPath(cloth_root)
    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    for name, val in (
        ("physxDeformableBody:selfCollision", True),
        ("physxDeformableBody:solverPositionIterationCount", 16),
        ("physxDeformableBody:collisionPairUpdateFrequency", 4),
        ("physxDeformableBody:collisionIterationMultiplier", 4),
    ):
        a = root_prim.GetAttribute(name)
        if a and a.IsValid():
            a.Set(val)

    # ---- fabric material, also via the official helper ----
    mat_path = f"{root_path}/fabric_material"
    deformableUtils.add_surface_deformable_material(
        stage, mat_path,
        dynamic_friction=0.5,
        surface_thickness=thickness,
        surface_stretch_stiffness=stretch_stiffness,
        surface_shear_stiffness=shear_stiffness,
        surface_bend_stiffness=bend_stiffness,
    )
    physicsUtils.add_physics_material_to_prim(stage, root_prim, mat_path)

    # ---- sew to every hoop, again via the helper ----
    # HIDE THE HOOPS FROM RENDERING (they keep their colliders and joints).
    # On a real flexible duct the wire helix is bonded inside the skin -- what
    # you see is fabric with a corrugated profile, never bare wire. Here the
    # hoops are a simulation device for bending stiffness, not something the
    # duct is supposed to show. Chasing full coverage geometrically does not
    # converge: the seams have cloth-hoop collision filtered out by the
    # attachments, so at a 180-degree fold the fabric can always be pulled
    # inside the hoop radius no matter how much clearance or slack is added.
    #
    # This is a RENDERING choice, not a physics one: the hoops still collide
    # with the bar and the ground, still carry the joints, and still drive the
    # fabric through the seams.
    for rp in rings:
        UsdGeom.Imageable(stage.GetPrimAtPath(rp)).MakeInvisible()

    attachments = []
    for i, rp in enumerate(rings):
        att_path = f"{root_path}/seam_{i:03d}"
        if deformableUtils.create_auto_deformable_attachment(
                stage, Sdf.Path(att_path), Sdf.Path(cloth_root), Sdf.Path(rp)):
            attachments.append(att_path)

    cloth_path = skin_path

    return {
        "root": root_path,
        "rings": rings,
        "cloth": cloth_path,
        "material": mat_path,
        "attachments": attachments,
        "spec": spec,
        "n_vertices": len(pts),
        "n_triangles": len(tris),
        "n_seams": len(seam_loops),
    }
