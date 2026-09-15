"""Build ONE duct under its own prim, so several can coexist on a stage.

The layout tool used to inline this and hard-code /World/ring_NN, which allowed
exactly one duct per session. Track building needs many, spawned on demand, so
each duct now lives under /World/duct_NN/ with its own hoops and sleeves.

The sharp edges here are all ones this project has already paid for:

  * the hoop collider is a ring of capsules whose axes are CHORDS, so a cloth
    vertex on the circle sits up to one sagitta, R*(1-cos(pi/n)), outside the
    collider. Thinner tube => more segments, or the seams bind NOTHING while
    create_auto_deformable_attachment still returns True.
  * the sleeve must be drawn ON the hoop centreline. Offsetting it outward to
    hide the hoops is what made an earlier version bind zero vertices.
  * seam success is counted from the attachment prims actually authored, never
    from that return value.
"""

from __future__ import annotations

import math

import numpy as np
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

from omni.physx.scripts.ifaces import get_physx_attachment_private_interface

from duct_sim import geometry as _geom
from duct_sim.geometry import create_ring, create_ring_visual, bind_look


def create_seam(stage, seam_path, cloth_path, rigid_path,
                overlap_offset=0.006, filtering_offset=0.012,
                rigid_surface=False, surface_sampling=0.02):
    """Attach cloth to a hoop, with the two offsets the one-shot helper leaves unset.

    deformableUtils.create_auto_deformable_attachment authors only the two
    relationships and then runs the setup, so every other knob keeps its schema
    default -- and two of those defaults are actively wrong for a duct:

      collisionFilteringOffset      default -inf
          enableCollisionFiltering is True, but an offset of -inf means NO
          vertex is ever close enough to earn a filtering id. So filtering is
          nominally on and actually off, and on play the hoop's collider
          depenetrates the fabric that the seam is holding -- the fabric visibly
          pops away from the ring the moment the simulation starts, which is
          exactly the reported "no gap while stopped, gap as soon as I press
          play".

      deformableVertexOverlapOffset default 0.0
          only vertices strictly INSIDE the collider attach. For an 8 mm tube
          that is survivable; for a 2 mm tube it is marginal. A positive offset
          widens the capture band so thin hoops still hold their fabric.

    The offsets have to be authored BEFORE setup_auto_deformable_attachment
    runs, because that call is what generates the attachment data.
    """
    scope = UsdGeom.Scope.Define(stage, seam_path)
    prim = scope.GetPrim()
    if not prim.ApplyAPI("PhysxAutoDeformableAttachmentAPI"):
        return False
    prim.GetRelationship("physxAutoDeformableAttachment:attachable0").SetTargets(
        [Sdf.Path(cloth_path)])
    prim.GetRelationship("physxAutoDeformableAttachment:attachable1").SetTargets(
        [Sdf.Path(rigid_path)])
    # Filtering is a trade. It stops the hoop shoving the fabric it holds, but
    # it ALSO removes the barrier that stops the hoop passing through the
    # fabric -- which is why a wide filter makes the rings poke out through the
    # skin as soon as the duct bends. With the fabric drawn OUTSIDE the hoop
    # there is no penetration to suppress, so filtering should be off and
    # ordinary collision should do its job.
    enable_filtering = filtering_offset > 0.0
    for name, val in (
            ("physxAutoDeformableAttachment:enableCollisionFiltering", enable_filtering),
            ("physxAutoDeformableAttachment:collisionFilteringOffset",
             float(filtering_offset) if enable_filtering else 0.0),
            ("physxAutoDeformableAttachment:deformableVertexOverlapOffset", float(overlap_offset)),
            ("physxAutoDeformableAttachment:enableDeformableVertexAttachments", True),
            # MEASURED: THIS DOES NOT PULL THE FABRIC. Sampling points on the
            # rigid surface adds attachment points, but each one still targets
            # wherever the fabric already was -- localPositionsSrc1 records the
            # offset at creation. Fabric drawn 50 mm inside the hoops measured
            # 146.0 mm at the hoop with this off AND with it on, identical to
            # the decimal. An attachment cannot express "constrain as if these
            # were touching"; it always preserves the existing gap. Kept only so
            # the negative result is reproducible -- see scripts/test_taut.py.
            ("physxAutoDeformableAttachment:enableRigidSurfaceAttachments",
             bool(rigid_surface)),
            ("physxAutoDeformableAttachment:rigidSurfaceSamplingDistance",
             float(surface_sampling))):
        a = prim.GetAttribute(name)
        if a and a.IsValid():
            a.Set(val)
    return bool(get_physx_attachment_private_interface()
                .setup_auto_deformable_attachment(str(seam_path)))


def min_segments(tube_r: float, hoop_r: float, margin: float = 2.0) -> int:
    """Segments needed to keep the chord sagitta under tube_r / margin."""
    want = tube_r / margin
    if want >= hoop_r:
        return 8
    return max(8, int(math.ceil(math.pi / math.acos(1.0 - want / hoop_r))))


def _hoop_mass(prim, spec, solid):
    """Give a hoop its mass, never a density when it is a solid disc.

    A density on a SOLID stand-in for a thin shell is wrong by the ratio of the
    volumes, and that ratio is large: the disc is pi*(R+t)^2*2t against the
    ring's 2*pi^2*R*t^2 -- 3,389 cm3 against 568, so 6.0x. A 3 m duct went from
    7 kg to 42 kg and fell over like dominoes the moment it was played.

    Third time this exact trap has bitten: 195 kg/m3 on solid rigid sleeves made
    a 139 kg duct, and 120 kg/m3 on a solid FEM cylinder made a 90 kg one. The
    rule that came out of those is the rule here -- a solid volume standing in
    for a thin wall has its MASS set, not its density.
    """
    import math as _m
    api = UsdPhysics.MassAPI.Apply(prim)
    if not solid:
        api.CreateDensityAttr(spec.ring_density)
        return
    R = spec.diameter * 0.5
    t = getattr(spec, "ring_tube", 0.012)
    torus_volume = 2.0 * _m.pi ** 2 * R * t ** 2
    api.CreateMassAttr(float(torus_volume * spec.ring_density))

def spawn_duct_single(
    stage,
    index: int,
    n_rings: int,
    spacing: float,
    spec,
    material_path: str,
    origin=(0.0, 0.0, 0.5),
    heading_deg: float = 0.0,
    n_circ: int = 28,
    clearance: float = 0.004,
    filtering_offset=None,          # None = match the capture band
    self_collision: bool = False,
    self_collision_distance: float = 0.0,
    rib_visual: bool = False,
    rib_tube: float = 0.0,
    rib_inset: float = 0.0015,
    smooth_render: bool = True,
    speculative_ccd: bool = True,
    taut: bool = False,
    surface_sampling: float = 0.02,
    ring_colour=(0.03, 0.03, 0.03),
    cloth_colour=(0.78, 0.60, 0.05),
    weave_tile: float = 0.04,       # metres of duct per weave texture tile
    anchor_ends: bool = False,
    solid_hoop: bool = False,     # one convex disc per hoop, not 16 capsules
    verbose: bool = True,
):
    """ONE continuous fabric tube with the hoops INSIDE it, sewn along the outside.

    This replaces the per-gap sleeve construction, and it fixes the gap problem
    at the root rather than working around it.

    The old design had a contradiction built in. A seam only binds cloth
    vertices that lie INSIDE the hoop's collision shape -- but "inside a
    collider" is precisely what the solver treats as penetration, so on play it
    pushed those same vertices back out. Fabric that the seam was holding was
    being shoved off the hoop by the hoop itself. Tuning collisionFilteringOffset
    suppresses the symptom; it does not remove the contradiction.

    Here the fabric is drawn just OUTSIDE the hoop tube -- radius
    R + ring_thickness + clearance -- exactly as real ducting wraps the outside
    of its wire hoops. Nothing penetrates anything, so there is no depenetration
    to fight, and the seam still binds because deformableVertexOverlapOffset
    attaches vertices that are merely NEAR the collider rather than inside it.

    Two more things fall out of it:
      * the fabric is one continuous mesh, so there is no sleeve-to-sleeve
        boundary at every hoop that can open up
      * the hoops sit inside the fabric, so the black skeleton stops showing
        through the yellow skin
    """
    R, TUBE = spec.radius, spec.ring_thickness
    root = f"/World/duct_{index:02d}"
    UsdGeom.Xform.Define(stage, root)

    span = spacing * (n_rings - 1)
    ox, oy, oz = origin
    th = math.radians(heading_deg)
    ux, uy = math.cos(th), math.sin(th)

    ring_paths = []
    for i in range(n_rings):
        t = -span / 2 + spacing * i
        path = f"{root}/ring_{i:03d}"
        create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments,
                    solid=solid_hoop)
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(ox + ux * t, oy + uy * t, oz))
        q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
             * Gf.Rotation(Gf.Vec3d(0, 0, 1), heading_deg))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
        for seg in prim.GetChildren():
            UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
            bind_look(stage, seg, ring_colour, roughness=_geom.RING_ROUGHNESS, ior=_geom.RING_IOR)
        if rib_visual:
            # hide the capsule chain and draw a proper torus instead. The
            # capsules stay as the collider; only what you SEE changes.
            for seg in prim.GetChildren():
                UsdGeom.Imageable(seg).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        is_end = anchor_ends and i in (0, n_rings - 1)
        if is_end:
            UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(0.0)
        else:
            _hoop_mass(prim, spec, solid_hoop)
        ring_paths.append(path)

    # --- the single sleeve, offset clear of the hoop tube ---
    # clearance > 0 : fabric OUTSIDE the hoops (hoops hidden inside)
    # clearance < 0 : fabric INSIDE the hoops (hoops visible as outer ribs)
    # Either way the fabric never sits ON the tube, which is the whole point --
    # overlapping the collider is what made the solver push them apart.
    cloth_r = (R + TUBE + clearance) if clearance >= 0 else (R - TUBE + clearance)
    circ_step = 2 * math.pi * cloth_r / n_circ
    # at least one free row of vertices between hoops, or the fabric is fully
    # pinned and cannot drape at all
    div = max(2, round(spacing / circ_step))
    n_stations = (n_rings - 1) * div + 1

    pts, tris = [], []
    for j in range(n_stations):
        t = -span / 2 + span * j / (n_stations - 1)
        cx, cy = ox + ux * t, oy + uy * t
        for i in range(n_circ):
            a = 2 * math.pi * i / n_circ
            rx, ry = -uy * cloth_r * math.cos(a), ux * cloth_r * math.cos(a)
            pts.append([cx + rx, cy + ry, oz + cloth_r * math.sin(a)])
    # UVs, face-varying. Vertex-interpolated UVs cannot work on a closed tube:
    # the seam column is shared between u=~1 and u=0, so one strip of faces
    # would run the texture backwards across the whole duct. Per-corner UVs let
    # that column carry both values. u wraps the circumference, v runs the
    # length, both in metres / WEAVE_TILE_M so the thread size is physical and
    # does not change when the duct gets longer.
    circumference = 2.0 * math.pi * cloth_r
    uvs = []
    for j in range(n_stations - 1):
        v0 = (span * j / (n_stations - 1)) / weave_tile
        v1 = (span * (j + 1) / (n_stations - 1)) / weave_tile
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a, b = j * n_circ + i, j * n_circ + i2
            c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
            tris += [[a, b, c], [a, c, d]]
            u0 = (i * circumference / n_circ) / weave_tile
            u1 = ((i + 1) * circumference / n_circ) / weave_tile
            uvs += [(u0, v0), (u1, v0), (u1, v1)]
            uvs += [(u0, v0), (u1, v1), (u0, v1)]

    if rib_visual:
        # at the FABRIC's radius, not the hoop's, so the rib sits on the skin
        rt = rib_tube if rib_tube > 0 else TUBE * 1.6
        # Sit the rib slightly INSIDE the fabric. Drawn exactly on the fabric
        # radius it is coplanar with the skin, so at a tight fold -- where the
        # fabric bunches and moves -- the rib pops out through it and reads as
        # "the ring went through the cloth". Tucked a millimetre or two under,
        # it stays covered unless the fabric genuinely pulls away.
        rib_r = cloth_r - rib_inset if clearance < 0 else cloth_r + rib_inset
        for i, rpth in enumerate(ring_paths):
            create_ring_visual(stage, f"{rpth}/rib", rib_r, rt,
                               colour=ring_colour)

    sroot = f"{root}/skin"
    skin = f"{sroot}/mesh"
    UsdGeom.Xform.Define(stage, sroot)
    m = UsdGeom.Mesh.Define(stage, skin)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr().Set([Gf.Vec3f(*cloth_colour)])
    UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
    bind_look(stage, m, cloth_colour, roughness=_geom.CLOTH_ROUGHNESS,
              ior=_geom.CLOTH_IOR, weave=True)
    if smooth_render:
        # Render-only smoothing. The solver still sees the coarse triangles;
        # this just stops a 28-sided tube reading as a faceted prism, and takes
        # the hard edges off the chunky folds. Costs nothing in the solver.
        m.CreateSubdivisionSchemeAttr().Set("catmullClark")

    if not deformableUtils.create_auto_surface_deformable_hierarchy(
            stage, root_prim_path=sroot, simulation_mesh_path=f"{sroot}/simMesh",
            cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True):
        if verbose:
            print(f"[duct {index:02d}] deformable hierarchy FAILED", flush=True)
        return root, ring_paths, 0, 0

    rp = stage.GetPrimAtPath(sroot)
    rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    # selfCollision is what stops the fabric passing through ITSELF. With it
    # off a fold is permanent: once the tube creases inward nothing pushes it
    # back out, which is how a bend turns into the crumpled fan. It costs
    # solver time, so it is a deliberate switch rather than always-on.
    attrs = [("physxDeformableBody:selfCollision", bool(self_collision)),
             ("physxDeformableBody:solverPositionIterationCount", 16),
             ("physxDeformableBody:collisionPairUpdateFrequency", 4),
             ("physxDeformableBody:collisionIterationMultiplier", 4),
             # cuts the tunnelling that makes a fold appear to pass straight
             # through a hoop: contact offset adapts to how fast the fabric is
             # moving instead of being fixed
             ("physxDeformableBody:enableSpeculativeCCD", bool(speculative_ccd))]
    if self_collision:
        # selfCollisionFilterDistance DISABLES self-collision for pairs closer
        # than this, so it must sit ABOVE the fabric's own rest spacing. I had
        # it at 0.3x the spacing, which is backwards: every immediate neighbour
        # (25-46 mm apart) then counted as a self-contact and the solver pushed
        # them apart, inflating the sheet -- the fabric went taut and sprang
        # back hard. Just over the largest rest spacing filters exactly the
        # neighbours and nothing else.
        axial_step = spacing / div
        rest_spacing = max(circ_step, axial_step)
        d = (self_collision_distance
             if self_collision_distance > 0 else rest_spacing * 1.1)
        attrs.append(("physxDeformableBody:selfCollisionFilterDistance", float(d)))
        if verbose:
            print(f"[duct {index:02d}] selfCollision ON, filter {d*1e3:.1f} mm "
                  f"(rest spacing {rest_spacing*1e3:.1f} mm). Folds closer than "
                  f"the filter are NOT prevented -- that is the cost of not "
                  f"having every neighbour self-collide.", flush=True)
    for nm, val in attrs:
        a = rp.GetAttribute(nm)
        if a and a.IsValid():
            a.Set(val)
    physicsUtils.add_physics_material_to_prim(stage, rp, material_path)

    # The capture band has to reach from the fabric across the tube: clearance
    # plus the full tube diameter, plus a margin. In taut mode the fabric is
    # deliberately far inside the hoop, so the band has to span that gap for the
    # hoop's surface to find it at all.
    overlap = abs(clearance) + 2 * TUBE + 0.003
    # FILTERING IS NOT OPTIONAL, and it must not be wide either.
    #   off      -> the vertices the seam welds to the hoop also receive contact
    #               forces from that same hoop. The weld and the contact fight,
    #               and the duct collapses into a crumpled fan within a second.
    #               (measured: turning it off did exactly this)
    #   too wide -> free fabric well away from the seam also stops colliding
    #               with the hoop, so on a bend the hoop travels straight out
    #               through the skin.
    # Matching the filter to the capture band filters precisely the welded
    # vertices and nothing else.
    filtering = overlap if filtering_offset is None else filtering_offset
    n_bound = 0
    for i, ring in enumerate(ring_paths):
        seam = f"{sroot}/seam_{i:03d}"
        create_seam(stage, seam, sroot, ring,
                    overlap_offset=overlap, filtering_offset=filtering)
        sp = stage.GetPrimAtPath(seam)
        n_bound += len(sp.GetChildren()) if sp and sp.IsValid() else 0

    # Stash what the seams were built with. The filtering setting is baked in
    # by setup_auto_deformable_attachment, so changing it later means deleting
    # and recreating the attachments -- which needs these numbers back.
    droot = stage.GetPrimAtPath(root)
    droot.CreateAttribute("duct:overlapOffset", Sdf.ValueTypeNames.Float).Set(float(overlap))
    droot.CreateAttribute("duct:clothPath", Sdf.ValueTypeNames.String).Set(sroot)

    if verbose:
        print(f"[duct {index:02d}] SINGLE sleeve: {n_rings} hoops @ {spacing} m "
              f"= {span:.2f} m, {len(pts)} verts, {len(tris)} tris, "
              f"{n_bound} seam elements "
              f"({'OK' if n_bound else 'ZERO -- FABRIC NOT ATTACHED'})", flush=True)
        side = "outside" if clearance >= 0 else "inside"
        print(f"[duct {index:02d}] fabric r {cloth_r*1e3:.1f} mm sits "
              f"{abs(clearance)*1e3:.1f} mm {side} the hoop tube "
              f"(hoop outer {(R+TUBE)*1e3:.1f} mm), overlap band "
              f"{overlap*1e3:.1f} mm, filtering {(overlap if filtering_offset is None else filtering_offset)*1e3:.1f} mm, "
              f"mesh {n_circ} x {n_stations}", flush=True)
    return root, ring_paths, 1, n_bound


def spawn_duct(
    stage,
    index: int,
    n_rings: int,
    spacing: float,
    spec,
    material_path: str,
    origin=(0.0, 0.0, 0.5),
    heading_deg: float = 0.0,
    n_circ: int = 28,
    ring_colour=(0.03, 0.03, 0.03),
    cloth_colour=(0.78, 0.60, 0.05),
    anchor_ends: bool = False,
    overlap_offset: float = 0.006,
    filtering_offset: float = 0.012,
    solid_hoop: bool = False,     # one convex disc per hoop, not 16 capsules
    verbose: bool = True,
):
    """Create duct `index`. Returns (root_path, ring_paths, n_sleeves, n_bound)."""
    R, TUBE = spec.radius, spec.ring_thickness
    root = f"/World/duct_{index:02d}"
    UsdGeom.Xform.Define(stage, root)

    span = spacing * (n_rings - 1)
    ox, oy, oz = origin
    th = math.radians(heading_deg)
    ux, uy = math.cos(th), math.sin(th)

    ring_paths = []
    for i in range(n_rings):
        t = -span / 2 + spacing * i
        path = f"{root}/ring_{i:03d}"
        create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments,
                    solid=solid_hoop)
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(ox + ux * t, oy + uy * t, oz))
        # hoop plane perpendicular to the duct axis
        q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
             * Gf.Rotation(Gf.Vec3d(0, 0, 1), heading_deg))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
        for seg in prim.GetChildren():
            UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
            bind_look(stage, seg, ring_colour, roughness=_geom.RING_ROUGHNESS, ior=_geom.RING_IOR)
        if rib_visual:
            # hide the capsule chain and draw a proper torus instead. The
            # capsules stay as the collider; only what you SEE changes.
            for seg in prim.GetChildren():
                UsdGeom.Imageable(seg).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        is_end = anchor_ends and i in (0, n_rings - 1)
        if is_end:
            UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(0.0)
        else:
            _hoop_mass(prim, spec, solid_hoop)
        ring_paths.append(path)

    # axial resolution chosen so the elements come out roughly square; holding
    # it fixed gave 6 mm x 45 mm slivers at 0.1 m spacing and the cloth solver
    # tore them apart
    circ_step = 2 * math.pi * R / n_circ
    L = max(2, round(spacing / circ_step))

    n_ok = n_bound = 0
    for k in range(n_rings - 1):
        t0 = -span / 2 + spacing * k
        t1 = t0 + spacing
        pts, tris = [], []
        for j in range(L + 1):
            t = t0 + (t1 - t0) * j / L
            cx, cy = ox + ux * t, oy + uy * t
            for i in range(n_circ):
                a = 2 * math.pi * i / n_circ
                # circle in the plane perpendicular to (ux, uy)
                rx, ry = -uy * R * math.cos(a), ux * R * math.cos(a)
                pts.append([cx + rx, cy + ry, oz + R * math.sin(a)])
        for j in range(L):
            for i in range(n_circ):
                i2 = (i + 1) % n_circ
                a, b = j * n_circ + i, j * n_circ + i2
                c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
                tris += [[a, b, c], [a, c, d]]

        sroot = f"{root}/sleeve_{k:03d}"
        skin = f"{sroot}/skin"
        UsdGeom.Xform.Define(stage, sroot)
        m = UsdGeom.Mesh.Define(stage, skin)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
        m.CreateDoubleSidedAttr(True)
        m.CreateDisplayColorAttr().Set([Gf.Vec3f(*cloth_colour)])
        bind_look(stage, m, cloth_colour, roughness=_geom.CLOTH_ROUGHNESS, ior=_geom.CLOTH_IOR)

        if not deformableUtils.create_auto_surface_deformable_hierarchy(
                stage, root_prim_path=sroot, simulation_mesh_path=f"{sroot}/simMesh",
                cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
                set_visibility_with_guide_purpose=True):
            continue
        n_ok += 1

        rp = stage.GetPrimAtPath(sroot)
        rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        for nm, val in (("physxDeformableBody:selfCollision", False),
                        ("physxDeformableBody:solverPositionIterationCount", 16),
                        ("physxDeformableBody:collisionPairUpdateFrequency", 4),
                        ("physxDeformableBody:collisionIterationMultiplier", 4)):
            a = rp.GetAttribute(nm)
            if a and a.IsValid():
                a.Set(val)
        physicsUtils.add_physics_material_to_prim(stage, rp, material_path)

        for side, ri in ((0, k), (1, k + 1)):
            seam = f"{sroot}/seam_{side}"
            create_seam(stage, seam, sroot, ring_paths[ri],
                        overlap_offset=overlap_offset,
                        filtering_offset=filtering_offset)
            # deliberately not named `sp`: in the GUI scripts that name holds
            # the physics scene prim, and shadowing it there crashed the app
            seam_prim = stage.GetPrimAtPath(seam)
            n_bound += (len(seam_prim.GetChildren())
                        if seam_prim and seam_prim.IsValid() else 0)

    if verbose:
        sag = R * (1 - math.cos(math.pi / spec.ring_segments))
        print(f"[duct {index:02d}] {n_rings} hoops @ {spacing} m = {span:.2f} m, "
              f"{n_ok} sleeves, {n_bound} seam elements "
              f"({'OK' if n_bound else 'ZERO -- FABRIC NOT ATTACHED'}), "
              f"mesh {n_circ}x{L}, sagitta {sag*1e3:.2f} mm", flush=True)
    return root, ring_paths, n_ok, n_bound


def rebuild_seams(stage, filtering_offset, verbose=True):
    """Delete and recreate every seam with a new collision-filtering offset.

    Filtering cannot be changed live: setup_auto_deformable_attachment bakes it
    into the generated attachment data. So the toggle deletes the attachment
    prims and authors them again. The caller must stop the simulation around
    this (and drop any RigidPrim views first, or the restart leaves them
    pointing at freed GPU handles).

    filtering_offset < 0 means "match the capture band", which filters exactly
    the welded vertices and nothing else.
    """
    # Collect PATHS, not prim handles, and finish traversing before deleting
    # anything. list(stage.Traverse()) snapshots every prim on the stage --
    # including the seam Scopes this function is about to remove -- and the
    # loop then walks onto a deleted handle:
    #   RuntimeError: Accessed invalid expired 'Scope' prim </…/seam_000>
    duct_paths = [str(p.GetPath()) for p in stage.Traverse()
                  if p.GetName().startswith("duct_")]

    n_seams = n_bound = 0
    for duct_path in duct_paths:
        prim = stage.GetPrimAtPath(duct_path)
        if not prim or not prim.IsValid():
            continue
        ov_attr = prim.GetAttribute("duct:overlapOffset")
        cp_attr = prim.GetAttribute("duct:clothPath")
        if not (ov_attr and ov_attr.IsValid() and cp_attr and cp_attr.IsValid()):
            continue
        overlap = float(ov_attr.Get() or 0.011)
        cloth = cp_attr.Get()
        filt = overlap if filtering_offset < 0 else filtering_offset

        cloth_prim = stage.GetPrimAtPath(cloth)
        if not cloth_prim or not cloth_prim.IsValid():
            continue
        rings = sorted(str(c.GetPath()) for c in prim.GetChildren()
                       if c.GetName().startswith("ring_"))
        seam_paths = [str(c.GetPath()) for c in cloth_prim.GetChildren()
                      if c.GetName().startswith("seam")]
        # DO NOT REUSE THE PATHS. Removing /…/seam_000 and immediately
        # redefining a Scope at the same path yields an expired prim:
        #   Accessed invalid expired 'Scope' prim </World/duct_00/skin/seam_000>
        # Authoring the new seams under a fresh generation prefix sidesteps the
        # whole recomposition question.
        gen = 0
        for sp_path in seam_paths:
            n = sp_path.rsplit("/", 1)[-1]
            parts = n.split("_")
            if len(parts) >= 3 and parts[0] == "seamg":
                try:
                    gen = max(gen, int(parts[1]) + 1)
                except ValueError:
                    pass
            else:
                gen = max(gen, 1)
        for sp_path in seam_paths:
            stage.RemovePrim(Sdf.Path(sp_path))
        for i, ring in enumerate(rings):
            seam = f"{cloth}/seamg_{gen}_{i:03d}"
            create_seam(stage, seam, cloth, ring,
                        overlap_offset=overlap, filtering_offset=filt)
            sp = stage.GetPrimAtPath(seam)
            n_bound += len(sp.GetChildren()) if sp and sp.IsValid() else 0
            n_seams += 1

    if verbose:
        mode = ("ring<->cloth collision ON everywhere (filtering off)"
                if filtering_offset == 0.0 else
                f"filtering {filtering_offset*1e3:.1f} mm"
                if filtering_offset > 0 else
                "filtering matched to the capture band")
        print(f"[duct] rebuilt {n_seams} seams, {n_bound} elements -- {mode}",
              flush=True)
    return n_seams, n_bound


def spawn_duct_path(
    stage,
    index: int,
    stations,
    spec,
    material_path: str,
    z: float = 0.0,
    n_circ: int = 28,
    clearance: float = 0.004,
    filtering_offset=None,
    self_collision: bool = False,
    self_collision_distance: float = 0.0,
    rib_visual: bool = False,
    rib_tube: float = 0.0,
    rib_inset: float = 0.0015,
    smooth_render: bool = True,
    speculative_ccd: bool = True,
    taut: bool = False,             # measured to have no effect; see create_seam
    surface_sampling: float = 0.02,
    ring_colour=(0.03, 0.03, 0.03),
    cloth_colour=(0.78, 0.60, 0.05),
    weave_tile: float = 0.04,       # metres of duct per weave texture tile
    solid_hoop: bool = False,     # one convex disc per hoop, not 16 capsules
    verbose: bool = True,
):
    """A duct that FOLLOWS a drawn centreline instead of running straight.

    `stations` is [(x, y, heading_deg), ...] already resampled at the hoop
    spacing by duct_sim.layout.resample. Each hoop is placed at its station and
    turned to face along the path, and the fabric tube is swept through the
    same stations so it follows the curve rather than cutting the corner.

    The straight builder is the special case of this with a constant heading;
    everything else -- seams, offsets, ribs -- behaves identically.
    """
    R, TUBE = spec.radius, spec.ring_thickness
    root = f"/World/duct_{index:02d}"
    UsdGeom.Xform.Define(stage, root)

    if z <= 0.0:
        # A LITTLE ABOVE THE FLOOR, not exactly on it. Starting a deformable in
        # contact means the first step has to resolve ground contact on every
        # vertex at once, and the straight builder never hit this because the
        # GUI spawns it at --height and lets it fall. Dropping a few centimetres
        # gives the solver a settled state to work from.
        z = spec.drop_to_floor + 0.05

    ring_paths = []
    for i, (x, y, h) in enumerate(stations):
        path = f"{root}/ring_{i:04d}"
        create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments,
                    solid=solid_hoop)
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
             * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(h)))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
        for seg in prim.GetChildren():
            UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
            bind_look(stage, seg, ring_colour, roughness=_geom.RING_ROUGHNESS, ior=_geom.RING_IOR)
            if rib_visual:
                UsdGeom.Imageable(seg).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        _hoop_mass(prim, spec, solid_hoop)
        ring_paths.append(path)

    cloth_r = (R + TUBE + clearance) if clearance >= 0 else (R - TUBE + clearance)
    circ_step = 2 * math.pi * cloth_r / n_circ

    if rib_visual:
        rt = rib_tube if rib_tube > 0 else TUBE * 1.6
        rib_r = cloth_r - rib_inset if clearance < 0 else cloth_r + rib_inset
        for rpth in ring_paths:
            create_ring_visual(stage, f"{rpth}/rib", rib_r, rt, colour=ring_colour)

    # fabric loops: the hoop stations plus interpolated ones between them, so
    # the tube has free vertices to drape with
    spacing = 0.0
    if len(stations) > 1:
        spacing = math.hypot(stations[1][0] - stations[0][0],
                             stations[1][1] - stations[0][1])
    div = max(2, round(spacing / circ_step)) if spacing > 0 else 2

    loops = []
    for i in range(len(stations) - 1):
        x0, y0, h0 = stations[i]
        x1, y1, h1 = stations[i + 1]
        dh = ((h1 - h0 + 180.0) % 360.0) - 180.0      # shortest turn
        for j in range(div):
            t = j / div
            loops.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, h0 + dh * t))
    loops.append(stations[-1])

    pts, tris = [], []
    for (cx, cy, h) in loops:
        th = math.radians(h)
        ux, uy = math.cos(th), math.sin(th)
        for i in range(n_circ):
            a = 2 * math.pi * i / n_circ
            rx, ry = -uy * cloth_r * math.cos(a), ux * cloth_r * math.cos(a)
            pts.append([cx + rx, cy + ry, z + cloth_r * math.sin(a)])
    # Face-varying UVs. Vertex-interpolated UVs cannot work on a closed tube:
    # the seam column is shared between u=~1 and u=0, so one strip of faces
    # would run the texture backwards along the whole duct. v accumulates the
    # real arc length so the weave does not stretch or compress through a turn.
    circumference = 2.0 * math.pi * cloth_r
    arc = [0.0]
    for j in range(len(loops) - 1):
        arc.append(arc[-1] + math.hypot(loops[j + 1][0] - loops[j][0],
                                        loops[j + 1][1] - loops[j][1]))
    uvs = []
    for j in range(len(loops) - 1):
        v0, v1 = arc[j] / weave_tile, arc[j + 1] / weave_tile
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a, b = j * n_circ + i, j * n_circ + i2
            c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
            tris += [[a, b, c], [a, c, d]]
            u0 = (i * circumference / n_circ) / weave_tile
            u1 = ((i + 1) * circumference / n_circ) / weave_tile
            uvs += [(u0, v0), (u1, v0), (u1, v1)]
            uvs += [(u0, v0), (u1, v1), (u0, v1)]

    sroot = f"{root}/skin"
    skin = f"{sroot}/mesh"
    UsdGeom.Xform.Define(stage, sroot)
    m = UsdGeom.Mesh.Define(stage, skin)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr().Set([Gf.Vec3f(*cloth_colour)])
    UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
    bind_look(stage, m, cloth_colour, roughness=_geom.CLOTH_ROUGHNESS,
              ior=_geom.CLOTH_IOR, weave=True)
    if smooth_render:
        m.CreateSubdivisionSchemeAttr().Set("catmullClark")

    if not deformableUtils.create_auto_surface_deformable_hierarchy(
            stage, root_prim_path=sroot, simulation_mesh_path=f"{sroot}/simMesh",
            cooking_src_mesh_path=skin, cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True):
        if verbose:
            print(f"[duct {index:02d}] deformable hierarchy FAILED", flush=True)
        return root, ring_paths, 0, 0

    rp = stage.GetPrimAtPath(sroot)
    rp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    attrs = [("physxDeformableBody:selfCollision", bool(self_collision)),
             ("physxDeformableBody:solverPositionIterationCount", 16),
             ("physxDeformableBody:collisionPairUpdateFrequency", 4),
             ("physxDeformableBody:collisionIterationMultiplier", 4),
             ("physxDeformableBody:enableSpeculativeCCD", bool(speculative_ccd))]
    if self_collision:
        axial = spacing / div if div else circ_step
        d = (self_collision_distance if self_collision_distance > 0
             else max(circ_step, axial) * 1.1)
        attrs.append(("physxDeformableBody:selfCollisionFilterDistance", float(d)))
    for nm, val in attrs:
        a = rp.GetAttribute(nm)
        if a and a.IsValid():
            a.Set(val)
    physicsUtils.add_physics_material_to_prim(stage, rp, material_path)

    overlap = abs(clearance) + 2 * TUBE + 0.003
    filtering = overlap if filtering_offset is None else filtering_offset
    droot = stage.GetPrimAtPath(root)
    droot.CreateAttribute("duct:overlapOffset", Sdf.ValueTypeNames.Float).Set(float(overlap))
    droot.CreateAttribute("duct:clothPath", Sdf.ValueTypeNames.String).Set(sroot)

    n_bound = 0
    for i, ring in enumerate(ring_paths):
        seam = f"{sroot}/seam_{i:04d}"
        create_seam(stage, seam, sroot, ring,
                    overlap_offset=overlap, filtering_offset=filtering,
                    rigid_surface=taut, surface_sampling=surface_sampling)
        sp = stage.GetPrimAtPath(seam)
        n_bound += len(sp.GetChildren()) if sp and sp.IsValid() else 0

    if verbose:
        print(f"[duct {index:02d}] PATH: {len(stations)} hoops, {len(pts)} verts, "
              f"{len(tris)} tris, {n_bound} seam elements "
              f"({'OK' if n_bound else 'ZERO -- FABRIC NOT ATTACHED'})", flush=True)
    return root, ring_paths, 1, n_bound


def _set_mass(prim, mass, density):
    api = UsdPhysics.MassAPI.Apply(prim)
    if mass > 0:
        api.CreateMassAttr(float(mass))
    else:
        api.CreateDensityAttr(float(density))


def _stabilise(prim, damping, pos_iters, vel_iters, sleep_threshold):
    """Settle a long hinged chain instead of letting it buzz.

    Hundreds of bodies joined by locked-translation D6s is a stiff system: with
    the default iteration count the solver cannot satisfy every constraint in a
    step, and the residual shows up as a permanent shiver that never dies down.
    More iterations, a little body damping, and a sleep threshold so a settled
    duct actually stops being integrated.
    """
    prim.ApplyAPI("PhysxRigidBodyAPI")
    for name, val in (("physxRigidBody:solverPositionIterationCount", int(pos_iters)),
                      ("physxRigidBody:solverVelocityIterationCount", int(vel_iters)),
                      ("physxRigidBody:linearDamping", float(damping)),
                      ("physxRigidBody:angularDamping", float(damping)),
                      ("physxRigidBody:sleepThreshold", float(sleep_threshold)),
                      ("physxRigidBody:stabilizationThreshold", float(sleep_threshold) * 0.2)):
        a = prim.GetAttribute(name)
        if a and a.IsValid():
            a.Set(val)


def _d6_compliant(stage, path, body0, body1, anchor0, anchor1,
                  bend_limit_deg, stiffness, damping, twist_scale=0.5):
    """D6 that locks translation and lets the joint bend against a soft drive.

    Translation is a hard lock, not a stiff spring: the hoops of a real duct are
    sewn into inextensible fabric, and modelling that as a spring is what makes
    the chain bounce like a slinky. Bending is a limited rotation with a drive
    pulling back to straight -- that drive IS the compliance.
    """
    j = UsdPhysics.Joint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr().Set(anchor0)
    j.CreateLocalPos1Attr().Set(anchor1)
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    for axis in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
        lim.CreateLowAttr(0.0)
        lim.CreateHighAttr(0.0)
    for axis, deg in (("rotX", bend_limit_deg * twist_scale),
                      ("rotY", bend_limit_deg),
                      ("rotZ", bend_limit_deg)):
        lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
        lim.CreateLowAttr(float(-deg))
        lim.CreateHighAttr(float(deg))
    for axis in ("rotX", "rotY", "rotZ"):
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
        drv.CreateTypeAttr("force")
        drv.CreateTargetPositionAttr(0.0)
        drv.CreateStiffnessAttr(float(stiffness))
        drv.CreateDampingAttr(float(damping))
    return j


def spawn_duct_rigid(
    stage,
    index: int,
    stations,
    spec,
    z: float = 0.0,
    disc_thickness: float = 0.006,
    sleeve_inset: float = 0.006,
    # Measured, not guessed. Two things are wanted and they pull opposite ways:
    # the duct must not heave around on its own, and a hand must be able to
    # fold its end to the floor. STIFFNESS RESISTS BOTH -- gravity and the hand
    # equally -- so buying quiet with stiffness buys stubbornness with it: at
    # stiffness 200 the residual was a lovely 0.0034 m/s and 3 N moved the tip
    # exactly 0.0000 m. DAMPING resists only speed, so it settles the chain
    # without fighting a slow deliberate bend. Hence low stiffness, high
    # damping: 0.0088 m/s residual (still 5x quieter than the first guess) and
    # the tip folds 70 mm.
    bend_limit_deg: float = 14.0,
    stiffness: float = 30.0,
    damping: float = 30.0,
    density: float = 0.0,
    mass_per_m: float = 0.5,
    body_damping: float = 2.0,
    solver_pos_iters: int = 32,
    solver_vel_iters: int = 4,
    sleep_threshold: float = 0.005,
    ring_colour=(0.03, 0.03, 0.03),
    cloth_colour=(0.78, 0.60, 0.05),
    verbose: bool = True,
):
    """A duct made ONLY of rigid bodies: black discs and yellow sleeves, hinged.

    No cloth at all. Each hoop becomes a thin black disc and each gap a yellow
    cylinder, joined by D6s whose rotational drives give the compliance -- push
    it and it bends, let go and it eases back.

    The point is cost. The cloth was never the expensive part; the hoop
    colliders were. A hoop built from 32 capsules is 32 convex shapes, and a
    1,629-hoop track is ~52,000 of them, with the CPU at 386 % and the GPU
    idling at 2-9 %. Here a hoop is ONE cylinder and a gap is one more, so the
    same track is ~3,300 shapes: about 16x fewer.

    What it gives up is real fabric. The cross-section stays circular, so it
    cannot crumple, fold or drape -- a car hitting this finds a compliant tube,
    not a bag. For a track barrier that may be the better model anyway, and it
    is the difference between 7.6x real time and something usable.
    """
    R = spec.radius
    root = f"/World/duct_{index:02d}"
    UsdGeom.Xform.Define(stage, root)
    if z <= 0.0:
        z = R + disc_thickness + 0.05

    # SET MASS, DO NOT SET DENSITY. A duct is a thin-walled tube; the sleeve
    # body is a SOLID cylinder, so a density that looks reasonable produces an
    # absurd mass -- 195 kg/m3 gave 139 kg for a 6 m duct, about 47x a real
    # 400 mm duct's ~0.5 kg/m. Hundreds of overweight links on compliant joints
    # is exactly what makes the chain heave around like a worm.
    total_len = 0.0
    for i in range(len(stations) - 1):
        total_len += math.hypot(stations[i + 1][0] - stations[i][0],
                                stations[i + 1][1] - stations[i][1])
    n_bodies_est = max(1, 2 * len(stations) - 1)
    body_mass = (mass_per_m * total_len / n_bodies_est) if mass_per_m > 0 else 0.0

    def _place(prim, x, y, heading_deg):
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
             * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(heading_deg)))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))

    bodies = []          # (path, kind)
    for i, (x, y, h) in enumerate(stations):
        d = UsdGeom.Cylinder.Define(stage, f"{root}/disc_{i:04d}")
        d.CreateAxisAttr("Z")
        d.CreateRadiusAttr(float(R))
        d.CreateHeightAttr(float(disc_thickness))
        d.CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
        bind_look(stage, d, ring_colour, roughness=_geom.RING_ROUGHNESS, ior=_geom.RING_IOR)
        _place(d.GetPrim(), x, y, h)
        UsdPhysics.CollisionAPI.Apply(d.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(d.GetPrim())
        _set_mass(d.GetPrim(), body_mass, density)
        _stabilise(d.GetPrim(), body_damping, solver_pos_iters,
                   solver_vel_iters, sleep_threshold)
        bodies.append((f"{root}/disc_{i:04d}", "disc"))

        if i < len(stations) - 1:
            x1, y1, h1 = stations[i + 1]
            gap = math.hypot(x1 - x, y1 - y)
            length = max(1e-3, gap - disc_thickness)
            dh = ((h1 - h + 180.0) % 360.0) - 180.0
            s = UsdGeom.Cylinder.Define(stage, f"{root}/sleeve_{i:04d}")
            s.CreateAxisAttr("Z")
            s.CreateRadiusAttr(float(R - sleeve_inset))
            s.CreateHeightAttr(float(length))
            s.CreateDisplayColorAttr().Set([Gf.Vec3f(*cloth_colour)])
            bind_look(stage, s, cloth_colour, roughness=_geom.CLOTH_ROUGHNESS, ior=_geom.CLOTH_IOR)
            _place(s.GetPrim(), (x + x1) * 0.5, (y + y1) * 0.5, h + dh * 0.5)
            UsdPhysics.CollisionAPI.Apply(s.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(s.GetPrim())
            _set_mass(s.GetPrim(), body_mass, density)
            _stabilise(s.GetPrim(), body_damping, solver_pos_iters,
                       solver_vel_iters, sleep_threshold)
            bodies.append((f"{root}/sleeve_{i:04d}", "sleeve"))

    # hinge consecutive bodies at the face they share. Local +Z runs along the
    # duct for every body, because _place maps local Z onto the heading.
    n_joints = 0
    for k in range(len(bodies) - 1):
        p0, kind0 = bodies[k]
        p1, _ = bodies[k + 1]
        h0 = (disc_thickness if kind0 == "disc"
              else float(stage.GetPrimAtPath(p0).GetAttribute("height").Get()))
        h1 = float(stage.GetPrimAtPath(p1).GetAttribute("height").Get())
        _d6_compliant(stage, f"{p0}/joint", p0, p1,
                      Gf.Vec3f(0, 0, float(h0) * 0.5),
                      Gf.Vec3f(0, 0, float(-h1) * 0.5),
                      bend_limit_deg, stiffness, damping)
        n_joints += 1

    if verbose:
        print(f"[duct {index:02d}] mass {mass_per_m} kg/m over {total_len:.2f} m "
              f"= {mass_per_m * total_len:.2f} kg total, "
              f"{body_mass * 1000:.1f} g per body", flush=True)
        print(f"[duct {index:02d}] RIGID: {len(stations)} discs + "
              f"{len(bodies) - len(stations)} sleeves = {len(bodies)} bodies, "
              f"{len(bodies)} collision shapes, {n_joints} joints "
              f"(vs ~{len(stations) * spec.ring_segments} shapes for the cloth "
              f"build)", flush=True)
    return root, [b for b, _ in bodies], len(bodies), n_joints


def _tube_mesh(stations, radius, n_circ, z, rib_amp=0.0, rib_period=0.10,
               arc_step=0.05):
    """Points and triangles for a tube swept along `stations`.

    rib_amp modulates the radius along the path, so the corrugations of real
    ducting come out of the geometry instead of needing separate hoop bodies:
        r(s) = radius + rib_amp * sin(2*pi*s / rib_period)
    """
    pts, tris = [], []
    s = 0.0
    for i, (cx, cy, h) in enumerate(stations):
        if i:
            s += math.hypot(cx - stations[i - 1][0], cy - stations[i - 1][1])
        r = radius + (rib_amp * math.sin(2 * math.pi * s / rib_period)
                      if rib_amp else 0.0)
        th = math.radians(h)
        ux, uy = math.cos(th), math.sin(th)
        for k in range(n_circ):
            a = 2 * math.pi * k / n_circ
            rx, ry = -uy * r * math.cos(a), ux * r * math.cos(a)
            pts.append([cx + rx, cy + ry, z + r * math.sin(a)])
    for j in range(len(stations) - 1):
        for k in range(n_circ):
            k2 = (k + 1) % n_circ
            a, b = j * n_circ + k, j * n_circ + k2
            c, d = (j + 1) * n_circ + k2, (j + 1) * n_circ + k
            tris += [[a, b, c], [a, c, d]]
    return pts, tris


def spawn_duct_static(
    stage,
    index: int,
    stations,
    spec,
    z: float = 0.0,
    n_circ: int = 20,
    rib_amp: float = 0.012,
    rib_period: float = 0.10,
    capsule_collider: bool = True,
    capsule_every: float = 0.25,
    colour=(0.78, 0.60, 0.05),
    rib_colour=(0.03, 0.03, 0.03),
    verbose: bool = True,
):
    """A duct that does not move: one swept tube mesh, static collider.

    No bodies, no joints, no solver work at all -- the duct is scenery. The
    corrugations come from modulating the sweep radius, so it still LOOKS like
    ducting to a camera or an RTX lidar.

    Two collider choices, and the default is deliberate. A triangle mesh is
    exact but every contact against it is a mesh query; a row of capsules along
    the centreline is approximate but contacts are analytic, which is both
    faster and far better behaved for a car hitting it at speed. For a track
    barrier the capsule row is the better trade.
    """
    R = spec.radius
    root = f"/World/duct_{index:02d}"
    UsdGeom.Xform.Define(stage, root)
    if z <= 0.0:
        z = R + 0.002

    pts, tris = _tube_mesh(stations, R, n_circ, z, rib_amp, rib_period)
    mesh = UsdGeom.Mesh.Define(stage, f"{root}/tube")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*colour)])
    bind_look(stage, mesh, colour, roughness=_geom.CLOTH_ROUGHNESS, ior=_geom.CLOTH_IOR)

    n_col = 0
    if capsule_collider:
        # analytic capsules along the centreline, invisible, no rigid body ->
        # static colliders
        total = 0.0
        for i in range(len(stations) - 1):
            total += math.hypot(stations[i + 1][0] - stations[i][0],
                                stations[i + 1][1] - stations[i][1])
        step = max(1, int(round(capsule_every /
                                (total / max(1, len(stations) - 1)))))
        for i in range(0, len(stations) - 1, step):
            j = min(i + step, len(stations) - 1)
            x0, y0, _ = stations[i]
            x1, y1, _ = stations[j]
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-4:
                continue
            cap = UsdGeom.Capsule.Define(stage, f"{root}/col_{i:04d}")
            cap.CreateAxisAttr("Z")
            cap.CreateRadiusAttr(float(R))
            cap.CreateHeightAttr(float(seg))
            xf = UsdGeom.Xformable(cap)
            xf.AddTranslateOp().Set(
                Gf.Vec3d((x0 + x1) * 0.5, (y0 + y1) * 0.5, float(z)))
            head = math.degrees(math.atan2(y1 - y0, x1 - x0))
            q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
                 * Gf.Rotation(Gf.Vec3d(0, 0, 1), head))
            xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
            UsdGeom.Imageable(cap).CreateVisibilityAttr().Set("invisible")
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
            n_col += 1
    else:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mca = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mca.CreateApproximationAttr("none")
        n_col = 1

    if verbose:
        print(f"[duct {index:02d}] STATIC: 1 tube mesh "
              f"({len(pts)} verts, {len(tris)} tris), {n_col} static colliders, "
              f"0 bodies, 0 joints", flush=True)
    return root, [], 1, n_col


class HybridSkin:
    """A high-res ribbed tube whose points are driven by a rigid chain.

    Physics stays the cheap disc-and-sleeve chain; the visible surface is a
    separate mesh that is re-swept each step from the chain's poses. Cameras
    and RTX lidar see corrugated ducting, the solver sees capsules.

    Which one a sensor sees is not a detail to hand-wave: an RTX lidar traces
    the RENDER scene (omni.sensors.nv.lidar depends on omni.hydra.rtx), while a
    physics-query lidar traces colliders. Those are different surfaces here --
    by the rib amplitude -- so the choice of sensor changes the data.
    """

    def __init__(self, stage, mesh_path, body_paths, n_circ, radius,
                 rib_amp=0.012, rib_period=0.10, every=2):
        self.stage = stage
        self.mesh = UsdGeom.Mesh.Get(stage, mesh_path)
        self.body_paths = list(body_paths)
        self.n_circ = n_circ
        self.radius = radius
        self.rib_amp = rib_amp
        self.rib_period = rib_period
        self.every = max(1, int(every))
        self._view = None
        self._n = 0

    def _ensure(self):
        if self._view is None:
            from isaacsim.core.prims import RigidPrim
            self._view = RigidPrim(self.body_paths)
        return self._view

    def update(self):
        """Re-sweep the visual tube from where the bodies actually are."""
        self._n += 1
        if self._n % self.every:
            return
        try:
            pos, _ = self._ensure().get_world_poses()
            pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos)
        except Exception:
            return
        if len(pos) < 2:
            return
        stations = []
        s = 0.0
        for i in range(len(pos)):
            if i:
                s += float(np.hypot(pos[i, 0] - pos[i - 1, 0],
                                    pos[i, 1] - pos[i - 1, 1]))
            j = min(i + 1, len(pos) - 1)
            k = max(i - 1, 0)
            head = math.degrees(math.atan2(pos[j, 1] - pos[k, 1],
                                           pos[j, 0] - pos[k, 0]))
            stations.append((float(pos[i, 0]), float(pos[i, 1]), head))
        pts = []
        s = 0.0
        for i, (cx, cy, h) in enumerate(stations):
            if i:
                s += math.hypot(cx - stations[i - 1][0], cy - stations[i - 1][1])
            r = self.radius + self.rib_amp * math.sin(2 * math.pi * s / self.rib_period)
            th = math.radians(h)
            ux, uy = math.cos(th), math.sin(th)
            zc = float(pos[i, 2])
            for k in range(self.n_circ):
                a = 2 * math.pi * k / self.n_circ
                pts.append([cx - uy * r * math.cos(a),
                            cy + ux * r * math.cos(a),
                            zc + r * math.sin(a)])
        self.mesh.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))


def spawn_duct_hybrid(stage, index, stations, spec, n_circ=20,
                      rib_amp=0.012, rib_period=0.10, skin_every=2,
                      colour=(0.78, 0.60, 0.05), verbose=True, **rigid_kw):
    """Rigid chain for physics, separate ribbed tube for the eye."""
    root, paths, n_bodies, n_joints = spawn_duct_rigid(
        stage, index, stations, spec, verbose=False, **rigid_kw)
    # the chain's own boxes become invisible; the swept tube is what is seen
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")

    z = spec.radius + 0.006 + 0.05
    pts, tris = _tube_mesh(stations, spec.radius, n_circ, z, rib_amp, rib_period)
    mp = f"{root}/skin"
    mesh = UsdGeom.Mesh.Define(stage, mp)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*colour)])
    bind_look(stage, mesh, colour, roughness=_geom.CLOTH_ROUGHNESS, ior=_geom.CLOTH_IOR)

    skin = HybridSkin(stage, mp, paths, n_circ, spec.radius,
                      rib_amp, rib_period, skin_every)
    if verbose:
        print(f"[duct {index:02d}] HYBRID: {n_bodies} bodies / {n_joints} joints "
              f"for physics, {len(pts)} verts re-swept every {skin_every} steps "
              f"for the eye", flush=True)
    return root, paths, skin, n_bodies


def spawn_duct_fem(
    stage,
    index: int,
    stations,
    spec,
    material_path: str,
    z: float = 0.0,
    n_circ: int = 12,
    wall: float = 0.0,
    ring_every: float = 0.0,
    ring_tube: float = 0.010,
    ring_density: float = 300.0,
    colour=(0.78, 0.60, 0.05),
    ring_colour=(0.03, 0.03, 0.03),
    verbose: bool = True,
):
    """A solid low-resolution cylinder as a volume (FEM) deformable.

    Deliberately NOT a hollow thin-walled tube. Tetrahedralising a thin wall
    needs a very dense mesh to get even one element across it, which is both
    expensive and numerically fragile -- the standard advice, and it matches
    what a thin shell already cost here. A solid low-res cylinder is the usual
    compromise: it squashes and springs back when a car hits it, and it cannot
    represent the hollow interior at all.

    Included so the comparison is honest, not because it is expected to win.
    """
    R = spec.radius
    root = f"/World/duct_{index:02d}"
    UsdGeom.Xform.Define(stage, root)
    if z <= 0.0:
        z = R + 0.02

    r_out = R if wall <= 0 else R
    pts, tris = _tube_mesh(stations, r_out, n_circ, z)
    # cap both ends so the sweep is a closed solid: the cooker needs a watertight
    # surface to tetrahedralise
    n = len(stations)
    base = len(pts)
    first_c = [sum(p[k] for p in pts[:n_circ]) / n_circ for k in range(3)]
    last_c = [sum(p[k] for p in pts[(n - 1) * n_circ:n * n_circ]) / n_circ
              for k in range(3)]
    pts.append(first_c)
    pts.append(last_c)
    ci0, ci1 = base, base + 1
    for k in range(n_circ):
        k2 = (k + 1) % n_circ
        tris.append([ci0, k2, k])
        tris.append([ci1, (n - 1) * n_circ + k, (n - 1) * n_circ + k2])

    skin = f"{root}/solid"
    m = UsdGeom.Mesh.Define(stage, skin)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    m.CreateDisplayColorAttr().Set([Gf.Vec3f(*colour)])
    bind_look(stage, m, colour, roughness=_geom.CLOTH_ROUGHNESS, ior=_geom.CLOTH_IOR)

    ok = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage,
        root_prim_path=root,
        simulation_tetmesh_path=f"{root}/simTet",
        collision_tetmesh_path=f"{root}/colTet",
        cooking_src_mesh_path=skin,
        simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=True,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        if verbose:
            print(f"[duct {index:02d}] FEM: hierarchy FAILED", flush=True)
        return root, [], 0, 0

    # NOT PhysxDeformableBodyAPI -- no such schema in 6.1; the applicable one
    # is PhysxBaseDeformableBodyAPI, and create_auto_volume_deformable_hierarchy
    # has already applied whatever the body needs. Only the material is left.
    rp = stage.GetPrimAtPath(root)
    for api in ("PhysxBaseDeformableBodyAPI",):
        try:
            rp.ApplyAPI(api)
        except Exception:
            pass
    physicsUtils.add_physics_material_to_prim(stage, rp, material_path)

    # --- optional hoops bonded to the body, so they FOLLOW it as it deforms ---
    # A ring drawn on the outside and left there would sit still while the duct
    # squashed underneath it. These are real rigid bodies whose collider sits
    # just INSIDE the solid, bonded by the same attachment mechanism the cloth
    # seams use; the torus that is actually seen is drawn on the outside.
    ring_paths = []
    n_bound = 0
    if ring_every > 0:
        total = 0.0
        for i in range(len(stations) - 1):
            total += math.hypot(stations[i + 1][0] - stations[i][0],
                                stations[i + 1][1] - stations[i][1])
        seg = total / max(1, len(stations) - 1)
        step = max(1, int(round(ring_every / seg)))
        for i in range(0, len(stations), step):
            x, y, h = stations[i]
            path = f"{root}/hoop_{i:04d}"
            # ONE thin cylinder, not a 24-capsule ring. The collider's only job
            # is to overlap the body so the bond can form -- it never has to be
            # ring-shaped, because the FEM body is solid and nothing passes
            # through the middle. The capsule ring cost 31 x 24 = 744 shapes and
            # tripled the step time (9.5 -> 24 ms) for no behaviour at all.
            disc = UsdGeom.Cylinder.Define(stage, path)
            disc.CreateAxisAttr("Z")
            disc.CreateRadiusAttr(float(r_out - ring_tube * 0.5))
            disc.CreateHeightAttr(float(ring_tube * 1.2))
            prim = disc.GetPrim()
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
            q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
                 * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(h)))
            xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.02)
            create_ring_visual(stage, f"{path}/rib", r_out + ring_tube * 0.4,
                               ring_tube, n_major=32, n_minor=8,
                               colour=ring_colour)
            seam = f"{root}/bond_{i:04d}"
            # a narrower capture band: the bond needs the vertices at the hoop,
            # not a thick slab of them
            create_seam(stage, seam, root, path,
                        overlap_offset=ring_tube,
                        filtering_offset=ring_tube)
            spm = stage.GetPrimAtPath(seam)
            n_bound += len(spm.GetChildren()) if spm and spm.IsValid() else 0
            ring_paths.append(path)

    if verbose:
        print(f"[duct {index:02d}] FEM: solid sweep {len(pts)} verts -> "
              f"tetrahedral body", flush=True)
        if ring_paths:
            print(f"[duct {index:02d}] FEM: {len(ring_paths)} hoops bonded to "
                  f"the body, {n_bound} bond elements "
                  f"({'OK' if n_bound else 'ZERO -- hoops will not follow'})",
                  flush=True)
    return root, ring_paths, 1, n_bound
