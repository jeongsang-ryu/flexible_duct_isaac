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

from duct_sim.geometry import create_ring, create_ring_visual


def create_seam(stage, seam_path, cloth_path, rigid_path,
                overlap_offset=0.006, filtering_offset=0.012):
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
            ("physxAutoDeformableAttachment:enableDeformableVertexAttachments", True)):
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
    ring_colour=(0.03, 0.03, 0.03),
    cloth_colour=(0.95, 0.80, 0.10),
    anchor_ends: bool = False,
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
        create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments)
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(ox + ux * t, oy + uy * t, oz))
        q = (Gf.Rotation(Gf.Vec3d(0, 0, 1), heading_deg)
             * Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
        for seg in prim.GetChildren():
            UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
        if rib_visual:
            # hide the capsule chain and draw a proper torus instead. The
            # capsules stay as the collider; only what you SEE changes.
            for seg in prim.GetChildren():
                UsdGeom.Imageable(seg).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        is_end = anchor_ends and i in (0, n_rings - 1)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(
            0.0 if is_end else spec.ring_density)
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
    for j in range(n_stations - 1):
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a, b = j * n_circ + i, j * n_circ + i2
            c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
            tris += [[a, b, c], [a, c, d]]

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
    # plus the full tube diameter, plus a margin.
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
    cloth_colour=(0.95, 0.80, 0.10),
    anchor_ends: bool = False,
    overlap_offset: float = 0.006,
    filtering_offset: float = 0.012,
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
        create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments)
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(ox + ux * t, oy + uy * t, oz))
        # hoop plane perpendicular to the duct axis
        q = (Gf.Rotation(Gf.Vec3d(0, 0, 1), heading_deg)
             * Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
        for seg in prim.GetChildren():
            UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
        if rib_visual:
            # hide the capsule chain and draw a proper torus instead. The
            # capsules stay as the collider; only what you SEE changes.
            for seg in prim.GetChildren():
                UsdGeom.Imageable(seg).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        is_end = anchor_ends and i in (0, n_rings - 1)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(
            0.0 if is_end else spec.ring_density)
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
    anchor_stations=(),
    ring_colour=(0.03, 0.03, 0.03),
    cloth_colour=(0.95, 0.80, 0.10),
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

    anchor_set = set(int(a) for a in anchor_stations)

    ring_paths = []
    for i, (x, y, h) in enumerate(stations):
        path = f"{root}/ring_{i:04d}"
        create_ring(stage, path, R, TUBE, n_seg=spec.ring_segments)
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        q = (Gf.Rotation(Gf.Vec3d(0, 0, 1), float(h))
             * Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0))
        xf.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
        for seg in prim.GetChildren():
            UsdGeom.Gprim(seg).CreateDisplayColorAttr().Set([Gf.Vec3f(*ring_colour)])
            if rib_visual:
                UsdGeom.Imageable(seg).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # an anchored hoop is STATIC (density 0), the same trick the seam demos
        # use -- it holds position while the rest of the duct moves around it
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(
            0.0 if i in anchor_set else spec.ring_density)
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
    for j in range(len(loops) - 1):
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a, b = j * n_circ + i, j * n_circ + i2
            c, d = (j + 1) * n_circ + i2, (j + 1) * n_circ + i
            tris += [[a, b, c], [a, c, d]]

    sroot = f"{root}/skin"
    skin = f"{sroot}/mesh"
    UsdGeom.Xform.Define(stage, sroot)
    m = UsdGeom.Mesh.Define(stage, skin)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.array(tris).flatten().tolist()))
    m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr().Set([Gf.Vec3f(*cloth_colour)])
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
                    overlap_offset=overlap, filtering_offset=filtering)
        sp = stage.GetPrimAtPath(seam)
        n_bound += len(sp.GetChildren()) if sp and sp.IsValid() else 0

    if verbose:
        print(f"[duct {index:02d}] PATH: {len(stations)} hoops "
              f"({len(anchor_set)} anchored), {len(pts)} verts, "
              f"{len(tris)} tris, {n_bound} seam elements "
              f"({'OK' if n_bound else 'ZERO -- FABRIC NOT ATTACHED'})", flush=True)
    return root, ring_paths, 1, n_bound
