"""Approach A -- articulated rigid rings, fabric as a visual skin.

The duct is a chain of rigid hoops joined by D6 joints. Bending, twisting and
axial stretch are governed by joint drives with spring/damper gains, which is
what makes the chain settle instead of oscillating. The fabric between rings is
geometry only: it is NOT simulated, it is redrawn each frame from the ring
poses (see skin.py).

WHY THIS IS THE DEFAULT for "drag it with the mouse and it should look
natural". A 21-ring chain is ~21 bodies and ~20 joints -- trivial for PhysX at
60 Hz, and rigid-body drag is the interaction Kit's mouse grab is built for.
Approach B (real cloth) has to solve thousands of particles under an
interactive constraint and is far easier to make blow up.

WHAT IT CANNOT DO: no wrinkles, no fabric slack, no self-collision of the
cloth. If the research question needs those, that is exactly what B is for.
"""

from __future__ import annotations

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from .geometry import create_ring
from .spec import DuctSpec


def _apply_rigid_body(stage: Usd.Stage, path: str, density: float,
                      angular_damping: float = 0.0) -> None:
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr(float(density))
    if angular_damping > 0.0:
        # WHY ANGULAR DAMPING IS NEEDED AT ALL. A hoop standing in the YZ plane
        # is a wheel: push the duct sideways and it ROLLS about the duct axis
        # rather than sliding. Measured on the unmodified build, a 1 N push
        # moved the 2 kg duct 0.74 m while the hoops spun 153 deg -- 72% of the
        # 213 deg that pure rolling would give. Rolling meets almost no
        # resistance, so no amount of surface friction or joint tuning stops it,
        # and the duct slides away straight instead of bending.
        #
        # A real flex duct does not roll because its fabric is dragging along
        # the floor the whole time. Approach A's fabric is visual only, so that
        # resistance has to be reintroduced; angular damping is the direct
        # stand-in for "dragging a skin across the ground".
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateAngularDampingAttr(
            float(angular_damping)
        )


def _d6_between(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    anchor_in_b0: Gf.Vec3f,
    anchor_in_b1: Gf.Vec3f,
    bend_limit_deg: float,
    bend_stiffness: float,
    bend_damping: float,
) -> None:
    """A D6 that locks translation and allows damped, limited bending/twist.

    Translation is locked rather than given a stiff drive: a flex duct's hoops
    are sewn into the fabric, so the spacing between them is essentially
    inextensible, and modelling it as a stiff spring instead of a hard
    constraint is what produces the slinky-like axial bouncing that looks wrong.
    """
    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr().Set(anchor_in_b0)
    joint.CreateLocalPos1Attr().Set(anchor_in_b1)
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # lock all three translations
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(0.0)
        limit.CreateHighAttr(0.0)

    # limited bending about X and Y, limited twist about Z
    for axis, lim in (("rotX", bend_limit_deg),
                      ("rotY", bend_limit_deg),
                      ("rotZ", bend_limit_deg * 0.5)):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(float(-lim))
        limit.CreateHighAttr(float(lim))

    # drives pull each joint back toward straight; damping is what stops the
    # chain ringing after a drag is released
    for axis in ("rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(float(bend_stiffness))
        drive.CreateDampingAttr(float(bend_damping))


def build_duct_a(
    stage: Usd.Stage,
    root_path: str = "/World/DuctA",
    spec: DuctSpec | None = None,
    bend_limit_deg: float = 12.0,
    bend_stiffness: float = 20.0,
    bend_damping: float = 4.0,
    angular_damping: float = 0.0,
    ring_segments: int | None = None,
) -> dict:
    """Create the articulated duct. Returns handles used by the skin + runner."""
    spec = spec or DuctSpec()
    # The hoop's resting height is derived from the segment count (see
    # DuctSpec.drop_to_floor), so the two must not be allowed to disagree.
    if ring_segments is None:
        ring_segments = spec.ring_segments
    elif ring_segments != spec.ring_segments:
        raise ValueError(
            f"ring_segments={ring_segments} contradicts spec.ring_segments="
            f"{spec.ring_segments}; the floor height is computed from the spec "
            f"value, so overriding one without the other makes the duct hover "
            f"or sink"
        )
    UsdGeom.Xform.Define(stage, root_path)

    ring_paths = []
    for i in range(spec.num_rings):
        rp = f"{root_path}/ring_{i:03d}"
        create_ring(stage, rp, spec.radius, spec.ring_thickness, n_seg=ring_segments)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(rp))
        xf.AddTranslateOp().Set(Gf.Vec3d(*spec.ring_center(i)))
        if spec.axis_is_x:
            # create_ring builds the hoop in its own XY plane (axis = +Z). For a
            # duct lying along +X the hoop plane must be YZ, so rotate +Z onto
            # +X. The joint frames below are expressed along the same local +Z,
            # so they rotate with the ring and stay correct.
            xf.AddOrientOp().Set(
                Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0).GetQuat())
            )
        _apply_rigid_body(stage, rp, spec.ring_density, angular_damping)
        ring_paths.append(rp)

    # anchor(s)
    if spec.fix_first_ring:
        fj = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/anchor_first")
        fj.CreateBody1Rel().SetTargets([ring_paths[0]])
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    if spec.fix_last_ring:
        fj = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/anchor_last")
        fj.CreateBody1Rel().SetTargets([ring_paths[-1]])
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))

    # Ring-to-ring joints. Anchors sit on the midplane between the two hoops so
    # the joint frame is where the fabric actually bends, not at a ring centre.
    #
    # THE SIGN DEPENDS ON THE LAYOUT and getting it wrong is not a cosmetic
    # error. Anchors are expressed in each ring's LOCAL frame, whose +Z is the
    # duct axis. Hanging, ring i+1 sits at local -Z from ring i; on the floor
    # the hoops are rotated so local +Z is world +X and ring i+1 sits at local
    # +Z. Reusing the hanging signs on the floor puts the two anchors 0.15 m
    # apart instead of coincident, so PhysX yanks every ring by 0.1 m on the
    # first solve -- which presents as the duct tilting ~18 deg and bouncing to
    # z=0.66 m, i.e. it looks exactly like a physics/damping problem and is not.
    half = spec.ring_spacing * 0.5
    step = +half if spec.axis_is_x else -half
    for i in range(spec.num_rings - 1):
        _d6_between(
            stage,
            f"{root_path}/joint_{i:03d}",
            ring_paths[i],
            ring_paths[i + 1],
            Gf.Vec3f(0.0, 0.0, step),
            Gf.Vec3f(0.0, 0.0, -step),
            bend_limit_deg,
            bend_stiffness,
            bend_damping,
        )

    # Friction matters only in the floor layout, but applying it always keeps
    # the two layouts otherwise identical. Without it a poke sends the whole
    # duct skating across the ground instead of denting it locally.
    # A physics material is a UsdShade.Material carrying UsdPhysics.MaterialAPI,
    # and it is bound with UsdShade.MaterialBindingAPI using the "physics"
    # purpose. There is no UsdPhysics.MaterialBindingAPI -- reaching for one is
    # the obvious-looking mistake, and it fails at build time rather than
    # silently leaving the duct frictionless.
    mat_path = f"{root_path}/duct_material"
    mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
    mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat.CreateStaticFrictionAttr().Set(0.9)
    mat.CreateDynamicFrictionAttr().Set(0.8)
    mat.CreateRestitutionAttr().Set(0.05)
    material = UsdShade.Material(mat_prim)
    for rp in ring_paths:
        for seg in stage.GetPrimAtPath(rp).GetChildren():
            binding = UsdShade.MaterialBindingAPI.Apply(seg)
            binding.Bind(material,
                         UsdShade.Tokens.weakerThanDescendants,
                         "physics")

    return {"root": root_path, "rings": ring_paths, "spec": spec,
            "material": mat_path}
