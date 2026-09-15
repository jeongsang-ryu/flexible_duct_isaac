"""Approach B -- rigid hoops with a REAL simulated fabric sleeve.

Same hoops as approach A, but the fabric between them is a PhysX particle
cloth: one continuous sleeve running the whole length, attached to every hoop.
That matches how a flex duct is actually made (fabric is continuous, hoops are
sewn into it) and it is what lets the fabric wrinkle, go slack and billow --
none of which approach A can show, because there the fabric is only drawn.

THE COST, STATED UP FRONT. The sleeve is n_circ x n_axial particles; at the
defaults below that is 32 x 120 = 3840 particles with stretch/bend constraints,
solved every substep, plus one attachment per hoop. This is the arm that can
become unstable under an aggressive mouse drag, and the parameters that control
that trade-off (`solver_position_iterations`, `particle_contact_offset`,
stretch/bend stiffness) are arguments rather than buried constants.

PARTICLE SPACING IS DERIVED, NOT CHOSEN. PhysX wants the particle contact
offset to be a little larger than the actual inter-particle spacing; if it is
smaller the sheet self-interpenetrates, if it is much larger the solver
stiffens and the cloth behaves like cardboard. So it is computed from the mesh
resolution here instead of being a magic number.
"""

from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from .geometry import create_ring, create_tube_mesh
from .spec import DuctSpec


def build_duct_b(
    stage: Usd.Stage,
    root_path: str = "/World/DuctB",
    spec: DuctSpec | None = None,
    n_circ: int = 32,
    axial_per_gap: int = 6,
    stretch_stiffness: float = 10000.0,
    bend_stiffness: float = 200.0,
    shear_stiffness: float = 500.0,
    cloth_density: float = 120.0,       # kg/m^3, light woven fabric
    solver_position_iterations: int = 16,
    rings_dynamic: bool = True,
    ring_segments: int = 16,
) -> dict:
    """Create hoops + a simulated fabric sleeve attached to each hoop."""
    spec = spec or DuctSpec()
    UsdGeom.Xform.Define(stage, root_path)

    # ---------------- particle system ----------------
    ps_path = f"{root_path}/particleSystem"
    particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, ps_path)

    # spacing between neighbouring cloth particles, in metres
    circ_spacing = (2.0 * math.pi * spec.radius) / n_circ
    axial_spacing = spec.ring_spacing / axial_per_gap
    rest_offset = 0.6 * min(circ_spacing, axial_spacing)
    contact_offset = rest_offset * 1.5

    particle_system.CreateParticleContactOffsetAttr().Set(float(contact_offset))
    particle_system.CreateRestOffsetAttr().Set(float(rest_offset))
    particle_system.CreateSolidRestOffsetAttr().Set(float(rest_offset))
    particle_system.CreateFluidRestOffsetAttr().Set(float(rest_offset * 0.6))
    particle_system.CreateSolverPositionIterationCountAttr().Set(int(solver_position_iterations))
    particle_system.CreateEnableCCDAttr().Set(True)
    particle_system.CreateMaxVelocityAttr().Set(50.0)

    # ---------------- hoops ----------------
    ring_paths = []
    for i in range(spec.num_rings):
        rp = f"{root_path}/ring_{i:03d}"
        create_ring(stage, rp, spec.radius, spec.ring_thickness, n_seg=ring_segments)
        UsdGeom.Xformable(stage.GetPrimAtPath(rp)).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, spec.ring_z(i))
        )
        prim = stage.GetPrimAtPath(rp)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(float(spec.ring_density))
        if not rings_dynamic:
            # kinematic hoops are the debugging configuration: the fabric still
            # simulates but the hoops cannot be pushed around by it
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        ring_paths.append(rp)

    if spec.fix_first_ring:
        fj = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/anchor_first")
        fj.CreateBody1Rel().SetTargets([ring_paths[0]])
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    if spec.fix_last_ring:
        fj = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/anchor_last")
        fj.CreateBody1Rel().SetTargets([ring_paths[-1]])
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))

    # ---------------- fabric sleeve ----------------
    n_axial = axial_per_gap * (spec.num_rings - 1)
    cloth_path = f"{root_path}/fabric"
    mesh = create_tube_mesh(
        stage, cloth_path, spec.radius,
        z_top=spec.ring_z(0), z_bot=spec.ring_z(spec.num_rings - 1),
        n_circ=n_circ, n_axial=n_axial,
    )
    cloth_api = PhysxSchema.PhysxParticleClothAPI.Apply(mesh.GetPrim())
    PhysxSchema.PhysxAutoParticleClothAPI.Apply(mesh.GetPrim())
    cloth_api.CreateParticleSystemRel().SetTargets([Sdf.Path(ps_path)])
    cloth_api.CreateSelfCollisionAttr().Set(True)
    # springs are authored by the auto-cloth API from the quad topology
    auto = PhysxSchema.PhysxAutoParticleClothAPI(mesh.GetPrim())
    auto.CreateSpringStretchStiffnessAttr().Set(float(stretch_stiffness))
    auto.CreateSpringBendStiffnessAttr().Set(float(bend_stiffness))
    auto.CreateSpringShearStiffnessAttr().Set(float(shear_stiffness))
    auto.CreateSpringDampingAttr().Set(0.2)

    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateDensityAttr(float(cloth_density))

    # ---------------- sew fabric to each hoop ----------------
    # One attachment per hoop. Auto-attachment picks the cloth vertices that lie
    # within the hoop's collision shape, which for a sleeve built at exactly the
    # hoop radius is the ring of vertices at that height -- i.e. the seam.
    attachments = []
    for i, rp in enumerate(ring_paths):
        att_path = f"{root_path}/attach_{i:03d}"
        att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, att_path)
        att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path)])
        att.GetActor1Rel().SetTargets([Sdf.Path(rp)])
        auto_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
        auto_api.CreateEnableDeformableVertexAttachmentsAttr().Set(True)
        auto_api.CreateEnableRigidSurfaceAttachmentsAttr().Set(True)
        # radius is in world units and must reach from the sleeve surface to the
        # hoop's tube centre, else the seam finds no candidate vertices and the
        # fabric silently falls off the hoops
        auto_api.CreateDeformableVertexOverlapOffsetAttr().Set(
            float(spec.ring_thickness * 3.0)
        )
        attachments.append(att_path)

    return {
        "root": root_path,
        "rings": ring_paths,
        "cloth": cloth_path,
        "particle_system": ps_path,
        "attachments": attachments,
        "spec": spec,
        "resolution": (n_circ, n_axial),
        "particle_count": n_circ * (n_axial + 1),
    }
