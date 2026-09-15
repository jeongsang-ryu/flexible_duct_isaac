"""Procedural USD geometry for the duct: hoop rings and the fabric tube.

Kept free of Isaac-specific imports on purpose -- everything here is plain
pxr.Usd/UsdGeom, so both implementations (A: articulated rings, B: particle
cloth) build the SAME geometry and any visual difference between them comes
from the physics, not from one of them drawing a slightly different duct.

Why rings are built from capsule SEGMENTS rather than a torus mesh: a torus is
non-convex, so a torus collider needs convex decomposition, which for a thin
hoop produces a ragged approximation and is the usual source of jitter in a
chain of stacked rings. N capsules laid around a circle are each convex,
exactly represent the hoop's cross-section, and cost less.
"""

from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


def ring_segment_transforms(radius: float, n_seg: int):
    """Yield (translation, orientation, half_height) for each capsule segment.

    Each segment is a capsule spanning one chord of the circle, rotated to lie
    tangentially. Using the CHORD length (not the arc length) is what keeps the
    segment endpoints exactly on the circle -- using the arc would make the
    hoop slightly too large and, with 16 segments, visibly polygonal.
    """
    for i in range(n_seg):
        a0 = 2.0 * math.pi * i / n_seg
        a1 = 2.0 * math.pi * (i + 1) / n_seg
        p0 = Gf.Vec3d(radius * math.cos(a0), radius * math.sin(a0), 0.0)
        p1 = Gf.Vec3d(radius * math.cos(a1), radius * math.sin(a1), 0.0)
        mid = (p0 + p1) * 0.5
        chord = p1 - p0
        half_h = chord.GetLength() * 0.5
        # capsule's local axis is +Z by default in UsdGeom; rotate +Z onto the chord
        d = chord.GetNormalized()
        z = Gf.Vec3d(0, 0, 1)
        axis = Gf.Cross(z, d)
        if axis.GetLength() < 1e-9:
            rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), 0.0)
        else:
            angle = math.degrees(math.acos(max(-1.0, min(1.0, Gf.Dot(z, d)))))
            rot = Gf.Rotation(axis.GetNormalized(), angle)
        yield mid, rot, half_h


def create_ring(
    stage: Usd.Stage,
    path: str,
    radius: float,
    tube_radius: float,
    n_seg: int = 16,
    with_collision: bool = True,
) -> UsdGeom.Xform:
    """A hoop at `path`, centred on the origin of its own frame, in the XY plane."""
    xform = UsdGeom.Xform.Define(stage, path)
    for i, (mid, rot, half_h) in enumerate(ring_segment_transforms(radius, n_seg)):
        seg_path = f"{path}/seg_{i:02d}"
        cap = UsdGeom.Capsule.Define(stage, seg_path)
        cap.CreateAxisAttr("Z")
        cap.CreateRadiusAttr(float(tube_radius))
        cap.CreateHeightAttr(float(max(2.0 * half_h - 2.0 * tube_radius, 1e-4)))
        xf = UsdGeom.Xformable(cap)
        xf.AddTranslateOp().Set(mid)
        xf.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
        if with_collision:
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return xform


def create_tube_mesh(
    stage: Usd.Stage,
    path: str,
    radius: float,
    z_top: float,
    z_bot: float,
    n_circ: int = 32,
    n_axial: int = 8,
) -> UsdGeom.Mesh:
    """An open cylindrical sleeve (no caps) between two heights.

    Quad topology, not triangles: PhysX particle cloth builds its stretch and
    bend constraints from the mesh edges, and a quad grid gives an even
    constraint layout. Triangulating first adds diagonal edges that stiffen the
    sheet anisotropically -- the fabric then resists shearing one way more than
    the other, which reads as an unnatural crease pattern.
    """
    pts, counts, idx = [], [], []
    for j in range(n_axial + 1):
        t = j / n_axial
        z = z_top + (z_bot - z_top) * t
        for i in range(n_circ):
            a = 2.0 * math.pi * i / n_circ
            pts.append(Gf.Vec3f(radius * math.cos(a), radius * math.sin(a), z))
    for j in range(n_axial):
        for i in range(n_circ):
            i2 = (i + 1) % n_circ
            a = j * n_circ + i
            b = j * n_circ + i2
            c = (j + 1) * n_circ + i2
            d = (j + 1) * n_circ + i
            counts.append(4)
            idx.extend([a, b, c, d])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateDoubleSidedAttr(True)   # a sleeve is visible from inside too
    return mesh


def create_ring_visual(stage, path, radius, tube_radius,
                       n_major=48, n_minor=10, colour=(0.05, 0.05, 0.05)):
    """A real torus for DISPLAY only -- no collision, no convexity requirement.

    The hoop's collider has to be a chain of capsules because PhysX rigid
    shapes must be convex. Nothing forces the VISUAL to be the same thing, and
    using the capsules as the visual is what makes the hoop read as a fat tube
    standing off the fabric. Drawing an actual torus at the fabric's own radius
    makes it read as a rib on the duct, which is what real ducting looks like,
    while the capsule chain keeps doing the physics underneath (hidden).
    """
    import numpy as np

    pts, tris = [], []
    for i in range(n_major):
        a = 2.0 * math.pi * i / n_major
        ca, sa = math.cos(a), math.sin(a)
        for j in range(n_minor):
            b = 2.0 * math.pi * j / n_minor
            r = radius + tube_radius * math.cos(b)
            pts.append([r * ca, r * sa, tube_radius * math.sin(b)])
    for i in range(n_major):
        i2 = (i + 1) % n_major
        for j in range(n_minor):
            j2 = (j + 1) % n_minor
            a = i * n_minor + j
            b = i * n_minor + j2
            c = i2 * n_minor + j2
            d = i2 * n_minor + j
            tris += [[a, b, c], [a, c, d]]

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pts, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray(np.array(tris, dtype=np.int32).flatten().tolist()))
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*colour)])
    return mesh
