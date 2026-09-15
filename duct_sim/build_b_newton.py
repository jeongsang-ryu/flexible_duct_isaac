"""Approach B -- real simulated fabric, via Newton cloth coupled to PhysX hoops.

WHY A HYBRID RATHER THAN ALL-NEWTON. Isaac Sim 6.1 removed PhysX's cloth and
attachment schemas (verified: PhysxParticleClothAPI, PhysxAutoParticleClothAPI,
PhysxPhysicsAttachment and PhysxAutoAttachmentAPI are all gone; only the fluid
/ granular particle APIs remain), so cloth has to come from Newton. But moving
the HOOPS into Newton too would cost the thing that made approach A usable:
Kit's built-in mouse grab acts on PhysX rigid bodies, and there is no
equivalent for Newton bodies without writing the interaction from scratch.

So the hoops stay exactly as they are in approach A -- PhysX rigid bodies,
mouse-draggable, already validated -- and only the fabric moves to Newton:

    PhysX steps the hoops  ->  read hoop poses
                           ->  drive the pinned seam particles to match
    Newton steps the cloth ->  read particle positions
                           ->  write them into a USD mesh for display

HOW THE FABRIC IS SEWN ON. Newton's ModelBuilder has no "attach particle to
rigid body" constraint, so the seams are made KINEMATIC instead: the ring of
particles sitting at each hoop is given mass 0, which pins it, and its position
is rewritten every step from that hoop's current transform. A zero-mass
particle is not integrated, so it follows exactly and contributes no force back
into the cloth solve -- the fabric hangs off the hoops rather than dragging
them around. That one-way coupling is a real limitation and is stated in
`known_limitations` below rather than left to be discovered.
"""

from __future__ import annotations

import math

import numpy as np

from .spec import DuctSpec


def _mat_to_quat(m):
    """3x3 rotation -> (x, y, z, w), the order Newton/warp transforms expect."""
    import numpy as _np
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s_ = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s_
        x = (m[2][1] - m[1][2]) / s_
        y = (m[0][2] - m[2][0]) / s_
        z = (m[1][0] - m[0][1]) / s_
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s_ = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s_
        x = 0.25 * s_
        y = (m[0][1] + m[1][0]) / s_
        z = (m[0][2] + m[2][0]) / s_
    elif m[1][1] > m[2][2]:
        s_ = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s_
        x = (m[0][1] + m[1][0]) / s_
        y = 0.25 * s_
        z = (m[1][2] + m[2][1]) / s_
    else:
        s_ = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s_
        x = (m[0][2] + m[2][0]) / s_
        y = (m[1][2] + m[2][1]) / s_
        z = 0.25 * s_
    return float(x), float(y), float(z), float(w)


def _hoop_segments(radius: float, n_seg: int):
    """Capsule placements for one hoop, matching geometry.create_ring exactly.

    Duplicated in Newton's own convention rather than imported, because the two
    engines take different transform types -- but the LAYOUT must stay identical
    or the collider and the visible hoop are different objects.
    """
    import numpy as _np
    out = []
    for i in range(n_seg):
        a0 = 2.0 * math.pi * i / n_seg
        a1 = 2.0 * math.pi * (i + 1) / n_seg
        p0 = _np.array([radius * math.cos(a0), radius * math.sin(a0), 0.0])
        p1 = _np.array([radius * math.cos(a1), radius * math.sin(a1), 0.0])
        mid = (p0 + p1) * 0.5
        chord = p1 - p0
        half_h = float(_np.linalg.norm(chord) * 0.5)
        d = chord / max(_np.linalg.norm(chord), 1e-12)
        z = _np.array([0.0, 0.0, 1.0])
        axis = _np.cross(z, d)
        na = _np.linalg.norm(axis)
        if na < 1e-9:
            quat = (0.0, 0.0, 0.0, 1.0)
        else:
            axis = axis / na
            ang = math.acos(max(-1.0, min(1.0, float(_np.dot(z, d)))))
            sa = math.sin(ang * 0.5)
            quat = (float(axis[0] * sa), float(axis[1] * sa),
                    float(axis[2] * sa), math.cos(ang * 0.5))
        out.append((mid, quat, max(half_h - radius * 0.0, 1e-4)))
    return out


class NewtonFabric:
    """A simulated fabric sleeve whose seams follow externally-driven hoops."""

    def __init__(
        self,
        spec: DuctSpec,
        ring_world_fn,
        n_circ: int = 24,
        rings_per_gap: int = 4,
        # LIGHT FABRIC. 0.2 kg/m^2 was heavier than the duct it hangs on: the
        # sleeve's area is pi*D*L = 2.5 m^2 at 2 m length, so 0.2 gives ~0.50 kg
        # of cloth against ~1.96 kg of hoops -- a quarter of the duct's mass in
        # fabric, which visibly dragged the whole thing down. Real flexible-duct
        # skin is thin coated polyester nearer 0.08-0.12 kg/m^2; 0.05 here
        # deliberately sits at the light end so the fabric follows the hoops
        # instead of loading them.
        density: float = 0.05,         # kg/m^2 areal
        # STIFFNESS IS SCALED WITH THE MASS. An explicit solver's stability
        # depends on k/m, not on k: the same tri_ke that was fine at
        # 0.2 kg/m^2 is 4x as violent at 0.05, because each particle now weighs
        # 6.5e-5 kg (inverse mass ~15,000). Lightening the fabric for looks
        # without dropping the springs is what produced NaN particles after
        # ~400 steps. 250 keeps k/m where it was at the heavier setting.
        tri_ke: float = 2.5e2,         # in-plane stretch
        tri_ka: float = 2.5e2,         # area preservation
        tri_kd: float = 5.0e0,         # in-plane damping
        edge_ke: float = 1.0e-2,       # bending; low = limp fabric, high = card
        edge_kd: float = 1.0e-4,
        iterations: int = 12,
        radius_clearance: float = 0.012,
        slack: float = 0.10,
        collide_floor: bool = True,
        collide_hoops: bool = True,
        floor_z: float = 0.0,
        device: str | None = None,
    ):
        import newton
        import warp as wp

        self._wp = wp
        self.spec = spec
        self.ring_world_fn = ring_world_fn
        self.n_circ = n_circ
        self.rings_per_gap = rings_per_gap

        n_ring = spec.num_rings
        # One loop of vertices per hoop, plus `rings_per_gap` free loops in each
        # gap. Only the hoop loops are pinned; the free loops are what can
        # wrinkle and sag, so rings_per_gap is the knob that decides whether the
        # fabric can show slack at all. At 0 the sleeve would be pinned
        # everywhere and could not move.
        self.loops_per_gap = rings_per_gap + 1
        self.n_loops = (n_ring - 1) * self.loops_per_gap + 1
        # The sleeve has to enclose the hoops, so its radius is measured from
        # the hoop's OUTER surface (radius + tube_radius), not from the hoop
        # centreline. The first version scaled the centreline radius by 1.02,
        # which put the fabric 4 mm INSIDE the hoop surface -- so every hoop
        # protruded through the fabric everywhere, not just at bends.
        self.radius = spec.radius + spec.ring_thickness + radius_clearance
        # SLACK IS WHAT LETS IT BEND. The sleeve is sewn to a hoop every 0.1 m,
        # and a mesh built taut between those seams has exactly the rest length
        # of a straight duct. Bend it and the OUTSIDE of the bend needs more
        # material than exists, so the fabric pulls itself straight across the
        # corner -- passing inside the hoop radius and letting the hoops stand
        # out through it. That is what the spikes along the arch were. Real
        # flex-duct skin is gathered between its hoops precisely so it has
        # material to give up when the duct bends.
        self.slack = slack

        verts, faces, self.seam_loops = self._build_mesh()
        self.n_particles = len(verts)

        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*v) for v in verts],
            indices=faces.flatten().tolist(),
            density=density,
            tri_ke=tri_ke, tri_ka=tri_ka, tri_kd=tri_kd,
            edge_ke=edge_ke, edge_kd=edge_kd,
            particle_radius=0.004,
        )

        # pin the seams: mass 0 => kinematic, driven from the hoops each step
        self.seam_indices = np.concatenate(
            [np.arange(l * n_circ, (l + 1) * n_circ) for l in self.seam_loops]
        )
        for idx in self.seam_indices:
            builder.particle_mass[int(idx)] = 0.0

        # HOOP COLLIDERS. Pinning the seams is not enough: between seams the
        # fabric has nothing under it and gravity pulls it to a measured 0.173 m
        # radius, inside the hoops' 0.208 m outer surface, so the hoops stand
        # out through the sleeve. Newton cannot see the PhysX hoops, so each one
        # is mirrored here as a kinematic body carrying a ring of capsules, and
        # its transform is rewritten every step from the PhysX pose.
        self.hoop_bodies = []
        if collide_hoops:
            import newton as _n
            for i in range(n_ring):
                bid = builder.add_body(xform=wp.transform_identity(),
                                       is_kinematic=True)
                for (mid, rot, half_h) in _hoop_segments(spec.radius,
                                                         spec.ring_segments):
                    builder.add_shape_capsule(
                        body=bid,
                        xform=wp.transform(wp.vec3(*mid), wp.quat(*rot)),
                        radius=spec.ring_thickness,
                        half_height=float(half_h),
                    )
                self.hoop_bodies.append(bid)

        if collide_floor:
            # The PhysX ground plane does not exist as far as Newton is
            # concerned: the two run in separate solvers with separate worlds.
            # Without this the sleeve falls straight through the floor and
            # piles up below it, which is what the first drape render showed.
            builder.add_shape_plane(
                plane=(0.0, 0.0, 1.0, -float(floor_z)),
                width=20.0, length=20.0, body=-1,
            )

        # SolverVBD is Gauss-Seidel over vertices: it updates a whole colour
        # group in parallel, so it needs a graph colouring where no two
        # vertices in a group share an edge. Newton does not do this
        # implicitly -- finalize() succeeds and the solver then refuses to
        # construct with "model.particle_color_groups is empty".
        # include_bending=True colours the bending edges as well, which matters
        # here because bending is what makes fabric look like fabric.
        builder.color(include_bending=True)

        self.model = builder.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        from newton.solvers import SolverVBD
        self.solver = SolverVBD(self.model, iterations=iterations)

        self._unit = np.array(
            [[math.cos(2 * math.pi * i / n_circ),
              math.sin(2 * math.pi * i / n_circ), 0.0] for i in range(n_circ)]
        )
        self.faces = faces

    # ------------------------------------------------------------------
    def _build_mesh(self):
        """Vertices, triangle indices, and which loops sit on a hoop.

        The sleeve is built from the hoops' ACTUAL current poses, read through
        ring_world_fn, not from DuctSpec.ring_center. Those two agree only when
        the duct is still in its authored pose; in the drape scene the hoops are
        lifted onto a 2 m bar after construction, and seeding the fabric at the
        authored floor height instead meant the first drive_seams() teleported
        every seam 2 m upward while the free loops stayed behind. The cloth was
        stretched by ~10x its rest length in one step and detonated. Reading the
        real poses makes the fabric start exactly on the hoops wherever they are.
        """
        n_ring = self.spec.num_rings
        frames = [self.ring_world_fn(i) for i in range(n_ring)]
        verts = []
        seam_loops = []
        loop = 0
        for k in range(n_ring - 1):
            (c0, b0), (c1, b1) = frames[k], frames[k + 1]
            for s in range(self.loops_per_gap):
                t = s / self.loops_per_gap
                c = c0 * (1 - t) + c1 * t
                b = b0 if t < 0.5 else b1
                if s == 0:
                    seam_loops.append(loop)
                # bulge to zero at each seam and maximum midway between them
                rr = self.radius * (1.0 + self.slack * math.sin(math.pi * t))
                for i in range(self.n_circ):
                    a = 2 * math.pi * i / self.n_circ
                    local = np.array([rr * math.cos(a), rr * math.sin(a), 0.0])
                    verts.append(c + b @ local)
                loop += 1
        seam_loops.append(loop)
        c, b = frames[n_ring - 1]
        for i in range(self.n_circ):
            a = 2 * math.pi * i / self.n_circ
            local = np.array([self.radius * math.cos(a),
                              self.radius * math.sin(a), 0.0])
            verts.append(c + b @ local)

        faces = []
        for j in range(self.n_loops - 1):
            for i in range(self.n_circ):
                i2 = (i + 1) % self.n_circ
                a = j * self.n_circ + i
                b = j * self.n_circ + i2
                c_ = (j + 1) * self.n_circ + i2
                d = (j + 1) * self.n_circ + i
                faces.append([a, b, c_])
                faces.append([a, c_, d])
        return np.array(verts), np.array(faces, dtype=np.int32), seam_loops

    # ------------------------------------------------------------------
    def drive_seams(self):
        """Move the pinned seam rings onto the hoops' current poses."""
        q = self.state_0.particle_q.numpy()
        for seam_i, loop in enumerate(self.seam_loops):
            centre, basis = self.ring_world_fn(seam_i)
            ring_pts = centre + (basis @ (self._unit * self.radius).T).T
            q[loop * self.n_circ:(loop + 1) * self.n_circ] = ring_pts
        self.state_0.particle_q.assign(q)

    def drive_hoops(self):
        """Move the mirrored hoop colliders onto the PhysX hoops' poses."""
        if not self.hoop_bodies:
            return
        wp = self._wp
        bq = self.state_0.body_q.numpy()
        for i, bid in enumerate(self.hoop_bodies):
            centre, basis = self.ring_world_fn(i)
            qx, qy, qz, qw = _mat_to_quat(basis)
            bq[bid] = [centre[0], centre[1], centre[2], qx, qy, qz, qw]
        self.state_0.body_q.assign(bq)

    def step(self, dt: float, substeps: int = 8):
        self.drive_seams()
        self.drive_hoops()
        sub = dt / substeps
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def positions(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()

    @staticmethod
    def known_limitations() -> list[str]:
        return [
            "one-way coupling: the fabric follows the hoops but exerts no force "
            "back on them, so it cannot damp hoop motion or stop the duct rolling",
            "no cloth-hoop collision: between seams the fabric can pass through "
            "a hoop under a hard bend",
            "the hoops are PhysX bodies and the fabric is a Newton body, so they "
            "are stepped by two different solvers at the same dt",
        ]
