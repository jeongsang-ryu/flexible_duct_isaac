"""Physical specification of the flexible duct, shared by both implementations.

The two approaches (A: articulated rigid rings, B: rings + PhysX particle
cloth) must describe the SAME object, or comparing them tells you nothing
about the approaches and only about the geometry. Everything dimensional
therefore lives here and is imported by both -- no number is written twice.

Geometry as specified by the lab (2026-09-14):
    duct inner diameter   0.40 m
    ring (fork) spacing   0.10 m
    fabric                spans each 0.10 m gap between consecutive rings

LENGTH AND END CONSTRAINT were not specified. Defaults below are a hanging
2 m duct fixed at one end, which is the configuration that shows bending,
sag and swing under a mouse drag -- i.e. the thing that has to look natural.
Both are constructor arguments, so changing them is a one-line edit, not a
rewrite.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DuctSpec:
    # --- given ---
    diameter: float = 0.40           # m, duct inner diameter
    ring_spacing: float = 0.10       # m, distance between consecutive rings

    # --- chosen defaults (see module docstring) ---
    length: float = 2.00             # m, total duct length

    # LAYOUT. "floor" lays the duct along +X resting on the ground, both ends
    # free, so pushing any part of it deforms THAT part locally -- this is the
    # configuration for poking at it. "hang" suspends it from z=0 downward,
    # which shows bending and swing but means a push mostly swings the whole
    # thing instead of denting it.
    layout: str = "floor"

    fix_first_ring: bool = False     # anchor ring 0 to the world
    fix_last_ring: bool = False      # anchor the far end

    # --- ring cross-section ---
    # A real flex-duct hoop is a thin stiffening wire, not a structural tube.
    # 8 mm keeps it visible in the viewport while staying light enough that
    # the chain's dynamics are dominated by the joints, not by ring inertia.
    ring_thickness: float = 0.008    # m, radius of the ring's circular cross-section
    ring_density: float = 400.0      # kg/m^3, light plastic-coated wire
    ring_segments: int = 16          # capsules per hoop (see drop_to_floor)

    @property
    def radius(self) -> float:
        return self.diameter * 0.5

    @property
    def num_rings(self) -> int:
        # +1 because both endpoints carry a ring: a 2.0 m duct at 0.1 m
        # spacing has 21 rings, not 20, and an off-by-one here would silently
        # make the duct 0.1 m short.
        return int(round(self.length / self.ring_spacing)) + 1

    def ring_z(self, i: int) -> float:
        """Height of ring i in the HANGING layout, measured down from z=0."""
        return -i * self.ring_spacing

    def ring_center(self, i: int) -> tuple:
        """World position of ring i's centre for the active layout."""
        if self.layout == "hang":
            return (0.0, 0.0, self.ring_z(i))
        # floor: axis along +X, centred on the origin, lifted so the hoop's
        # lowest point just touches z=0.
        x = (i - (self.num_rings - 1) * 0.5) * self.ring_spacing
        return (x, 0.0, self.drop_to_floor)

    @property
    def drop_to_floor(self) -> float:
        """Centre height that puts the hoop's lowest SURFACE exactly on z=0.

        This is `radius + ring_thickness`, the CIRCUMRADIUS, and the reasoning
        matters because the obvious-looking correction is wrong. The hoop is a
        polygon of capsule chords, so it is tempting to use the inradius
        (radius*cos(pi/n), where the chord MIDPOINTS sit). But a capsule is a
        swept sphere covering its entire chord including the endpoints, and the
        endpoints are the polygon VERTICES, which lie on the circumradius. With
        n=16 a vertex falls exactly at the bottom (270 deg is a multiple of
        22.5 deg), so the lowest surface is vertex - tube_radius.

        Using the inradius here sinks the duct 3.8 mm into the floor.
        """
        return self.radius + self.ring_thickness

    @property
    def axis_is_x(self) -> bool:
        return self.layout != "hang"

    def describe(self) -> str:
        return (
            f"duct: D={self.diameter:.2f} m, L={self.length:.2f} m, "
            f"{self.num_rings} rings @ {self.ring_spacing:.2f} m, "
            f"layout={self.layout}, "
            f"fixed_first={self.fix_first_ring}, fixed_last={self.fix_last_ring}"
        )
