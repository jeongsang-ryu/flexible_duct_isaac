# flexible_duct_isaac

A flexible ventilation duct in Isaac Sim 6.1 — rigid hoops every few centimetres
inside a continuous fabric sleeve — built to stand in for the soft barriers of a
RoboRacer track.

Draw the track in a browser, load it into Isaac, drag it into final position,
save the arranged layout back out as USD.

```
conda activate hmclab6
cd duct_sim

# 1. lay the track out           tools/track_planner.html  (open in a browser)
# 2. build it
python scripts/duct_build_gui.py --layout track.json \
    --ring-thickness 0.002 --ring-density 7800 --clearance -0.004 --rib-visual
# 3. place it        SHIFT + drag a hoop
# 4. keep it         R bakes the bend, S saves the layout to USD
```

Verified on Isaac Sim **6.1.0.0** / Python 3.12 / RTX 4080 SUPER.

---

## How the duct is put together

| part | what it is | why |
|---|---|---|
| hoop collider | 32+ capsules around a circle | PhysX rigid shapes must be **convex**; a torus collider needs decomposition, which comes out ragged on a thin hoop and jitters |
| hoop visual | a real torus (`--rib-visual`) | convexity constrains the *collider* only. Drawing the visual separately is what makes a hoop read as a rib on the duct instead of a fat tube standing off it |
| fabric | one continuous PhysX surface deformable | one sleeve per gap left a seam at every hoop that could open up |
| seam | per-vertex position constraints to the hoop | not a weld of surfaces: each captured vertex is held at a stored position **in the hoop's frame** |

### The fabric goes OUTSIDE the hoops, never on them

This is the single most important detail, and it took the longest to find.

A seam binds cloth vertices that are *inside* the hoop's collision shape. But
"inside a collider" is exactly what the solver treats as penetration, so on play
it pushes those same vertices back out — the fabric visibly pops off the hoop
the moment the simulation starts. Symptom: **no gap while stopped, a gap as soon
as you press play.**

Drawing the fabric clear of the tube removes the contradiction instead of
suppressing it. `--clearance` sets the gap and its sign picks the side:

```
--clearance  0.004   fabric outside, hoops hidden within it
--clearance -0.004   fabric inside,  hoops visible as outer ribs
```

The seam still binds because `deformableVertexOverlapOffset` attaches vertices
that are merely *near* the collider. Real ducting is built this way too.

### Collision filtering has to match the seam, not exceed it

`create_auto_deformable_attachment` writes only its two relationships and leaves
every other knob at its schema default — and two of those defaults are wrong
here:

```
collisionFilteringOffset        default -inf   filtering is "on" and filters nothing
deformableVertexOverlapOffset   default 0.0    only strictly-inside vertices bind
```

Measured behaviour across the range:

| filtering | welded vertices | free fabric | result |
|---|---|---|---|
| `-inf` (stock default) | collide | collide | hoop shoves off the fabric it holds → **gap** |
| far wider than the seam | filtered | **filtered** | hoop travels out through the skin → **pokes through** |
| off entirely | collide | collide | weld and contact fight → **collapses into a crumpled fan** |
| **= the capture band** | filtered | collide | holds |

`create_seam()` authors these before the setup call, because the setup call is
what bakes them in.

---

## Things that cost a lot of time

**GPU results never come back to USD.** Under GPU dynamics the simulated state
lives in Fabric. `mesh.GetPointsAttr().Get()` returns the authored pose forever —
measured: USD reported `0.000000 m` of movement while the cloth had fallen 1.0 m.
So `stage.Export()` after arranging a track silently saves the *straight* duct.
`duct_sim/freeze.py` reads cloth points through `usdrt` and rigid poses straight
from PhysX (Fabric had the cloth but **not** the transforms) and bakes both onto
the stage before exporting.

**A `RigidPrim` built before physics warm-up poisons every later read.** It
reports the authored pose forever. Symptom: a bend report printing `z=0.300`,
`y=0.000` every sample while the recording plainly shows the duct moving. Build
those views a few steps in.

**The built-in mouse grab cannot work in a cloth scene.** It calls
`PxRigidDynamic::addForce`, which PhysX refuses under `eENABLE_DIRECT_GPU_API` —
and cloth forces `device="cuda"`, which forces that flag on. Enabling
`omni.physx.ui` makes the gesture fire and error every frame until the window
dies. `duct_sim/mouse_drag.py` rebuilds the gesture on
`RigidPrim.apply_forces`, which is the legal path.

**Starting state matters, three times over.** Turning collision filtering off
from t=0 collapses the duct, but toggling to it after the duct settles is fine.
Self-collision from t=0 behaves differently from self-collision switched on
later. And a duct spawned resting exactly on the floor crumples, because the
first step has to resolve ground contact on every vertex at once — spawn it a
few centimetres up and let it fall.

**Hoops cannot be thinned freely.** The collider is a chain of capsules whose
axes are *chords*, so mid-chord it sits one sagitta `R(1-cos(π/n))` inside the
circle the cloth vertices lie on. Thinner tube ⇒ more segments, or the seams
bind nothing *and still return `True`*. The builder raises the segment count
automatically. At Ø0.4 m: 16 segments floors the tube radius at 3.84 mm, 32 at
0.96 mm.

**Self-collision filter distance goes ABOVE the mesh spacing, not below.** It
disables self-collision for pairs *closer* than the value. Set under the rest
spacing, every immediate neighbour counts as a self-contact, the solver pushes
them apart, and the fabric inflates and springs back hard.

**There is no plasticity parameter.** PhysX surface deformables have no yield
stress, so "stays where I bent it" means moving the target, not the material:
`duct_sim/plastic.py` rewrites `restShapePoints` and `restBendAngles` to the
current shape. `restBendAnglesDefault` is `flatDefault`, so any pair without an
explicit rest angle wants to be flat. Rewriting the rest shape alone does
nothing visible — PhysX holds the body cooked on the GPU, so the change needs a
stop/play, and the arranged geometry must be baked into USD first or the restart
snaps it straight.

---

## Measurements

Ø0.4 m duct, 0.05 m hoop spacing, 2 mm hoop tube, RTX 4080 SUPER.

**Mesh resolution is nearly free at this scale** — the GPU sits at ~24 %, so
more work fits in the same time:

| `--n-circ` | cloth vertices | triangles | ms/step |
|---|---|---|---|
| 28 | 1,708 | 3,360 | 5.0 |
| 56 | 3,416 | 6,720 | 5.3 |
| 84 | 7,644 | 15,120 | 4.5 |

Under an identical force the finer mesh bends far more (0.012 → 0.257 m): a
coarse mesh has nowhere to fold, so it is artificially stiff. Single
measurement — treat the magnitude as indicative.

**The cloth is almost free; Isaac itself is not.** 38 sleeves versus 1 cost
**2 MiB** of GPU memory and +11 % step time. The 7.2 GB resident is the RTX
renderer and framework.

**The duct stretches, and only in one direction.** Nothing but the fabric sets
hoop spacing — there are no joints between hoops — and fabric resists tension
but cannot push. At `--stretch 5000` a 2.30 m duct measured 3.09 m (+34 %).
Raising stretch stiffness helps, but a distance constraint between neighbouring
hoops is the right fix and is not implemented yet.

---

## Layout

```
duct_sim/
  spec.py        one source of duct dimensions
  geometry.py    hoop colliders (capsule chain) and the display torus
  builder.py     spawn straight / along a drawn path; seams; seam rebuild
  layout.py      read a planner export, resample to hoop stations
  freeze.py      read the SIMULATED state and bake it into USD
  plastic.py     rewrite the rest shape so a bend stops springing back
  mouse_drag.py  shift+drag hoops through the GPU-legal force path
scripts/
  duct_build_gui.py       the tool: spawn, place, bake, save
  record_gui.sh           film a GUI scene in a nested X server
  test_*.py               the measurements the README quotes
tools/
  track_planner.html      the 2-D planner (open directly, works offline)
```

## Open

- No distance constraint between hoops, so the duct can still stretch.
- Whether a LiDAR sees the *simulated* duct or the authored one is unverified —
  `scripts/test_lidar_sees_bend.py` is written but has not been run. The sensor
  depends on `omni.hydra.rtx`, i.e. it traces the renderer's scene, which is
  Fabric-fed, so it probably does; that is an inference, not a result.
- Full 30×20 m track (1000+ hoops) has not been built yet.
