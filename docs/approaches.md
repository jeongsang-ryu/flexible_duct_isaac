# Five ways to build a flexible duct in Isaac Sim 6.1

All five built, all five measured on the same bench: a **6 m duct, Ø0.4 m,
0.05 m hoop spacing, 121 stations**, headless, RTX 4080 SUPER, 400 warm-up
steps then 600 timed steps.

Step time is the **marginal** cost over those 600 steps, not a running average.
That distinction is not pedantry: on a full track the first 300 steps took 253 s
while PhysX claimed its GPU buffers, and a cumulative figure still read
842 ms/step there when the real cost was 63.5 ms — quoting the average made the
scene look 101× slower than real time instead of 7.6×.

| | ms/step | × real time | GPU | build | shapes | bodies |
|---|---|---|---|---|---|---|
| **static** | **0.35** | **0.04×** | 1,921 MiB | 0.04 s | 24 | 0 |
| **fem** | 1.70 | 0.20× | 2,969 MiB | 0.01 s | 1 | 1 |
| **rigid** | 3.38 | 0.40× | 2,353 MiB | 1.33 s | 241 | 241 |
| **cloth** | 5.40 | 0.65× | 7,247 MiB | 4.14 s | 1,936 | 121 |
| **hybrid** | 7.85 | 0.94× | 2,433 MiB | 1.48 s | 241 | 241 |

`× real time` below 1.0 means faster than wall clock at 120 Hz. **At 6 m all
five are real-time capable**; the differences only bite at track scale, where
the same cloth build measured 63.5 ms/step for 81 m and 1,629 hoops.

---

## static — one swept tube, nothing moves

A tube mesh swept along the centreline, with the radius modulated
`r(s) = R + a·sin(2πs/λ)` so the corrugations are geometry rather than bodies.
Colliders are a row of **capsules**, not a triangle mesh: contacts stay
analytic, which is faster and far better behaved against a car at speed.

**15× cheaper than the next option and 25× faster than real time.** No solver
work at all — the duct is scenery. Randomising the track per episode is
trivial: perturb the control points and re-sweep.

Cannot move, so a car cannot push it aside. Displacement has to be faked by
noising the control points between episodes.

**This is the one for parallel RL.**

## fem — a solid cylinder as a volume deformable

Deliberately **not** a hollow thin-walled tube: tetrahedralising a thin wall
needs a very dense mesh to get one element across it, which is expensive and
fragile. A solid low-resolution cylinder is the usual compromise.

Cheaper than expected (1.70 ms) because it is one body with one collider. It
squashes and recovers, but it has no hollow interior at all, and volume
deformables are GPU-only.

Watch the schema names: `PhysxDeformableBodyAPI` **does not exist** in 6.1 —
it is `PhysxBaseDeformableBodyAPI`, and the auto-hierarchy helper already
applies what the body needs.

## rigid — disc and sleeve chain

One thin black disc per hoop, one yellow cylinder per gap, hinged by D6 joints
whose rotational drives supply the compliance.

**Set mass, never density.** The sleeve body is a solid cylinder while a duct
is a thin-walled tube, so a plausible density made a 6 m duct weigh **139 kg** —
47× a real 400 mm duct's ~0.5 kg/m. Hundreds of overweight links on compliant
joints heave like a worm, and no joint tuning fixes it.

**Stiffness is the wrong knob for stability.** It resists gravity and a user's
hand equally: at stiffness 200 the duct was beautifully quiet (0.0034 m/s
residual) and 3 N moved its tip exactly **0.0000 m**. Damping resists only
speed, so it settles the chain without fighting a slow deliberate bend.

| stiffness | damping | residual m/s | waviness | tip bend under 3 N |
|---|---|---|---|---|
| 200 | 20 | 0.0034 | 0.001 | **0.000 m** |
| **30** | **30** | **0.0088** | **0.009** | **0.070 m** |
| 10 | 30 | 0.0178 | 0.036 | 0.076 m |
| 0 | 60 | 0.0085 | 0.073 | 0.016 m |

Note the last row: no spring at all, yet barely bends — damping that high
resists the hand too. It is not free either.

## cloth — PhysX surface deformable sewn to hoops

The most faithful: it crumples, folds and drapes, and a tight bend wrinkles on
the inside the way real ducting does. It is also the only build whose GPU
footprint is in a different class — **7.2 GB against ~2.4 GB** — because it
carries the deformable pipeline.

The hard-won part is where the fabric sits relative to the hoops, and why the
collision filter has to match the seam exactly. See the main README.

## hybrid — rigid physics, ribbed tube for the eye

The rigid chain does the physics with its bodies hidden; a high-resolution
ribbed tube is re-swept from their poses every other step.

**The most expensive of the five (7.85 ms)** — the skinning costs more than the
physics it wraps. Re-sweeping 2,420 vertices in Python every other step is the
naive implementation; Warp or `UsdSkel` would change that number, and this
figure should be read as an upper bound rather than the method's ceiling.

Which surface a sensor sees matters here, because the two differ by the rib
amplitude: an RTX lidar traces the **render** scene
(`omni.sensors.nv.lidar` depends on `omni.hydra.rtx`), a physics-query lidar
traces the **colliders**.

---

## Choosing

- **Parallel RL** → `static`. 0.35 ms, no bodies, trivially randomised.
- **A car that shoves the barrier** → `rigid`. Real displacement at 0.40× real
  time, and 16× fewer shapes than cloth.
- **Showing what a duct does** → `cloth`. Nothing else wrinkles.
- **Pretty sensor data on cheap physics** → `hybrid`, once the skinning moves
  off Python.
- **fem** is a curiosity here: cheap, but it models a solid rod, not a duct.

## Seeing them

All five build straight from the GUI, so they can be compared by eye rather
than only by table:

```bash
python scripts/duct_build_gui.py            # cloth
python scripts/duct_build_gui.py --rigid    # disc/sleeve chain
python scripts/duct_build_gui.py --fem      # solid continuum
python scripts/duct_build_gui.py --static   # swept tube, nothing moves
```

`--fem-youngs` sets how soft the continuum is (1e5 rubbery, 1e6 firm);
`--stiffness` / `--damping` / `--mass-per-m` tune the chain.

## Which surface a sensor sees — still unanswered

Whether a lidar returns the *simulated* duct or the *authored* one is the one
question here with no answer, and it matters most for `hybrid`, where the render
and collision surfaces differ by the rib amplitude by design.

`scripts/test_lidar_sees_bend.py` has been run three times and **every run was
invalid for a different reason**, each time my test's fault rather than a
sensor finding:

1. the push launched the free duct **140 m** away; the scan looked at the floor
   near the origin and found nothing, which says nothing about the sensor
2. pinning both ends left the middle nowhere to go — it bent **0.021 m**, a
   tenth of the duct's own radius, far too little to tell the two hypotheses
   apart
3. pinning one end produced a real **0.867 m** bend, but only **36 of 2,400
   rays** hit anything, all on three hoops — too small a sample to conclude from

The script now refuses to print a verdict when the bend is under 1.5× the duct
radius, so at least it can no longer report a conclusion it has not earned. The
honest next step is probably an actual RTX lidar in the viewport, looked at,
rather than a fourth scripted attempt.

## Reproducing

```bash
for m in static rigid hybrid cloth fem; do
  python scripts/bench_approaches.py --mode "$m"
done
```

## Caveats

Measured at **6 m**, headless, one run each, no repeats — treat small gaps as
noise. Scaling is not linear across methods: cloth went from 5.4 ms at 6 m to
63.5 ms at 81 m, and `static` should barely move. The 81 m numbers for the
other four have not been taken.
