# What a track costs, measured

81 m layout (`track.json`), 9 ducts, 1,629 hoops, RTX 4080 SUPER.
Marginal step time over 300 steps after 300 warm-up, physics and render timed
separately. `dt` is **1/60 s, read from `sim.get_physics_dt()`** -- not assumed.

## The three builds

| build | ms/step (physics) | x real time @60Hz | GPU MiB | build | bodies |
|---|---|---|---|---|---|
| cloth (capsule hoops) | 25.86 | 1.55 | 3,451 | 54 s | 1,629 |
| rigid | 19.60 | 1.18 | 4,100 | ~2 min | 1,629 |
| static (frozen) | **0.32** | **0.019** | 1,941 | 0.5 s | **0** |

With rendering, the cloth track is 41.3 ms (2.48x real time) and static is
4.5 ms.

## Static does not grow with the track

0.35 ms at 6 m, 0.32 ms at 81 m. There are no rigid bodies -- only 324 static
capsule colliders sitting in the broadphase -- so length costs nothing. A frozen
track leaves essentially the whole step budget to the vehicles. **What limits
how many cars fit is the cars, not the track.**

## At 200 Hz

Contact-rich vehicle dynamics usually wants more than 60 Hz. Step cost is not
generally constant in `dt` -- a smaller step converges with less work -- so
scaling a 60 Hz figure by 200/60 is an upper bound, not a result. Measured:

| build | 60 Hz | 200 Hz | x real time @200Hz |
|---|---|---|---|
| static | 0.32 ms | 0.30 ms | **0.060** |
| cloth | 35.06 ms | 21.28 ms | **4.26** |

Static's step cost is flat, so scaling happened to be right there. Cloth's is
not: it **falls 39%** at the smaller step, because less penetration accumulates
and the solver iterates less. Scaling the 60 Hz figure would have said 7.0x
real time; the measurement is 4.26x. The upper bound was 65% pessimistic, which
is why it was worth measuring rather than multiplying.

Cloth timings repeat poorly: the same cloth_60 configuration measured 25.86 ms
in one run and 35.06 in another, a 36% spread. Treat every cloth figure here as
carrying that. Static repeats to within 0.02 ms.

**Consequence for a real-time ROS run:** the walls have to be the frozen static
build. 0.3 ms of a 5 ms budget leaves 4.7 ms for vehicle, sensors and bridge.
The cloth and rigid builds do not reach real time at 200 Hz, so a wall that
moves when a car hits it is not available at that rate on this hardware.

A bigger GPU does not change this. Sampled during the 81 m cloth run: GPU
utilisation 1-17%, 4.4 of 16 GB used, CPU ~600% of a possible 3200%. Nothing is
saturated. GPU dynamics was checked and is genuinely on -- enableGPUDynamics
True, broadphaseType GPU, TGS solver, GPU contact buffers allocated -- so this
is not a misconfiguration hiding a win. The step time is set by serial CPU-side
work, which more GPU or more cores will not shorten.

RL is a different question -- training wants throughput, not wall-clock, so a
build slower than real time is fine there.

## What actually costs what

Ablation at 81 m, one thing changed at a time:

| case | physics | render | note |
|---|---|---|---|
| full | 25.86 | 15.47 | |
| hoop visuals off | 26.80 | **30.74** | removing 1.56M triangles made render *slower* |
| capsules 16 -> 8 | 15.99 | 13.57 | biggest single win |
| n_circ 40 -> 20 | 17.31 | 16.94 | |
| no fabric at all | 17.44 | 28.91 | hoops alone cost twice the deformable |

Two results here contradict the obvious guess:

- **Triangle count is not render cost.** Dropping the torus visuals removes
  1.56M triangles but makes 26,064 capsule prims visible instead of 1,629
  merged meshes, and render time doubles. Prim count dominates.
- **The fabric is not the expensive part.** Hoops alone are 17.4 ms against the
  deformable's 8.4.

## The hoop collider trade

One convex disc per hoop instead of 16 capsules: physics 25.9 -> 19.4 ms, hoops
in isolation 17.4 -> 4.6, build 54 -> 10 s. But it cannot be used, because the
auto-attachment creates **one seam per collision shape**:

| shapes/hoop | seams/hoop |
|---|---|
| 16 | 17.0 |
| 8 | 9.0 |
| 4 | 5.0 |
| 2 | 3.0 |
| 1 | 2.0 |

Exactly `shapes + 1`, measured. So collapsing the collider collapses the sewing
by the same factor; the fabric is held at two points per hoop, cannot resist a
hoop rotating, and the duct goes over like dominoes under a light push. The
disc's win and its failure are the same fact.

Mass and inertia were also wrong for the disc and both had to be fixed (6.0x
too heavy from keeping the density on a solid; 1.77x too easy to tip from
letting PhysX derive the tensor from the shape). Neither was the cause.
