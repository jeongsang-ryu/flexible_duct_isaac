"""Approach B: PhysX hoops + a Newton-simulated fabric sleeve.

    python scripts/run_duct_b.py --headless --steps 200      # check
    python scripts/run_duct_b.py                              # interactive
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=float, default=2.0)
parser.add_argument("--layout", choices=["floor", "hang"], default="floor")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--drape", action="store_true", help="start draped over a bar")
parser.add_argument("--bar-height", type=float, default=2.0)
args = parser.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt  # noqa: E402

from duct_sim.build_a import build_duct_a  # noqa: E402
from duct_sim.build_b_newton import NewtonFabric  # noqa: E402
from duct_sim.spec import DuctSpec  # noqa: E402


def main() -> int:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
    UsdLux.DistantLight.Define(stage, "/World/key").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(600.0)

    g = UsdGeom.Cube.Define(stage, "/World/ground")
    g.CreateSizeAttr(1.0)
    ggx = UsdGeom.Xformable(g)
    ggx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ggx.AddScaleOp().Set(Gf.Vec3f(12, 12, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    spec = DuctSpec(length=args.length, layout=args.layout)
    info = build_duct_a(stage, "/World/DuctB", spec,
                        bend_limit_deg=30.0, bend_stiffness=1.0, bend_damping=0.3)
    rings = info["rings"]
    print(f"[B] {spec.describe()}", flush=True)

    if args.drape:
        bar = UsdGeom.Capsule.Define(stage, "/World/bar")
        bar.CreateAxisAttr("Y")
        bar.CreateRadiusAttr(0.04)
        bar.CreateHeightAttr(2.0)
        UsdGeom.Xformable(bar).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, args.bar_height))
        UsdPhysics.CollisionAPI.Apply(bar.GetPrim())
        lift = args.bar_height + 0.04 + spec.radius + spec.ring_thickness
        for i, rp in enumerate(rings):
            x, _, _ = spec.ring_center(i)
            UsdGeom.Xformable(stage.GetPrimAtPath(rp)).GetOrderedXformOps()[0].Set(
                Gf.Vec3d(x, 0.0, lift))

    cache = UsdGeom.XformCache()

    def ring_world(i):
        cache.Clear()
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(rings[i]))
        t = m.ExtractTranslation()
        r = m.ExtractRotationMatrix()
        # the fabric's local circle is built in YZ, so hand back the basis that
        # maps that circle onto this hoop
        return (np.array([t[0], t[1], t[2]]),
                np.array([[r[0][0], r[0][1], r[0][2]],
                          [r[1][0], r[1][1], r[1][2]],
                          [r[2][0], r[2][1], r[2][2]]]))

    fabric = NewtonFabric(spec, ring_world)
    print(f"[B] Newton cloth: {fabric.n_particles} particles, "
          f"{len(fabric.faces)} triangles, {len(fabric.seam_loops)} sewn seams",
          flush=True)
    for lim in NewtonFabric.known_limitations():
        print(f"[B]   limitation: {lim}", flush=True)

    mesh = UsdGeom.Mesh.Define(stage, "/World/DuctB/fabric_sim")
    counts = [3] * len(fabric.faces)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(fabric.faces.flatten().tolist()))
    mesh.CreateDoubleSidedAttr(True)
    pts_attr = mesh.CreatePointsAttr()

    from isaacsim.core.api import SimulationContext  # noqa: E402

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()
    dt = 1.0 / 60.0

    def tick(render: bool):
        sim.step(render=render)
        fabric.step(dt)
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(fabric.positions().astype(np.float32)))

    if args.steps > 0:
        for _ in range(args.steps):
            tick(not args.headless)
        p = fabric.positions()
        print(f"[B] {args.steps} steps ok. fabric z range "
              f"{p[:,2].min():.3f}..{p[:,2].max():.3f}, finite={np.isfinite(p).all()}",
              flush=True)
    else:
        print("[B] interactive: ctrl + left-drag a hoop.", flush=True)
        while simulation_app.is_running():
            tick(True)

    sys.stdout.flush()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
