"""Launch the duct in Isaac Sim, approach A or B, with mouse dragging enabled.

    python scripts/run_duct.py --approach a
    python scripts/run_duct.py --approach b
    python scripts/run_duct.py --approach a --headless --steps 300   # CI check

MOUSE INTERACTION. Kit's own physics grab is what moves the duct: with the
sim playing, ctrl + left-drag on a body applies a drag force. Nothing custom is
needed for approach A, because the hoops are ordinary rigid bodies -- which is
half the reason A is the recommended arm. For B the hoops are still rigid so
they grab the same way; the fabric follows through its attachments.

The argument parser runs BEFORE AppLauncher because Kit consumes argv, and the
AppLauncher import has to happen before any isaaclab.* submodule import -- both
are documented traps in the lab's INSTALL.md.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser(description="Flexible duct demo (approach A or B)")
parser.add_argument("--approach", choices=["a", "b"], default="a")
parser.add_argument("--length", type=float, default=2.0, help="duct length [m]")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=0,
                    help="if >0, run this many steps then exit (non-interactive check)")
parser.add_argument("--fix-last", action="store_true", help="anchor the far end")
parser.add_argument("--fix-first", action="store_true", help="anchor the near end")
parser.add_argument("--layout", choices=["floor", "hang"], default="floor",
                    help="floor: lies on the ground, both ends free (default); "
                         "hang: suspended from z=0")
args = parser.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

# --- only now may anything USD / physics be imported ---
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from duct_sim.spec import DuctSpec  # noqa: E402


def main() -> int:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    light = UsdLux.DistantLight.Define(stage, "/World/light")
    light.CreateIntensityAttr(2500.0)

    spec = DuctSpec(length=args.length, layout=args.layout,
                    fix_first_ring=args.fix_first, fix_last_ring=args.fix_last)

    # Ground: at z=0 for the floor layout (the duct rests on it), or well below
    # the free end when hanging so it is only a backstop.
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(ground)
    ground_top = 0.0 if spec.layout == "floor" else -spec.length - 0.6
    gx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ground_top - 0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(8.0, 8.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    print(f"[duct] approach={args.approach.upper()}  {spec.describe()}")

    skin = None
    if args.approach == "a":
        from duct_sim.build_a import build_duct_a
        from duct_sim.skin import DuctSkin
        info = build_duct_a(stage, "/World/DuctA", spec)
        skin = DuctSkin(stage, "/World/DuctA/fabric_visual", info["rings"], spec)
        print(f"[duct] A: {len(info['rings'])} rigid hoops, "
              f"{len(info['rings']) - 1} D6 joints, fabric = driven skin")
    else:
        from duct_sim.build_b import build_duct_b
        info = build_duct_b(stage, "/World/DuctB", spec)
        print(f"[duct] B: {len(info['rings'])} rigid hoops, "
              f"cloth {info['resolution'][0]}x{info['resolution'][1] + 1} "
              f"= {info['particle_count']} particles")

    from isaacsim.core.api import SimulationContext  # noqa: E402
    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()

    if args.steps > 0:
        for i in range(args.steps):
            sim.step(render=not args.headless)
            if skin is not None:
                skin.update()
        print(f"[duct] completed {args.steps} steps without error")
    else:
        print("[duct] interactive. ctrl + left-drag a hoop to pull the duct around.")
        while simulation_app.is_running():
            sim.step(render=True)
            if skin is not None:
                skin.update()

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
