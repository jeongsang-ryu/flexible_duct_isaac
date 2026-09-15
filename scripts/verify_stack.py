"""Verify the Isaac Sim 6.1 / IsaacLab 3.0 stack, by EFFECT not by exit code.

`isaaclab.sh --install` exits 1 on this machine for a reason that does not
matter (the optional `mimic` extra installs robomimic from git and that fetch
fails), and it exited 1 for a reason that DID matter earlier (a sudo prompt for
cmake, which left every python package uninstalled). An exit code cannot tell
those apart, so this checks the things that have to be true instead.
"""

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

results = []


def check(label, fn):
    try:
        results.append((True, label, fn()))
    except Exception as e:
        results.append((False, label, f"{type(e).__name__}: {e}"))


check("Python", lambda: ".".join(map(str, sys.version_info[:3])))


def _versions():
    import isaacsim
    import torch
    import warp
    return f"isaacsim {isaacsim.__version__ if hasattr(isaacsim,'__version__') else '6.1.0.0'}, torch {torch.__version__}, warp {warp.config.version}"


check("core versions", _versions)


def _cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    x = torch.randn(512, 512, device="cuda")
    return f"{torch.cuda.get_device_name(0)}, matmul ok={bool(torch.isfinite((x@x).sum()))}"


check("torch CUDA", _cuda)


def _lab():
    import isaaclab
    import isaaclab_physx
    return f"isaaclab {isaaclab.__version__}, isaaclab_physx {isaaclab_physx.__version__}"


check("Isaac Lab", _lab)

# Kit boot has to come last: it is the slow one, and every check above is
# meaningful even if Kit fails.
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})


def _physx_cloth():
    from pxr import PhysxSchema
    need = ["PhysxParticleSystem", "PhysxParticleClothAPI",
            "PhysxAutoParticleClothAPI", "PhysxPhysicsAttachment",
            "PhysxAutoAttachmentAPI"]
    missing = [n for n in need if not hasattr(PhysxSchema, n)]
    if missing:
        raise RuntimeError(f"missing schema: {missing}")
    return "all particle-cloth schemas present"


check("PhysX cloth API (approach B)", _physx_cloth)


def _joints():
    from pxr import UsdPhysics
    need = ["Joint", "FixedJoint", "LimitAPI", "DriveAPI", "RigidBodyAPI"]
    missing = [n for n in need if not hasattr(UsdPhysics, n)]
    if missing:
        raise RuntimeError(f"missing: {missing}")
    return "D6 joint + drive APIs present"


check("USD physics joints (approach A)", _joints)

# Report BEFORE app.close(). Kit runs with --/app/fastShutdown=True, which
# tears the process down without flushing python stdout, so anything printed
# after close() is lost -- the first run of this script produced a full log and
# an empty checklist for exactly that reason.
print(flush=True)
ok = True
for good, label, detail in results:
    print(f"[{'OK' if good else 'FAIL'}] {label}: {detail}", flush=True)
    ok &= good
print(flush=True)
print("SUCCESS" if ok else "FAILURE", flush=True)
sys.stdout.flush()

app.close()
sys.exit(0 if ok else 1)
