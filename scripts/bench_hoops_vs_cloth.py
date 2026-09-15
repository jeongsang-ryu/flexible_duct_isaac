"""Separate hoop count from cloth resolution as cost drivers.

Every length comparison so far moved both at once: a longer duct has more hoops
AND more fabric vertices, so "time scales with hoops" was never actually shown.
This holds length fixed and varies one at a time.

  A  base
  B  cloth doubled   (n_circ x2, same hoops)
  C  hoops doubled   (spacing /2, same n_circ)
  D  both

A->B big and A->C small  =>  the fabric is the cost; drop n_circ.
A->C big and A->B small  =>  the hoops are;        widen the spacing.
Both similar             =>  length itself is the wall; freeze is the only out.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_approaches.py")
OUT = "/tmp/bench_2x2"
os.makedirs(OUT, exist_ok=True)

LENGTH = 6.0
CASES = [
    ("A base",          0.10, 28),
    ("B cloth x2",      0.10, 56),
    ("C hoops x2",      0.05, 28),
    ("D both x2",       0.05, 56),
]

rows = []
for name, spacing, n_circ in CASES:
    tag = name.split()[0]
    print(f"\n=== {name}: spacing {spacing} m, n_circ {n_circ} ===", flush=True)
    r = subprocess.run(
        [sys.executable, BENCH, "--mode", "cloth", "--length", str(LENGTH),
         "--spacing", str(spacing), "--n-circ", str(n_circ),
         "--warmup", "400", "--measure", "600", "--out", OUT],
        capture_output=True, text=True)
    path = os.path.join(OUT, "cloth.json")
    if not os.path.exists(path):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        rows.append((name, spacing, n_circ, None, None, None, None))
        continue
    d = json.load(open(path))
    os.rename(path, os.path.join(OUT, f"cloth_{tag}.json"))
    rows.append((name, spacing, n_circ, d.get("bodies"), d.get("seam_elements"),
                 d.get("step_ms"), d.get("gpu_mb")))
    print(f"  -> {d.get('step_ms')} ms/step, {d.get('bodies')} hoops, "
          f"{d.get('gpu_mb')} MiB", flush=True)

print("\n" + "=" * 78)
print(f"{'case':14s} {'spacing':>8s} {'n_circ':>7s} {'hoops':>7s} "
      f"{'seams':>7s} {'ms/step':>9s} {'GPU MiB':>9s}")
print("-" * 78)
base = None
for name, sp, nc, hoops, seams, ms, gpu in rows:
    if ms is None:
        print(f"{name:14s} {sp:8.2f} {nc:7d}   FAILED")
        continue
    if base is None:
        base = ms
    print(f"{name:14s} {sp:8.2f} {nc:7d} {hoops or 0:7d} {seams or 0:7d} "
          f"{ms:9.2f} {gpu or 0:9.0f}   ({ms / base:.2f}x base)")
print("=" * 78)
json.dump([dict(zip(("case", "spacing", "n_circ", "hoops", "seams",
                     "step_ms", "gpu_mb"), r)) for r in rows],
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"\nwrote {OUT}/summary.json")
