import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insar_studio.core.synthetic import generate_demo_dataset
from insar_studio.core.io import load_stack
from insar_studio.core.network import build_network, persist_network
from insar_studio.core.coregistration import coregister_stack
from insar_studio.core.interferometry import process_interferograms

root = Path(__file__).resolve().parents[1] / "workspace" / "prototypes" / "p1"
root.mkdir(parents=True, exist_ok=True)
t0 = time.time()
meta = generate_demo_dataset(root)
print("demo", time.time() - t0, meta["rows"], meta["cols"])
stack = load_stack(root)
pairs, stats = build_network(stack.dates, stack.meta["bperp_m"], "sbas", {"max_temporal_baseline": 72, "max_perpendicular_baseline": 180})
print("pairs", len(pairs), stats)
persist_network(root, pairs, stats)
t0 = time.time()
coreg = coregister_stack(root, {"model": "affine"}, progress=lambda f, m: None)
print("coreg", time.time() - t0, coreg["coherence_before"], coreg["coherence_after"])
print("shifts", coreg["shifts"][:4])
t0 = time.time()
summary = process_interferograms(root, {"multilook": 2, "filter": "goldstein"}, progress=lambda f, m: print(f"{f:.2f} {m}") if int(f*100) % 20 == 0 else None)
print("ifg", time.time() - t0, summary["mean_coherence"], summary["mean_valid_fraction"], summary["closure"])
