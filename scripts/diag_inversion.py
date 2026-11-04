import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.inversion import run_inversion
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_real, downsample_complex

root = Path(__file__).resolve().parents[1] / "workspace" / "prototypes" / "p1"
t0=time.time()
summary = run_inversion(root, "sbas", {"multilook": 2, "aps_method": "spatiotemporal", "aps_iterations": 2}, progress=lambda f,m: print(f"{f:.2f} {m}") if int(f*100)%20==0 else None)
print("time", time.time()-t0)
print(json.dumps(summary, ensure_ascii=False, indent=2)[:1200])
truth=np.load(root/"data/stack/truth.npz")
meta=load_stack(root).meta
m=2
vel_truth=downsample_real(truth["velocity_mm_yr"],m)
# crop to inversion shape
arr=np.load(root/"products/arrays/velocity_mm_yr.npy")
vel_truth=vel_truth[:arr.shape[0],:arr.shape[1]]
valid=np.load(root/"products/arrays/valid.npy")
diff=arr-vel_truth
print("velocity valid", valid.sum(), "rmse", np.sqrt(np.nanmean(diff[valid]**2)), "bias", np.nanmean(diff[valid]))
print("vel range", np.nanmin(arr), np.nanmax(arr), "truth", np.nanmin(vel_truth), np.nanmax(vel_truth))
disp=np.load(root/"products/arrays/disp_ts_mm.npy")
truth_disp=truth["disp_los_mm"]-truth["disp_los_mm"][0]
truth_disp=downsample_real(truth_disp,m)
truth_disp=truth_disp[:,:disp.shape[1],:disp.shape[2]]
for t in [6,12,18,23]:
    d=disp[t]-truth_disp[t]
    print("t",t,"rmse",np.sqrt(np.nanmean(d[valid]**2)),"bias",np.nanmean(d[valid]),"p95",np.nanpercentile(np.abs(d[valid]),95))
dem=np.load(root/"products/arrays/dem_error_m.npy")
dem_truth=downsample_real(truth["dem_true"]-truth["dem_ref"],m)[:dem.shape[0],:dem.shape[1]]
print("dem err", np.nanmedian(dem[valid]), np.nanmedian(dem_truth[valid]), "corr", np.corrcoef(dem[valid],dem_truth[valid])[0,1])
print("tcoh", np.nanmedian(np.load(root/"products/arrays/temporal_coherence.npy")[valid]))
