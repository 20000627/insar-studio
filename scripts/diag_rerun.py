import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.interferometry import process_interferograms
from insar_studio.core.inversion import run_inversion
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_real

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
t0=time.time()
s=process_interferograms(root, {"multilook":2,"filter":"boxcar","boxcar_size":3}, save_figures=False)
print("ifg",time.time()-t0,s["mean_coherence"],s["closure"])
t0=time.time()
inv=run_inversion(root,"sbas",{"multilook":2,"aps_method":"spatiotemporal","aps_iterations":2,"filter":"boxcar","boxcar_size":3})
print("inv",time.time()-t0,json.dumps(inv,ensure_ascii=False)[:500])
truth=np.load(root/"data/stack/truth.npz"); m=2
vel=np.load(root/"products/arrays/velocity_mm_yr.npy"); valid=np.load(root/"products/arrays/valid.npy")
tv=downsample_real(truth["velocity_mm_yr"],m)[:vel.shape[0],:vel.shape[1]]
d=vel-tv
print("vel rmse",np.sqrt(np.nanmean(d[valid]**2)),"bias",np.nanmean(d[valid]),"vel range",np.nanmin(vel[valid]),np.nanmax(vel[valid]),"truth",np.nanmin(tv[valid]),np.nanmax(tv[valid]))
disp=np.load(root/"products/arrays/disp_ts_mm.npy")
td0=truth["disp_los_mm"]-truth["disp_los_mm"][0]
td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
td=td[:,:disp.shape[1],:disp.shape[2]]
for t in [6,12,18,23]:
    dd=disp[t]-td[t]
    print("t",t,"rmse",np.sqrt(np.nanmean(dd[valid]**2)),"bias",np.nanmean(dd[valid]),"p95",np.nanpercentile(np.abs(dd[valid]),95))
dem=np.load(root/"products/arrays/dem_error_m.npy"); tdem=downsample_real(truth["dem_true"]-truth["dem_ref"],m)[:dem.shape[0],:dem.shape[1]]
print("dem",np.nanmedian(dem[valid]),np.nanmedian(tdem[valid]),"corr",np.corrcoef(dem[valid],tdem[valid])[0,1])
print("tcoh",np.nanmedian(np.load(root/"products/arrays/temporal_coherence.npy")[valid]))
