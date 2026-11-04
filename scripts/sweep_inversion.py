import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.inversion import run_inversion
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_real

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
truth=np.load(root/"data/stack/truth.npz"); m=2
tv=downsample_real(truth["velocity_mm_yr"],m)
td0=truth["disp_los_mm"]-truth["disp_los_mm"][0]
td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
tdem=downsample_real(truth["dem_true"]-truth["dem_ref"],m)
for dem in [False,True]:
  for aps in [0,1,2]:
    for reg in [0.03,0.08]:
      s=run_inversion(root,"sbas",{"multilook":2,"estimate_dem_error":dem,"aps_iterations":aps,"regularization":reg})
      vel=np.load(root/"products/arrays/velocity_mm_yr.npy"); valid=np.load(root/"products/arrays/valid.npy")
      disp=np.load(root/"products/arrays/disp_ts_mm.npy"); de=np.load(root/"products/arrays/dem_error_m.npy")
      v=tv[:vel.shape[0],:vel.shape[1]]; d=td[:,:disp.shape[1],:disp.shape[2]]
      vr=np.sqrt(np.nanmean((vel-v)[valid]**2)); dr=np.sqrt(np.nanmean((disp-d)[:,valid]**2))
      corr=np.corrcoef(de[valid],tdem[:de.shape[0],:de.shape[1]][valid])[0,1] if np.isfinite(de[valid]).any() else 0
      print(f"dem={dem} aps={aps} reg={reg} vRMSE={vr:.2f} dRMSE={dr:.2f} demCorr={corr:.2f} vmed={np.nanmedian(vel[valid]):.2f} tcoh={np.nanmedian(np.load(root/'products/arrays/temporal_coherence.npy')[valid]):.2f}")
