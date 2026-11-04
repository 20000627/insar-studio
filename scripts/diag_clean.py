import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import invert_arrays
from insar_studio.core.utils import downsample_real, downsample_complex

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json")); pairs=net["pairs"]
m=2; k=4*np.pi/meta["wavelength_m"]; bp=np.asarray(truth["bperp"]); dem=truth["dem_true"]-truth["dem_ref"]
def cube(use_atm=True, use_dem=True):
    arr=[]
    for p in pairs:
        t=p["sec"]; r=p["ref"]
        e=k*(truth["disp_los_mm"][t]-truth["disp_los_mm"][r])/1000
        if use_atm: e=e+k*(truth["atmosphere_mm"][t]-truth["atmosphere_mm"][r])/1000
        if use_dem: e=e+k*(bp[t]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
        arr.append(downsample_complex(e.astype(np.complex64)[:220,:260],m).real.astype(np.float32))
    return np.stack(arr)
tv=downsample_real(truth["velocity_mm_yr"],m)
td0=truth["disp_los_mm"]-truth["disp_los_mm"][0]
td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
for atm in [False,True]:
  for dem in [False,True]:
    ph=cube(atm,dem); mask=np.ones_like(ph,dtype=bool); coh=np.ones_like(ph)*0.95
    for estdem in [False,True]:
      res=invert_arrays(root,ph,coh,mask,pairs,meta,{"estimate_dem_error":estdem,"aps_iterations":2 if atm else 0,"aps_sigma_px":10,"regularization":0.05})
      vel=res["velocity_mm_yr"]; valid=res["valid"]; disp=res["disp_mm"]
      vr=np.sqrt(np.nanmean((vel-tv[:vel.shape[0],:vel.shape[1]])[valid]**2)); dr=np.sqrt(np.nanmean((disp-td)[:,valid]**2))
      print("atm",atm,"dem",dem,"estdem",estdem,"vRMSE",round(vr,2),"dRMSE",round(dr,2),"vmed",round(np.nanmedian(vel[valid]),2),"demmed",round(np.nanmedian(res["dem_error_m"][valid]),1))
