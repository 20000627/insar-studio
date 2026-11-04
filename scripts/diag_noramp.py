import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube, invert_arrays
from insar_studio.core.utils import downsample_real
root=Path("workspace/prototypes/p1")
stack=load_stack(root);meta=stack.meta;truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json"));pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
rows,cols=phase.shape[1:];yy,xx=np.mgrid[0:rows,0:cols];xn=xx/(cols-1)-.5;yn=yy/(rows-1)-.5
# add removed plane back
for idx,p in enumerate(kept):
    meta_p=json.load(open(root/"interferograms"/p["id"]/"meta.json"))
    if meta_p.get("ramp",{}).get("removed"):
        phase[idx]+=meta_p["ramp"]["slope_x_rad"]*xn+meta_p["ramp"]["slope_y_rad"]*yn
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"]);disp=truth["disp_los_mm"];dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
td0=disp-disp[0];td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
t=np.asarray(meta["sampling_days"],float);t-=t.mean();tv=(t[:,None]*td.reshape(td.shape[0],-1)).sum(0)/(t**2).sum()*365.25;tv=tv.reshape(td.shape[1:])
def corr(use_atm,use_dem):
    ph=phase.copy()
    for idx,p in enumerate(kept):
        r,s=p["ref"],p["sec"]
        if use_atm: ph[idx]-=downsample_real(k*(atm[s]-atm[r])/1000,m)
        if use_dem: ph[idx]-=downsample_real(k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))),m)
    return ph
for atmf in (0,1):
  for demf in (0,1):
    for est in (0,1):
      for aps in (0,2):
        ph=corr(atmf,demf)
        res=invert_arrays(root,ph,coh,mask,kept,meta,{"estimate_dem_error":est,"aps_iterations":aps,"regularization":0.05})
        vel=res["velocity_mm_yr"];valid=res["valid"];dd=res["disp_mm"]
        vr=np.sqrt(np.nanmean((vel-tv[:vel.shape[0],:vel.shape[1]])[valid]**2));dr=np.sqrt(np.nanmean((dd-td[:,:dd.shape[1],:dd.shape[2]])[:,valid]**2))
        print(f"atm={atmf} demfix={demf} est={est} aps={aps} vRMSE={vr:5.2f} dRMSE={dr:5.2f} vmed={np.nanmedian(vel[valid]):6.2f} demMed={np.nanmedian(res['dem_error_m'][valid]):6.1f}")
