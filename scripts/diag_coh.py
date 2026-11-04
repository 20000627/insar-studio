import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube, invert_arrays
from insar_studio.core.utils import downsample_real
root=Path("workspace/prototypes/p1");stack=load_stack(root);meta=stack.meta;truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json"));pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"]);disp=truth["disp_los_mm"];dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
td0=disp-disp[0];td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
t=np.asarray(meta["sampling_days"],float);t-=t.mean();tv=(t[:,None]*td.reshape(td.shape[0],-1)).sum(0)/(t**2).sum()*365.25;tv=tv.reshape(td.shape[1:])
for demfix in (0,1):
    ph=phase.copy()
    for idx,p in enumerate(kept):
        r,s=p["ref"],p["sec"]
        ph[idx]-=downsample_real(k*(atm[s]-atm[r])/1000,m)
        if demfix: ph[idx]-=downsample_real(k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))),m)
    res=invert_arrays(root,ph,coh,mask,kept,meta,{"estimate_dem_error":False,"aps_iterations":0,"regularization":0.05})
    vel=res["velocity_mm_yr"];valid=res["valid"];dd=res["disp_mm"];mc=np.nanmean(coh,axis=0)
    for th in (0.0,0.5,0.7,0.8):
        v=valid&(mc>=th)
        dr=np.sqrt(np.nanmean((dd-td[:,:dd.shape[1],:dd.shape[2]])[:,v]**2));vr=np.sqrt(np.nanmean((vel-tv[:vel.shape[0],:vel.shape[1]])[v]**2))
        print("demfix",demfix,"coh>",th,"n",v.sum(),"dRMSE",round(dr,2),"vRMSE",round(vr,2))
    for tt in (6,12,18,23):
        e=dd[tt]-td[tt][:dd.shape[1],:dd.shape[2]]
        print(" t",tt,"rmse",round(np.sqrt(np.nanmean(e[valid]**2)),2),"med",round(np.nanmedian(e[valid]),2),"p95",round(np.nanpercentile(np.abs(e[valid]),95),2))
