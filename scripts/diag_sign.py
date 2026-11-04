import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube, invert_arrays
from insar_studio.core.utils import downsample_real, downsample_complex
root=Path("workspace/prototypes/p1")
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json")); pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"])
td0=truth["disp_los_mm"]-truth["disp_los_mm"][0]; td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
t=np.asarray(meta["sampling_days"],float);t-=t.mean();tv=(t[:,None]*td.reshape(td.shape[0],-1)).sum(0)/(t**2).sum()*365.25;tv=tv.reshape(td.shape[1:])
dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
for sign in (-1,0,1):
    ph=phase.copy()
    for idx,p in enumerate(kept):
        r,s=p["ref"],p["sec"]
        d=downsample_complex((k*(atm[s]-atm[r])/1000).astype(np.complex64)[:220,:260],m).real
        ph[idx]-=d
        if sign:
            d=downsample_complex((k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))).astype(np.complex64)[:220,:260],m).real
            ph[idx]+=sign*d
    res=invert_arrays(root,ph,coh,mask,kept,meta,{"estimate_dem_error":False,"aps_iterations":0,"regularization":0.05})
    vel=res["velocity_mm_yr"];valid=res["valid"];disp=res["disp_mm"]
    vr=np.sqrt(np.nanmean((vel-tv[:vel.shape[0],:vel.shape[1]])[valid]**2));dr=np.sqrt(np.nanmean((disp-td[:,:disp.shape[1],:disp.shape[2]])[:,valid]**2))
    print("sign",sign,"vRMSE",round(vr,2),"dRMSE",round(dr,2),"vmed",round(np.nanmedian(vel[valid]),2))
