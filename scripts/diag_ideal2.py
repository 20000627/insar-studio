import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import invert_arrays
from insar_studio.core.utils import downsample_real
root=Path("workspace/prototypes/p1");stack=load_stack(root);meta=stack.meta;truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json"));pairs=net["pairs"]
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"]);disp=truth["disp_los_mm"];dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
td0=disp-disp[0];td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
t=np.asarray(meta["sampling_days"],float);t-=t.mean();tv=(t[:,None]*td.reshape(td.shape[0],-1)).sum(0)/(t**2).sum()*365.25;tv=tv.reshape(td.shape[1:])
rows,cols=td.shape[1:];yy,xx=np.mgrid[0:rows,0:cols];xn=xx/(cols-1)-.5;yn=yy/(rows-1)-.5
cube=[]
for p in pairs:
    r,s=p["ref"],p["sec"]
    e=downsample_real(k*(disp[s]-disp[r]+atm[s]-atm[r])/1000+k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))),m)
    cube.append(e.astype(np.float32))
ideal=np.stack(cube);mask=np.ones_like(ideal,dtype=bool);coh=np.ones_like(ideal)*.9
for aps in (0,1,2):
  for est in (0,1):
    res=invert_arrays(root,ideal,coh,mask,pairs,meta,{"estimate_dem_error":est,"aps_iterations":aps,"regularization":0.05})
    vel=res["velocity_mm_yr"];v=res["valid"];dd=res["disp_mm"]
    print("ideal aps",aps,"est",est,"dRMSE",round(np.sqrt(np.nanmean((dd-td[:,:dd.shape[1],:dd.shape[2]])[:,v]**2)),3),"vRMSE",round(np.sqrt(np.nanmean((vel-tv[:vel.shape[0],:vel.shape[1]])[v]**2)),2),"dem",round(np.nanmedian(res["dem_error_m"][v]),1))
