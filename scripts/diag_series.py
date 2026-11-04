import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube, invert_arrays
from insar_studio.core.utils import downsample_real, downsample_complex

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json")); pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
m=2; k=4*np.pi/meta["wavelength_m"]; atm=truth["atmosphere_mm"]
ph=phase.copy()
for idx,p in enumerate(kept):
    ph[idx]-=downsample_complex((k*(atm[p["sec"]]-atm[p["ref"]])/1000).astype(np.complex64)[:220,:260],m).real
res=invert_arrays(root,ph,coh,mask,kept,meta,{"estimate_dem_error":True,"aps_iterations":0,"regularization":0.05})
disp=res["disp_mm"]; valid=res["valid"]
td0=truth["disp_los_mm"]-truth["disp_los_mm"][0]
td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
for pix in [(43,70),(55,65),(60,70),(50,60)]:
    if not valid[pix]: print(pix,"invalid"); continue
    d=disp[:,pix[0],pix[1]]; t=td[:,pix[0],pix[1]]-td[0,pix[0],pix[1]]
    print("pix",pix,"est",np.round(d,1))
    print("      truth",np.round(t,1))
    print("      diff",np.round(d-t,1),"std",np.std(d-t))
    # fit linear + annual to both
    days=meta["sampling_days"]; days=np.asarray(days)/365.25
    B=np.column_stack([days, np.sin(2*np.pi*days), np.cos(2*np.pi*days), np.ones_like(days)])
    ce=np.linalg.lstsq(B,d,rcond=None)[0]; ct=np.linalg.lstsq(B,t,rcond=None)[0]
    print("      vel est/truth",ce[0],ct[0])
