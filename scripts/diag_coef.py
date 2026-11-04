import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube
from insar_studio.core.utils import downsample_real
root=Path("workspace/prototypes/p1")
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json")); pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"]);disp=truth["disp_los_mm"]
dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
rows,cols=phase.shape[1:]; yy,xx=np.mgrid[0:rows,0:cols]; xn=xx/(cols-1)-0.5; yn=yy/(rows-1)-0.5
cs=[];corrs=[]
for idx,p in enumerate(kept[:10]):
    r,s=p["ref"],p["sec"]
    mod=downsample_real(k*(disp[s]-disp[r]+atm[s]-atm[r])/1000,m)
    dterm=downsample_real(k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))),m)
    y=phase[idx]-mod
    good=mask[idx]&(coh[idx]>0.7)
    A=np.column_stack([dterm[good],xn[good],yn[good],np.ones(good.sum())])
    coef,*_=np.linalg.lstsq(A,y[good],rcond=None)
    cs.append(coef[0]);corrs.append(np.corrcoef(dterm[good],y[good])[0,1])
print("coef c on dem term per pair",np.round(cs,2))
print("corr dterm vs residual",np.round(corrs,2))
# also estimate dem from all pairs with per-pair plane + per-date phase (free) -> collinear, use known disp+atm to remove
A=[];Y=[];W=[]
for idx,p in enumerate(kept):
    r,s=p["ref"],p["sec"]
    mod=downsample_real(k*(disp[s]-disp[r]+atm[s]-atm[r])/1000,m)
    dterm=downsample_real(k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))),m)
    good=mask[idx]&(coh[idx]>0.7)
    y=phase[idx]-mod
    # per-pair plane absorbs ramp
    for a,b in zip([dterm[good],xn[good],yn[good],np.ones(good.sum())],[1]): pass
    A.append(np.column_stack([dterm[good],xn[good],yn[good],np.ones(good.sum())]));Y.append(y[good])
# separate per pair c
cs2=[]
for a,y in zip(A,Y):
    c,*_=np.linalg.lstsq(a,y,rcond=None);cs2.append(c[0])
print("pairwise c median",np.median(cs2),"mean",np.mean(cs2),"range",np.min(cs2),np.max(cs2))
