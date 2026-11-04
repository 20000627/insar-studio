import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube
from insar_studio.core.utils import downsample_real
root=Path("workspace/prototypes/p1")
stack=load_stack(root);meta=stack.meta;truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json"));pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"]);disp=truth["disp_los_mm"];dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
rows,cols=phase.shape[1:];yy,xx=np.mgrid[0:rows,0:cols];xn=xx/(cols-1)-0.5;yn=yy/(rows-1)-0.5
for idx in [0,1,20,50]:
    p=kept[idx];r,s=p["ref"],p["sec"]
    dmod=downsample_real(k*(disp[s]-disp[r])/1000,m)
    dt=downsample_real(k*(bp[s]-bp[r])*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))),m)
    matm=downsample_real(k*(atm[s]-atm[r])/1000,m)
    good=mask[idx]&(coh[idx]>0.7)
    for name,y in [("raw",phase[idx]),("noatm",phase[idx]-matm),("noatm+dem",phase[idx]-matm-dt)]:
        res=y-dmod
        A=np.column_stack([xn[good],yn[good],np.ones(good.sum())])
        c,*_=np.linalg.lstsq(A,res[good],rcond=None)
        rr=res[good]-A@c
        print(idx,name,"fitc",np.round(c,2),"std",round(np.std(rr),2),"med",round(np.median(rr),2))
