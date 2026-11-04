import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube
from insar_studio.core.utils import downsample_real
root=Path("workspace/prototypes/p1");stack=load_stack(root);meta=stack.meta;truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json"));pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
m=2;k=4*np.pi/meta["wavelength_m"];atm=truth["atmosphere_mm"];bp=np.asarray(truth["bperp"]);disp=truth["disp_los_mm"];dem=(truth["dem_true"]-truth["dem_ref"]).astype(np.float32)
td0=disp-disp[0];td=np.stack([downsample_real(td0[i],m) for i in range(td0.shape[0])])
t=np.asarray(meta["sampling_days"],float);t-=t.mean();tv=(t[:,None]*td.reshape(td.shape[0],-1)).sum(0)/(t**2).sum()*365.25;tv=tv.reshape(td.shape[1:])
ph=phase.copy()
for idx,p in enumerate(kept):
    r,s=p["ref"],p["sec"];ph[idx]-=downsample_real(k*(atm[s]-atm[r])/1000,m)
p=len(kept);n=len(meta["dates"]);A=np.zeros((p,n))
for i,q in enumerate(kept): A[i,q["ref"]]-=1;A[i,q["sec"]]+=1
b=np.asarray([k*(bp[q["sec"]]-bp[q["ref"]])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))) for q in kept])
D=np.zeros((n-2,n))
for i in range(n-2): D[i,i:i+3]=(1,-2,1)
Y=ph.reshape(p,-1);maskf=mask.reshape(p,-1);coverage=maskf.mean(0);valid=(coverage>=.45).reshape(mask.shape[1:])
tdem=downsample_real(dem,m);rows,cols=mask.shape[1:]
for lam in (0.02,0.05,0.08,0.12,0.2,0.4,0.8):
    G=np.zeros((p+n-2,n+1));G[:p,:n]=A;G[:p,n]=b;G[p:,:n]=lam*D
    coef=np.linalg.pinv(G)@np.vstack([np.where(maskf,Y,0),np.zeros((n-2,Y.shape[1]))])
    X=coef[:n];eps=coef[n]
    dd=(X*meta["wavelength_m"]/(4*np.pi)*1000).reshape(n,rows,cols);dd-=dd[0]
    vr=np.sqrt(np.nanmean((tv[:rows,:cols]-dd)[:,valid]**2))
    dr=np.sqrt(np.nanmean((td[:,:rows,:cols]-dd)[:,valid]**2))
    de=eps.reshape(rows,cols)
    print("lam",lam,"vRMSE",round(vr,2),"dRMSE",round(dr,2),"epsMed",round(np.nanmedian(de[valid]),1),"epsCorr",round(np.corrcoef(de[valid],tdem[:rows,:cols][valid])[0,1],2))
