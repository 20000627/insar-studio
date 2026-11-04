import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube, _design
from insar_studio.core.utils import downsample_real

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
stack=load_stack(root); meta=stack.meta
net=json.load(open(root/"data/network.json")); pairs=net["pairs"]
phase,coh,mask,kept=load_pair_cube(root,pairs,corrected=False)
ref=json.load(open(root/"interferograms/reference.json"))
rr,cc=ref["row"],ref["col"]
print("ref",rr,cc,"value per pair minmax",np.nanmin(phase[:,rr,cc]),np.nanmax(phase[:,rr,cc]),"median",np.nanmedian(phase[:,rr,cc]))
print("ref mask",mask[:,rr,cc].mean())
print("phase global range",np.nanmin(phase),np.nanmax(phase),"median std",np.nanmedian(np.nanstd(phase,axis=(1,2))))
# pair phases at a pixel compared to truth
truth=np.load(root/"data/stack/truth.npz"); bp=np.asarray(truth["bperp"]); k=4*np.pi/meta["wavelength_m"]
pin=43*130+70
for idx in [0,1,10,40,79]:
    p=pairs[idx]; e=k*((truth["disp_los_mm"][p["sec"]]-truth["disp_los_mm"][p["ref"]])+(truth["atmosphere_mm"][p["sec"]]-truth["atmosphere_mm"][p["ref"]]))/1000
    e=e+k*(bp[p["sec"]]-bp[p["ref"]])*(truth["dem_true"]-truth["dem_ref"])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
    e=downsample_real(e,2)[rr,cc]
    print(idx,p["id"],"phase",phase[idx,rr,cc],"exp_centered",e-np.nanmedian(e),"coh",coh[idx,rr,cc])
# design condition
for reg in [0,0.08]:
    G,A,b,bs=_design(pairs,len(meta["dates"]),meta["bperp_m"],meta,reg)
    print("reg",reg,"G cond",np.linalg.cond(G),"b std",np.std(b),"b scale",bs,"A rank",np.linalg.matrix_rank(A))
    # solve one pixel
    y=phase[:,rr,cc].astype(float)
    x=np.linalg.pinv(G)@y
    print("x first",x[:6],"eps",x[-1]/bs,"resid",np.sqrt(np.mean((G@x-y)**2)))
# actual data at a stable pixel, print phases
for pix in [(rr,cc),(55,65),(60,70)]:
    y=phase[:,pix[0],pix[1]].astype(float)
    G,A,b,bs=_design(pairs,len(meta["dates"]),meta["bperp_m"],meta,0.08)
    x=np.linalg.pinv(G)@y
    print("pix",pix,"ts",np.round(x[:24]*4.426,1),"eps",x[-1]/bs,"resid",np.sqrt(np.mean((G@x-y)**2)))
