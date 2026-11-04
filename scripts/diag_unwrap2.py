import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_complex, wrap_phase, fill_nan_nearest
from insar_studio.core.unwrapping import unwrap_quality_guided
from insar_studio.core.filters import boxcar_filter, gaussian_filter_complex, goldstein_filter
from skimage.restoration import unwrap_phase

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
p=json.load(open(root/"interferograms/00_01/meta.json")); m=2
coreg=np.load(root/"data/processed/slc_coreg.npy",mmap_mode="r")
ref=np.asarray(coreg[0])[:220,:260]; sec=np.asarray(coreg[1])[:220,:260]
ifg=downsample_complex(sec*np.conj(ref),2)
dem=downsample_complex(np.asarray(stack.dem,dtype=np.complex64)[:220,:260],2).real
k=4*np.pi/meta["wavelength_m"]; b=meta["bperp_m"][1]-meta["bperp_m"][0]
ifg=ifg*np.exp(-1j*k*b*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))))
coh=np.load(root/"interferograms/00_01/coherence.npy"); mask=np.load(root/"interferograms/00_01/mask.npy")
t=1;r=0
exp=k*((truth["disp_los_mm"][t]-truth["disp_los_mm"][r])+(truth["atmosphere_mm"][t]-truth["atmosphere_mm"][r]))/1000
bperp=np.asarray(truth["bperp"]); exp+=k*(bperp[t]-bperp[r])*(truth["dem_true"]-truth["dem_ref"])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
exp_m=downsample_complex(exp.astype(np.complex64)[:220,:260],2).real
print("raw wrap err std", np.nanstd(wrap_phase(np.angle(ifg)-exp_m)[mask]))
def score(name, phase):
    u,info=unwrap_quality_guided(phase,coh,mask)
    d=u-exp_m; d-=np.nanmedian(d[mask])
    print(name,"quality std",np.nanstd(d[mask]),"p95",np.nanpercentile(np.abs(d[mask]),95),"frac",np.mean(np.abs(d[mask])>np.pi),"comp",info["components"])
    f=fill_nan_nearest(np.where(mask,phase,np.nan))
    us=unwrap_phase(f.astype(np.float32)); d=us-exp_m; d-=np.nanmedian(d[mask])
    print(name,"skimage std",np.nanstd(d[mask]),"p95",np.nanpercentile(np.abs(d[mask]),95),"frac",np.mean(np.abs(d[mask])>np.pi))
for name, ph in [
    ("raw",np.angle(ifg)),
    ("box3",np.angle(boxcar_filter(ifg,3))),
    ("box5",np.angle(boxcar_filter(ifg,5))),
    ("box9",np.angle(boxcar_filter(ifg,9))),
    ("gauss1",np.angle(gaussian_filter_complex(ifg,1))),
    ("gauss2",np.angle(gaussian_filter_complex(ifg,2))),
    ("gold",np.angle(goldstein_filter(ifg,0.65,32))),
]:
    score(name,ph.astype(np.float32))
