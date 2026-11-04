import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_complex, wrap_phase, fill_nan_nearest
from insar_studio.core.unwrapping import unwrap_quality_guided

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
pid="00_01"; p=json.load(open(root/f"interferograms/{pid}/meta.json"))
wrapped=np.load(root/f"interferograms/{pid}/wrapped.npy"); coh=np.load(root/f"interferograms/{pid}/coherence.npy"); mask=np.load(root/f"interferograms/{pid}/mask.npy")
t=p["sec"]; r=p["ref"]; k=4*np.pi/meta["wavelength_m"]
bperp=np.asarray(truth["bperp"])
exp=k*((truth["disp_los_mm"][t]-truth["disp_los_mm"][r])+(truth["atmosphere_mm"][t]-truth["atmosphere_mm"][r]))/1000
exp+=k*(bperp[t]-bperp[r])*(truth["dem_true"]-truth["dem_ref"])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
exp_m=downsample_complex(exp.astype(np.complex64)[:220,:260],2).real
print("wrapped range", wrapped.min(),wrapped.max(),"exp",exp_m.min(),exp_m.max())
print("wrap err", np.nanstd(wrap_phase(wrapped-exp_m)[mask]))

def score(name, unw):
    d=unw-exp_m
    d=d-np.nanmedian(d[mask])
    print(name, "std",np.nanstd(d[mask]), "p95",np.nanpercentile(np.abs(d[mask]),95), "frac>pi",np.mean(np.abs(d[mask])>np.pi), "range",np.nanmin(unw),np.nanmax(unw))

for cmin in [0.0,0.3,0.5,0.7]:
    unw,info=unwrap_quality_guided(wrapped,coh,mask,coherence_min=cmin)
    score(f"quality cmin={cmin}",unw)

from skimage.restoration import unwrap_phase
filled=fill_nan_nearest(np.where(mask,wrapped,np.nan))
for order in [1,2]:
    # gaussian smooth
    from scipy import ndimage
    f=ndimage.gaussian_filter(filled,order)
    u=unwrap_phase(f.astype(np.float32))
    u=np.where(mask,u,np.nan)
    score(f"skimage gauss{order}",u)

# unwrap clean expected to test method
u,info=unwrap_quality_guided(wrap_phase(exp_m), np.ones_like(coh), mask)
score("quality expected",u)
