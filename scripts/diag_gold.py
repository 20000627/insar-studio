import sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_complex, wrap_phase, fill_nan_nearest
from insar_studio.core.unwrapping import unwrap_quality_guided

root=Path(__file__).resolve().parents[1]/"workspace/prototypes/p1"
stack=load_stack(root); meta=stack.meta; truth=np.load(root/"data/stack/truth.npz")
m=2; coreg=np.load(root/"data/processed/slc_coreg.npy",mmap_mode="r")
ref=np.asarray(coreg[0])[:220,:260]; sec=np.asarray(coreg[1])[:220,:260]
ifg=downsample_complex(sec*np.conj(ref),2)
dem=downsample_complex(np.asarray(stack.dem,dtype=np.complex64)[:220,:260],2).real
k=4*np.pi/meta["wavelength_m"]; b=meta["bperp_m"][1]-meta["bperp_m"][0]
ifg=ifg*np.exp(-1j*k*b*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"]))))
coh=np.load(root/"interferograms/00_01/coherence.npy"); mask=np.load(root/"interferograms/00_01/mask.npy")
exp=k*((truth["disp_los_mm"][1]-truth["disp_los_mm"][0])+(truth["atmosphere_mm"][1]-truth["atmosphere_mm"][0]))/1000
bperp=np.asarray(truth["bperp"]); exp+=k*(bperp[1]-bperp[0])*(truth["dem_true"]-truth["dem_ref"])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
exp_m=downsample_complex(exp.astype(np.complex64)[:220,:260],2).real

def gold(ifg, alpha=0.5, patch=32, cap=None):
    z=np.fft.fftshift(np.fft.fft2(ifg)); a=np.abs(z)
    s=ndimage.uniform_filter(a,size=patch,mode="nearest")
    h=(s/(a+1e-8))**alpha
    if cap: h=np.clip(h,1/cap,cap)
    cr,cc=np.array(z.shape)//2; h[cr,cc]=1
    return np.fft.ifft2(np.fft.ifftshift(z*h)).astype(np.complex64)

def score(name, f):
    ph=np.angle(f)
    print(name,"fil phase diff std",np.nanstd(wrap_phase(ph-np.angle(ifg))[mask]),end=" ")
    u,info=unwrap_quality_guided(ph,coh,mask)
    d=u-exp_m; d-=np.nanmedian(d[mask])
    print("unwrap std",np.nanstd(d[mask]),"p95",np.nanpercentile(np.abs(d[mask]),95),"frac",np.mean(np.abs(d[mask])>np.pi))
for a in [0.2,0.35,0.5,0.65]:
    for cap in [None,2,1.5]:
        score(f"a={a} cap={cap}",gold(ifg,a,32,cap))
# gaussian spectrum smoothing instead of boxcar
def gold2(ifg,alpha=0.5,sigma=8):
    z=np.fft.fftshift(np.fft.fft2(ifg)); a=np.abs(z)
    s=ndimage.gaussian_filter(a,sigma=sigma)
    h=(s/(a+1e-8))**alpha
    cr,cc=np.array(z.shape)//2; h[cr,cc]=1
    return np.fft.ifft2(np.fft.ifftshift(z*h)).astype(np.complex64)
for s in [4,8,16]:
    score(f"gauss spec {s}",gold2(ifg,0.5,s))
