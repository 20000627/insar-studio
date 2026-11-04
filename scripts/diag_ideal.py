import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.utils import downsample_complex, downsample_real, wrap_phase

root = Path(__file__).resolve().parents[1] / "workspace" / "prototypes" / "p1"
stack=load_stack(root); meta=stack.meta
truth=np.load(root/"data/stack/truth.npz")
net=json.load(open(root/"data/network.json")); pairs=net["pairs"]
coh=np.load(root/"interferograms/00_01/coherence.npy")
mask=np.load(root/"interferograms/00_01/mask.npy")
unw=np.load(root/"interferograms/00_01/unwrapped.npy")
disp=np.load(root/"interferograms/00_01/displacement_mm.npy")
print("unw range", np.nanmin(unw), np.nanmax(unw), "disp", np.nanmin(disp), np.nanmax(disp))
print("coh", np.nanmean(coh), "mask", mask.mean(), "meta", json.load(open(root/"interferograms/00_01/meta.json"))["stats"])
m=2; rows=110; cols=130
p=pairs[0]; t=p["sec"]
k=4*np.pi/meta["wavelength_m"]
exp=k*((truth["disp_los_mm"][t]-truth["disp_los_mm"][0]) + (truth["atmosphere_mm"][t]-truth["atmosphere_mm"][0]))/1000
bperp=np.asarray(truth["bperp"])
exp=exp+k*(bperp[t]-bperp[0])*(truth["dem_true"]-truth["dem_ref"])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
exp_m=downsample_complex(exp.astype(np.complex64)[:220,:260],2).real
# reference at same pixel as product? examine differences after removing mean
diff=unw-exp_m
print("pair0 unw-exp median",np.nanmedian(diff[mask]),"std",np.nanstd(diff[mask]),"p95",np.nanpercentile(np.abs(diff[mask]),95))
# fit unw = exp + const + 2pi*k? examine histogram of diff/2pi
h=np.histogram(diff[mask]/ (2*np.pi), bins=20)
print("diff/2pi hist", h[0], h[1])
# check expected phase itself range
print("exp range", exp_m.min(), exp_m.max())
# ideal inversion cube
arr=[]
for p in pairs:
    t=p["sec"]; r=p["ref"]
    e=k*((truth["disp_los_mm"][t]-truth["disp_los_mm"][r]) + (truth["atmosphere_mm"][t]-truth["atmosphere_mm"][r]))/1000
    e=e+k*(bperp[t]-bperp[r])*(truth["dem_true"]-truth["dem_ref"])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
    e=downsample_complex(e.astype(np.complex64)[:220,:260],2).real
    arr.append(e.astype(np.float32))
ideal=np.stack(arr)
print("ideal shape", ideal.shape)
from insar_studio.core.inversion import invert_arrays
res=invert_arrays(root, ideal, np.ones_like(ideal)*0.9, np.ones_like(ideal,dtype=bool), pairs, meta, {"aps_method":"off","regularization":0.08}, extra_mask=None)
print("ideal vel", np.nanmedian(res["velocity_mm_yr"]), np.nanmin(res["velocity_mm_yr"]), np.nanmax(res["velocity_mm_yr"]))
truth_vel=downsample_real(truth["velocity_mm_yr"],2)
print("truth vel", np.nanmedian(truth_vel), np.nanmin(truth_vel), np.nanmax(truth_vel))
print("ideal dem", np.nanmedian(res["dem_error_m"]), "truth", np.nanmedian(downsample_real(truth["dem_true"]-truth["dem_ref"],2)))
