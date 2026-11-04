import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.utils import wrap_phase, downsample_complex

root = Path(__file__).resolve().parents[1] / "workspace" / "prototypes" / "p1"
stack = load_stack(root)
truth = np.load(root / "data/stack/truth.npz")
meta = stack.meta
m = 2
coreg = np.load(root / "data/stack/../processed/slc_coreg.npy", mmap_mode="r")
ref = np.asarray(coreg[0])[:218, :260]
sec = np.asarray(coreg[5])[:218, :260]
ifg = downsample_complex(sec * np.conj(ref), m)
dem = downsample_complex(np.asarray(stack.dem, dtype=np.complex64)[:218, :260], m).real
mask = downsample_complex(stack.mask.astype(np.complex64)[:218,:260], m).real > .55
k = 4*np.pi/meta["wavelength_m"]
b = meta["bperp_m"][5]-meta["bperp_m"][0]
topo = k*b*dem/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
ifg2 = ifg*np.exp(-1j*topo)
# expected differential phase
bperp = np.asarray(truth["bperp"])
disp = truth["disp_los_mm"]; atm = truth["atmosphere_mm"]
t = 5
exp = k*(disp[t][:218,:260]-disp[0][:218,:260])/1000 + k*(atm[t][:218,:260]-atm[0][:218,:260])/1000
# topo residual from dem error
exp += k*(bperp[t]-bperp[0])*(truth["dem_true"][:218,:260]-truth["dem_ref"][:218,:260])/(meta["slant_range_m"]*np.sin(np.deg2rad(meta["incidence_deg"])))
# downsample exp
exp_m = downsample_complex(exp.astype(np.complex64)[:218,:260], m).real
resid = wrap_phase(np.angle(ifg2) - exp_m)
print("phase residual std (rad)", np.nanstd(resid[mask]))
print("phase residual median abs", np.nanmedian(np.abs(resid[mask])))
print("truth shift", truth["shifts"][5], "coreg result", np.load(root/"data/processed/coreg.json").item() if False else "")
import json
cg=json.load(open(root/"data/processed/coreg.json"))
print("transform", cg["transforms"][5])
# compare truth shift vs median_shift
print("truth (col,row)", truth["shifts"][5], "est (col,row)", cg["transforms"][5].get("median_shift_col"), cg["transforms"][5].get("median_shift_row"))
