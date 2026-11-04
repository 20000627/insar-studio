import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insar_studio.core.io import load_stack
from insar_studio.core.inversion import load_pair_cube, invert_arrays
from insar_studio.core.utils import downsample_real, downsample_complex

root = Path(__file__).resolve().parents[1] / "workspace" / "prototypes" / "p1"
stack = load_stack(root); meta = stack.meta
truth = np.load(root / "data/stack/truth.npz", allow_pickle=True)
net = json.load(open(root / "data/network.json")); pairs = net["pairs"]
phase, coh, mask, kept = load_pair_cube(root, pairs, corrected=False)
m = 2; k = 4 * np.pi / meta["wavelength_m"]
atm = truth["atmosphere_mm"]
bp = np.asarray(truth["bperp"])
dem_true = truth["dem_true"]; dem_ref = truth["dem_ref"]
td0 = truth["disp_los_mm"] - truth["disp_los_mm"][0]
td = np.stack([downsample_real(td0[i], m) for i in range(td0.shape[0])])
# fair truth velocity = LS slope of truth displacement
t = np.asarray(meta["sampling_days"], dtype=float); t = t - t.mean()
td1 = td.reshape(td.shape[0], -1)
tv = (t[:, None] * td1).sum(0) / (t ** 2).sum() * 365.25
tv = tv.reshape(td.shape[1:])
tdem = downsample_real(dem_true - dem_ref, m)

def corrected(use_atm, use_dem):
    ph = phase.copy()
    for idx, p in enumerate(kept):
        r, s = p["ref"], p["sec"]
        if use_atm:
            d = downsample_complex((k * (atm[s] - atm[r]) / 1000).astype(np.complex64)[:220, :260], m).real
            ph[idx] -= d
        if use_dem:
            d = downsample_complex((k * (bp[s] - bp[r]) * tdem_full / (meta["slant_range_m"] * np.sin(np.deg2rad(meta["incidence_deg"])))).astype(np.complex64)[:220, :260], m).real
            ph[idx] -= d
    return ph

tdem_full = (dem_true - dem_ref).astype(np.float32)
for use_atm in (False, True):
    for use_dem in (False, True):
        ph = corrected(use_atm, use_dem)
        for estdem in (False, True):
            for aps in (0, 1, 2):
                res = invert_arrays(root, ph, coh, mask, kept, meta, {
                    "estimate_dem_error": estdem, "aps_iterations": aps,
                    "aps_sigma_px": 10.0, "regularization": 0.05,
                })
                vel = res["velocity_mm_yr"]; valid = res["valid"]; disp = res["disp_mm"]
                v = tv[:vel.shape[0], :vel.shape[1]]
                d = td[:, :disp.shape[1], :disp.shape[2]]
                vr = np.sqrt(np.nanmean((vel - v)[valid] ** 2)); dr = np.sqrt(np.nanmean((disp - d)[:, valid] ** 2))
                de = res["dem_error_m"]
                dt = tdem[:de.shape[0], :de.shape[1]]
                corr = np.corrcoef(de[valid], dt[valid])[0, 1] if np.isfinite(de[valid]).any() else 0
                print(f"atm={use_atm} demfix={use_dem} est={estdem} aps={aps} vRMSE={vr:5.2f} dRMSE={dr:5.2f} vmed={np.nanmedian(vel[valid]):6.2f} demMed={np.nanmedian(de[valid]):6.1f} corr={corr:4.2f} tcoh={np.nanmedian(res['temporal_coherence'][valid]):.2f}")
