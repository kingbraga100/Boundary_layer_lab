import numpy as np
import matplotlib.pyplot as plt

# =========================
# File path (edit if needed)
# =========================
FILE_PATH = r"C:\Users\ricar\OneDrive\Desktop\VKI\ELab - Experimental Laboratory Exercise\2025_BL\BL_GroupD_2025\data\Calibration\day3_Validyne_Calibration_V2.csv"


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FILE_PATH = DATA / "Calibration" / "day3_Validyne_Calibration_V2.csv"

# =========================
# Load data (EU decimal, tab)
# =========================
data = np.loadtxt(FILE_PATH, delimiter='\t', dtype=str)
time = np.array([float(x.replace(',', '.')) for x in data[:, 0]])
voltage = np.array([float(x.replace(',', '.')) for x in data[:, 1]])

# =========================
# Utilities (unchanged)
# =========================
def secs_to_samples(seconds, t):
    if len(t) < 2:
        return 1
    dt = np.median(np.diff(t))
    return max(1, int(round(seconds / dt)))

def moving_median(x, w):
    w = max(1, int(w) | 1)               # odd
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = np.median(xpad[i:i+2*pad+1])
    return out

def rolling_slope_and_mad(t, v, window_s=1.8):
    n = len(v)
    w = secs_to_samples(window_s, t)
    w = max(3, w | 1)
    pad = w // 2
    v_smooth = moving_median(v, max(3, (w//3) | 1))
    vpad = np.pad(v_smooth, (pad, pad), mode='edge')
    tpad = np.pad(t, (pad, pad), mode='edge')
    slopes = np.zeros(n)
    mads = np.zeros(n)
    for i in range(n):
        tw = tpad[i:i+2*pad+1]
        vw = vpad[i:i+2*pad+1]
        tm = tw.mean()
        x = tw - tm
        varx = np.sum(x*x)
        slopes[i] = 0.0 if varx <= 0 else np.sum(x*vw)/varx
        med = np.median(vw)
        mads[i] = np.median(np.abs(vw - med))
    return slopes, mads, w

def merge_short_gaps(mask, max_gap_samples):
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if 0 < (j - i) <= max_gap_samples:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m

def mask_to_regions(mask, t, min_len, min_edge_len=None, min_start_len=None):
    regions = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            length = j - i
            if i == 0 and min_start_len is not None:
                need = min_start_len
            elif j == n and min_edge_len is not None:
                need = min_edge_len
            else:
                need = min_len
            if length >= need:
                regions.append((i, j-1))
            i = j
        else:
            i += 1
    return regions

def deduplicate_with_ramp_guard(averages, regions, slopes, time,
                                min_step_v=None, ramp_guard_mult=6.0):
    avs = list(averages); regs = list(regions)
    if len(avs) <= 1:
        return np.array(avs), regs
    diffs = np.diff(avs)
    pos = diffs[diffs > 0]
    if min_step_v is None:
        if len(pos) > 0:
            min_step_v = max(0.45 * np.median(pos), 0.007)
        else:
            min_step_v = 0.015
    base_slope = np.percentile(np.abs(slopes), 30)
    ramp_thr = ramp_guard_mult * max(base_slope, 1e-9)
    keep_avs = [avs[0]]
    keep_regs = [regs[0]]
    for k in range(1, len(avs)):
        prev_i0, prev_i1 = keep_regs[-1]
        cur_i0, cur_i1 = regs[k]
        dv = avs[k] - keep_avs[-1]
        if prev_i1 + 1 <= cur_i0 - 1:
            gap_max_slope = np.max(np.abs(slopes[prev_i1+1:cur_i0]))
        else:
            gap_max_slope = 0.0
        if (dv <= min_step_v) and (gap_max_slope < ramp_thr):
            keep_regs[-1] = (prev_i0, cur_i1)
            keep_avs[-1] = max(keep_avs[-1], avs[k])
        else:
            keep_avs.append(avs[k])
            keep_regs.append(regs[k])
    return np.array(keep_avs), keep_regs

# =========================
# Detection settings (unchanged)
# =========================
window_s          = 1.9
min_dur_s         = 1.2
min_edge_dur_s    = 0.8
min_start_dur_s   = 0.12
gap_merge_s       = 1.0

# =========================
# Feature extraction
# =========================
slopes, mads, _ = rolling_slope_and_mad(time, voltage, window_s=window_s)

# Robust auto-thresholds
mad_base    = np.percentile(mads, 30)
slope_base  = np.percentile(np.abs(slopes), 30)
mad_thresh  = 3.8 * max(mad_base, 1e-9)
slope_thresh= 3.2 * max(slope_base, 1e-9)

stable_mask = (mads <= mad_thresh) & (np.abs(slopes) <= slope_thresh)
stable_mask = merge_short_gaps(stable_mask, secs_to_samples(gap_merge_s, time))

# =========================
# Build regions
# =========================
min_len       = secs_to_samples(min_dur_s, time)
min_edge_len  = secs_to_samples(min_edge_dur_s, time)
min_start_len = secs_to_samples(min_start_dur_s, time)

regions = mask_to_regions(stable_mask, time, min_len,
                          min_edge_len=min_edge_len,
                          min_start_len=min_start_len)

# (optional) tiny baseline at the very beginning
if not regions or regions[0][0] != 0:
    search_end = np.searchsorted(time, time[0] + 2.0, side="right")
    small_win = secs_to_samples(0.20, time)
    for i in range(max(1, search_end - small_win)):
        j = i + small_win
        if np.median(np.abs(slopes[i:j])) <= 0.5 * slope_thresh:
            regions = [(i, j-1)] + regions
            break

averages = np.array([np.mean(voltage[i0:i1+1]) for (i0, i1) in regions])
averages, regions = deduplicate_with_ramp_guard(
    averages, regions, slopes, time, min_step_v=None, ramp_guard_mult=6.0
)

# =========================
# *** NEW PART *** – 0 → 10 → 0 (21 plateaus)
# =========================
target = 21                                   # 0,1,…,10,…,1,0  → 21 steps
if len(regions) > target:
    # keep the longest `target` regions (the first one is always kept)
    first = regions[0]
    rest  = regions[1:]
    scores = []
    for (i0, i1), a in zip(rest, averages[1:]):
        dur = time[i1] - time[i0]
        scores.append((dur, (i0, i1), a))
    scores.sort(reverse=True)
    chosen = [first] + [r for _, r, _ in scores[:target-1]]
    chosen.sort(key=lambda ij: ij[0])
    regions = chosen
    averages = np.array([np.mean(voltage[i0:i1+1]) for (i0, i1) in regions])

# Pressure vector for the 21 plateaus
pressures = np.concatenate([np.arange(0, 11, 1), np.arange(9, -1, -1)])   # 0..10 then 9..0

# Trim if we somehow have fewer than 21 regions
if len(averages) > len(pressures):
    averages = averages[:len(pressures)]
elif len(averages) < len(pressures):
    pressures = pressures[:len(averages)]

# =========================
# *** THREE REGRESSIONS ***
# =========================
def linreg(x, y):
    """Return slope, intercept, R²"""
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2

# 1. Up-ramp (indices 0 … 10)
up_idx   = slice(0, 11)
a_up, b_up, r2_up = linreg(averages[up_idx], pressures[up_idx])

# 2. Down-ramp (indices 11 … 20)
down_idx = slice(11, 21)
a_down, b_down, r2_down = linreg(averages[down_idx], pressures[down_idx])

# 3. Combined
a_all, b_all, r2_all = linreg(averages, pressures)

# Print results
print("\n=== Calibration results (0 → 10 → 0) ===")
print(f"Up-ramp   : P = {a_up:.6f}*V + {b_up:.6f}   (R² = {r2_up:.4f})")
print(f"Down-ramp : P = {a_down:.6f}*V + {b_down:.6f}   (R² = {r2_down:.4f})")
print(f"Combined  : P = {a_all:.6f}*V + {b_all:.6f}   (R² = {r2_all:.4f})")

# =========================
# Plots
# =========================
time_intervals = [(time[i0], time[i1]) for (i0, i1) in regions]

# ---- Voltage vs Time with stable regions ----
plt.figure(figsize=(14,6))
plt.plot(time, voltage, label='Voltage', linewidth=1)
for k, ((t0, t1), avg_v) in enumerate(zip(time_intervals, averages)):
    plt.hlines(avg_v, t0, t1, colors='red', linewidth=3,
               label='Stable region' if k == 0 else "")
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Voltage vs Time with 21 Stable Regions')
plt.legend()
plt.grid(True)

# ---- Pressure vs Voltage (three fits) ----
plt.figure(figsize=(8,6))
plt.scatter(averages, pressures, c='k', label='Data points')

# Up-ramp line
x_up = np.linspace(averages[up_idx].min(), averages[up_idx].max(), 100)
plt.plot(x_up, a_up*x_up + b_up, 'b--', label='Up-ramp fit')

# Down-ramp line
x_down = np.linspace(averages[down_idx].min(), averages[down_idx].max(), 100)
plt.plot(x_down, a_down*x_down + b_down, 'g--', label='Down-ramp fit')

# Combined line
x_all = np.linspace(averages.min(), averages.max(), 100)
plt.plot(x_all, a_all*x_all + b_all, 'r-', label='Combined fit')

plt.text(0.02, 0.98,
         f"Up:   P = {a_up:.3f}V + {b_up:.3f}  (R²={r2_up:.4f})\n"
         f"Down: P = {a_down:.3f}V + {b_down:.3f}  (R²={r2_down:.4f})\n"
         f"All:   P = {a_all:.3f}V + {b_all:.3f}  (R²={r2_all:.4f})",
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

plt.xlabel('Voltage [V]')
plt.ylabel('Pressure [mm H₂O]')
plt.title('Pressure vs Voltage – 0 → 10 → 0 (21 plateaus)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()