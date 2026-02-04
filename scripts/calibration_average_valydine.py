import numpy as np
import matplotlib.pyplot as plt

# =========================
# File path (edit as needed)
# =========================
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FILE_PATH = DATA / "Calibration" / "calibration7.csv"

# =========================
# Load data (EU decimal, tab)
# =========================
data = np.loadtxt(FILE_PATH, delimiter='\t', dtype=str)
time = np.array([float(x.replace(',', '.')) for x in data[:, 0]])
voltage = np.array([float(x.replace(',', '.')) for x in data[:, 1]])

# =========================
# Helpers
# =========================
def secs_to_samples(seconds, t):
    """Convert duration in seconds to number of samples using median dt."""
    if len(t) < 2:
        return 1
    dt = np.median(np.diff(t))
    return max(1, int(round(seconds / dt)))

def detect_stable_regions(time, voltage,
                          window_s=1.0,       # rolling window (s)
                          min_dur_s=5.0,      # minimum plateau duration (s)
                          slope_thresh=None,  # |dv/dt| threshold (V/s), None => auto
                          std_thresh=None,    # rolling std threshold (V), None => auto
                          gap_merge_s=0.5):   # merge gaps shorter than this (s)
    """
    Plateau detector using rolling std + small slope, with merging for tiny gaps.
    Returns: regions [(i0,i1)], roll_std, slope, std_thresh, slope_thresh
    """
    n = len(voltage)
    w = secs_to_samples(window_s, time)

    # Rolling mean/std via cumulative sums (centered by edge-padding)
    pad = w // 2
    vpad = np.pad(voltage, (pad, pad), mode='edge')
    csum = np.cumsum(vpad, dtype=float)
    csum2 = np.cumsum(vpad*vpad, dtype=float)
    win_sum  = csum[w:] - csum[:-w]
    win_sum2 = csum2[w:] - csum2[:-w]
    roll_mean = win_sum / w
    roll_var  = np.maximum(win_sum2 / w - roll_mean**2, 0.0)
    roll_std  = np.sqrt(roll_var)
    roll_mean = roll_mean[:n]
    roll_std  = roll_std[:n]

    # Slope estimate (central difference)
    dt = np.gradient(time)
    dv = np.gradient(voltage)
    slope = np.divide(dv, dt, out=np.zeros_like(dv), where=dt > 0)

    # Auto thresholds if not provided (robust to noise/ramp levels)
    if std_thresh is None:
        base_std = np.percentile(roll_std, 20)
        std_thresh = 2.5 * base_std
    if slope_thresh is None:
        base_slope = np.percentile(np.abs(slope), 20)
        slope_thresh = 3.0 * base_slope

    # Initial stable mask
    stable_mask = (roll_std <= std_thresh) & (np.abs(slope) <= slope_thresh)

    # Merge short gaps to avoid splitting plateaus by spikes
    gap = secs_to_samples(gap_merge_s, time)
    if gap > 0:
        m = stable_mask.copy()
        i = 0
        n = len(m)
        while i < n:
            if not m[i]:
                j = i
                while j < n and not m[j]:
                    j += 1
                gap_len = j - i
                if 0 < gap_len <= gap:
                    m[i:j] = True
                i = j
            else:
                i += 1
        stable_mask = m

    # Convert mask to regions and enforce minimum duration
    min_len = secs_to_samples(min_dur_s, time)
    regions = []
    i = 0
    n = len(stable_mask)
    while i < n:
        if stable_mask[i]:
            j = i
            while j < n and stable_mask[j]:
                j += 1
            if time[j-1] - time[i] >= min_dur_s and (j - i) >= min_len:
                regions.append((i, j-1))
            i = j
        else:
            i += 1

    return regions, roll_std, slope, std_thresh, slope_thresh

# =========================
# Detect stable plateaus
# =========================
stable_regions, roll_std, slope, std_thr, slope_thr = detect_stable_regions(
    time, voltage,
    window_s=1.0,     # try 1.5–2.0 if ramps still sneak in
    min_dur_s=5.0,    # set to your plateau dwell time
    slope_thresh=None,
    std_thresh=None,
    gap_merge_s=0.5
)

# Compute average V per plateau and time intervals
averages = []
time_intervals = []
for start, end in stable_regions:
    averages.append(np.mean(voltage[start:end+1]))
    time_intervals.append((time[start], time[end]))
averages = np.array(averages)

print(f"Detected {len(time_intervals)} stable plateaus")
print(f"Auto thresholds -> std ≤ {std_thr:.6f} V, |dv/dt| ≤ {slope_thr:.6f} V/s")

# =========================
# Pressure mapping: 10 → 0
# =========================
# Map plateaus in chronological order to pressures 10,9,...,0.
# If more than 11 plateaus were detected, keep the first 11.
# If fewer than 11, fit with what's available (10 downwards).
target_count = 11  # 10..0
if len(averages) >= target_count:
    averages = averages[:target_count]
    time_intervals = time_intervals[:target_count]
    pressures = np.arange(10, -1, -1)  # 10,9,...,0 (length 11)
else:
    pressures = np.arange(10, 10 - len(averages), -1)  # e.g., 10,9,8,...
    print(f"Warning: detected only {len(averages)} stable plateaus; expected 11 (10→0).")

# =========================
# Linear regression: P = a*V + b
# =========================
coeffs = np.polyfit(averages, pressures, 1)
slope_fit, intercept_fit = coeffs
pressure_pred = slope_fit * averages + intercept_fit

# R^2
ss_res = np.sum((pressures - pressure_pred) ** 2)
ss_tot = np.sum((pressures - np.mean(pressures)) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f"Linear regression: Pressure = {slope_fit:.6f} * Voltage + {intercept_fit:.6f}")
print(f"R-squared: {r_squared:.4f}")

# =========================
# Plots
# =========================
plt.figure(figsize=(14,6))
plt.plot(time, voltage, label='Voltage', linewidth=1)
for k, ((t0, t1), avg_v) in enumerate(zip(time_intervals, averages)):
    plt.hlines(avg_v, t0, t1, colors='red', linewidth=3,
               label='Stable region' if k == 0 else "")
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Voltage vs Time with Stable Regions')
plt.legend()
plt.grid(True)

plt.figure(figsize=(6,5))
plt.scatter(averages, pressures, label='Data points')
ord_idx = np.argsort(averages)
plt.plot(averages[ord_idx], pressure_pred[ord_idx], 'r-', label='Linear fit')
plt.xlabel('Voltage [V]')
plt.ylabel('Pressure [mm H₂O]')
plt.title('Pressure vs Voltage (10 → 0)')
plt.legend()
plt.grid(True)

plt.show()
