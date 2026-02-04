import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate
from pathlib import Path

# =========================================
# CONFIGURATION
# =========================================
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[1]

# Data root
BASE_ROOT = ROOT / "data"

FOLDERS = {
    "Downstream 5 m/s":  BASE_ROOT / "hot_wire_downstream_5",
    "Downstream 10 m/s": BASE_ROOT / "hot_wire_downstream_10",
    "Upstream 5 m/s":    BASE_ROOT / "hot_wire_upstream_5",
    "Upstream 10 m/s":   BASE_ROOT / "hot_wire_upstream_10",
}


# Hot-wire V->U (anchored ΔV^4) — coefficients in m/s for powers of (V-V0)^1..^4
HW_DV4_COEFFS = [ +2.798, -18.46, +79.90, -45.09 ]

# Baseline voltages V0
V0_MAP = {
    "Downstream 5 m/s": 2.00594,
    "Downstream 10 m/s": 1.990594,
    "Upstream 5 m/s": 1.983034,
    "Upstream 10 m/s": 1.983034,
}

# Flow props (adjust if you want)
RHO = 1.20       # kg/m^3
NU  = 1.50e-5    # m^2/s
MU  = RHO * NU

# Welch / plotting
MAX_NPERSEG = 8192          # cap segment length
BINS_PER_DECADE = 40        # log-binning smoothness
PLOT_COMPENSATED = False    # set True to show f^{5/3} * S_uu(f)

# =========================================
# UTILITIES
# =========================================
def U_from_hotwire(V, V0):
    """Convert voltage array V to velocity using anchored ΔV^4 polynomial."""
    V = np.asarray(V, dtype=np.float64)
    dV = V - float(V0)
    return (HW_DV4_COEFFS[0]*dV
          + HW_DV4_COEFFS[1]*dV**2
          + HW_DV4_COEFFS[2]*dV**3
          + HW_DV4_COEFFS[3]*dV**4)

def read_point_file(file):
    """Read one point file with commas as decimals and tab/space separator."""
    with open(file, "r", encoding="utf-8") as f:
        lines = [ln.strip().replace(",", ".") for ln in f if ln.strip()]
    data = [list(map(float, ln.split())) for ln in lines]
    return pd.DataFrame(data, columns=["time_s", "p_dyn", "V_hw", "p_static"])

def sorted_point_files(folder_path):
    """Return [(y_index, Path), ...] sorted by numeric part in 'pointXX' name."""
    all_files = [f for f in folder_path.glob("point*") if f.suffix.lower() in (".csv", ".txt")]
    out = []
    for f in all_files:
        digits = ''.join(ch for ch in f.stem if ch.isdigit())
        if digits:
            out.append((int(digits), f))
    return sorted(out, key=lambda x: x[0])

def pick_fft_index(y_mm):
    """Choose ~mid boundary-layer position for representative spectrum."""
    if len(y_mm) == 0:
        return None
    y_target = 0.01 * np.max(y_mm)
    return int(np.argmin(np.abs(np.asarray(y_mm) - y_target)))

def log_bin_spectrum(f, S, bins_per_decade=40):
    """Log-binning to smooth PSD."""
    f = np.asarray(f, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    mask = (f > 0) & np.isfinite(S)
    f, S = f[mask], S[mask]
    logf = np.log10(f)
    n_bins = max(10, int((logf.max() - logf.min()) * bins_per_decade))
    edges = np.linspace(logf.min(), logf.max(), n_bins + 1)
    fbin, Sbin = [], []
    for i in range(n_bins):
        m = (logf >= edges[i]) & (logf < edges[i+1])
        if np.any(m):
            fbin.append(np.mean(f[m]))
            Sbin.append(np.mean(S[m]))
    return np.array(fbin), np.array(Sbin)

def integral_time_scale(u, fs):
    """Integral time scale t_L from autocorrelation (integrate until first zero)."""
    u = np.asarray(u, np.float64)
    u = u - np.mean(u)
    corr = correlate(u, u, mode="full")
    corr = corr[corr.size//2:]
    corr = corr / corr[0]
    dt = 1.0 / fs
    # find first zero crossing; if none, use all
    idx0 = np.argmax(corr < 0.0)
    if idx0 == 0:
        idx0 = len(corr)
    tL = np.trapz(corr[:idx0], dx=dt)
    return tL

def error_estimates(u, Umean, fs):
    """Return N, tL, 95% relative errors in mean U and TI, and TI value."""
    tL = integral_time_scale(u, fs)
    T  = len(u) / fs
    N  = T / tL if tL > 0 else np.nan
    Urms = np.std(u - Umean)
    TI = Urms / Umean if Umean > 0 else np.nan
    errU  = 2 * TI / np.sqrt(N) if N and N > 0 else np.nan
    errTI = 2 / np.sqrt(2*N)    if N and N > 0 else np.nan
    return N, tL, errU, errTI, TI

# =========================================
# MAIN
# =========================================
# Store all TI profiles for combined plot
all_ti_profiles = {}

for label, folder in FOLDERS.items():
    files = sorted_point_files(folder)
    if not files:
        print(f"⚠️ No valid files in {label}")
        continue

    V0 = V0_MAP[label]
    y_mm_list, Tu_list, U_series, t_series = [], [], [], []

    # ---------- Load all points & compute TI profile ----------
    for pos, file in files:
        df = read_point_file(file)
        y_mm = pos * 0.1
        t = df["time_s"].to_numpy(np.float64)
        V = df["V_hw"].to_numpy(np.float64)
        U = U_from_hotwire(V, V0)

        Umean = float(np.mean(U))
        Urms  = float(np.std(U - Umean))
        Tu    = 100.0 * Urms / Umean if Umean > 1e-9 else np.nan

        y_mm_list.append(y_mm)
        Tu_list.append(Tu)
        U_series.append(U.astype(np.float64))
        t_series.append(t.astype(np.float64))

    # sort by y
    order = np.argsort(y_mm_list)
    y_mm = np.array(y_mm_list)[order]
    Tu   = np.array(Tu_list)[order]
    U_series = [U_series[i] for i in order]
    t_series = [t_series[i] for i in order]

    # Store for combined plot
    all_ti_profiles[label] = (y_mm, Tu)
    
    print(f"\n✅ {label}: {len(y_mm)} points processed. Max Tu = {np.nanmax(Tu):.2f}%")

    # Individual TI profile
    plt.figure(figsize=(7,5))
    plt.plot(Tu, y_mm, "o-", ms=4)
    plt.xlabel("Turbulence Intensity Tu [%]")
    plt.ylabel("y [mm]")
    plt.title(f"Turbulence Intensity — {label}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    # ---------- Choose representative point for spectra ----------
    idx_fft = pick_fft_index(y_mm)
    if idx_fft is None:
        print(f"⚠️ No FFT index chosen for {label}")
        continue

    u = U_series[idx_fft].copy()
    t = t_series[idx_fft].copy()
    u -= np.mean(u)  # zero-mean for spectrum

    # sampling freq
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    print(f"  → Inferred fs ≈ {fs:.1f} Hz | N = {len(u)} samples | y ≈ {y_mm[idx_fft]:.2f} mm")

    # ---------- Welch PSD with NumPy Hann window (avoids SciPy bug) ----------
    nperseg = int(min(MAX_NPERSEG, len(u)))
    if nperseg < 64:
        print(f"⚠️ Not enough samples for Welch in {label} (n={len(u)})")
        continue

    # NumPy Hann (Hanning) window — avoids SciPy get_window bug
    window = np.hanning(nperseg).astype(np.float64)
    noverlap = nperseg // 2

    f, Pxx = welch(
        u.astype(np.float64),
        fs=fs,
        window=window,      # <- pass array, not string
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",  # approximate PSD
    )

    # Log-binning for smoother PSD
    fbin, Sbin = log_bin_spectrum(f, Pxx, bins_per_decade=BINS_PER_DECADE)

    # ---------- Plot spectrum ----------
    plt.figure(figsize=(7.8,5.3))
    if PLOT_COMPENSATED:
        # compensated: f^{5/3} * S(f) should be ~constant in inertial range
        comp = Sbin * (fbin ** (5.0/3.0))
        plt.loglog(fbin, comp, label=f"{label} @ y={y_mm[idx_fft]:.1f} mm")
        plt.ylabel(r"$f^{5/3} S_{uu}(f)$  [a.u.]")
        plt.title("Compensated Velocity Spectrum (Welch + log-binning)")
    else:
        plt.loglog(fbin, Sbin, label=f"{label} @ y={y_mm[idx_fft]:.1f} mm")
        # add a −5/3 slope guide through a point near 100 Hz
        fref = 100.0
        j = np.argmin(np.abs(fbin - fref))
        if np.isfinite(j) and 0 <= j < len(Sbin):
            A = Sbin[j] * (fref ** (5.0/3.0))
            fline = np.logspace(np.log10(fref/6), np.log10(fref*6), 200)
            plt.loglog(fline, A * (fline ** (-5.0/3.0)), "k--", lw=1.0, label="−5/3 slope")
        plt.ylabel(r"$S_{uu}(f)$  [a.u.]")
        plt.title("Velocity Spectrum (Welch + log-binning)")

    plt.xlabel("Frequency f [Hz]")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Integral time scale, N, and 95% errors ----------
    # Use *raw* (not log-binned) zero-mean signal for time-domain stats
    Umean_abs = np.mean(U_series[idx_fft])  # mean before de-meaning
    N, tL, errU, errTI, TI = error_estimates(U_series[idx_fft], Umean_abs, fs)

    print(f"  Integral time scale tL = {tL*1000:.2f} ms")
    print(f"  Independent samples   N = {N:.1f}")
    print(f"  TI (at that y)          = {TI*100:.2f}%")
    print(f"  Rel. error in mean U    = ±{(errU*100 if np.isfinite(errU) else np.nan):.2f}% (95%)")
    print(f"  Rel. error in TI        = ±{(errTI*100 if np.isfinite(errTI) else np.nan):.2f}% (95%)")

# =========================================
# COMBINED TI PROFILE PLOT
# =========================================
if all_ti_profiles:
    plt.figure(figsize=(8, 6))
    
    # Define colors and markers for different cases
    colors = {
        "Downstream 5 m/s": "blue",
        "Downstream 10 m/s": "red", 
        "Upstream 5 m/s": "green",
        "Upstream 10 m/s": "orange"
    }
    
    markers = {
        "Downstream 5 m/s": "o",
        "Downstream 10 m/s": "s",
        "Upstream 5 m/s": "^",
        "Upstream 10 m/s": "D"
    }
    
    for label, (y_mm, Tu) in all_ti_profiles.items():
        color = colors.get(label, "black")
        marker = markers.get(label, "o")
        plt.plot(Tu, y_mm, marker=marker, linestyle='-', color=color, 
                markersize=4, linewidth=1.5, label=label)
    
    plt.xlabel("Turbulence Intensity Tu [%]", fontsize=12)
    plt.ylabel("y [mm]", fontsize=12)
    plt.title("Turbulence Intensity Profiles - All Cases", fontsize=14)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Combined TI plot created with {len(all_ti_profiles)} profiles")
else:
    print("⚠️ No TI profiles to combine")