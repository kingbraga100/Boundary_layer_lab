import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
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



# Hot-wire ΔV^4 calibration (anchored)
HW_DV4_COEFFS = [ +2.798, -18.46, +79.90, -45.09 ]
V0_MAP = {
    "Downstream 5 m/s": 2.00594,
    "Downstream 10 m/s": 1.990594,
    "Upstream 5 m/s": 1.983034,
    "Upstream 10 m/s": 1.983034,
}

# Colors
COLORS = {
    "Downstream 5 m/s": "tab:blue",
    "Downstream 10 m/s": "tab:orange",
    "Upstream 5 m/s": "tab:green",
    "Upstream 10 m/s": "tab:red",
}

# ============================================================
# HELPERS
# ============================================================
def U_from_hotwire(V, V0):
    dV = V - V0
    return (HW_DV4_COEFFS[0]*dV
          + HW_DV4_COEFFS[1]*dV**2
          + HW_DV4_COEFFS[2]*dV**3
          + HW_DV4_COEFFS[3]*dV**4)

def read_point_file(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = [ln.strip().replace(",", ".") for ln in f if ln.strip()]
    data = [list(map(float, ln.split())) for ln in lines]
    return pd.DataFrame(data, columns=["time_s","p_dyn","V_hw","p_static"])

def sorted_point_files(folder_path):
    all_files = [f for f in folder_path.glob("point*") if f.suffix.lower() in [".csv",".txt"]]
    out = []
    for f in all_files:
        digits = ''.join(ch for ch in f.stem if ch.isdigit())
        if digits:
            out.append((int(digits), f))
    return sorted(out, key=lambda x: x[0])

def load_profile(label, path):
    """Return y_mm, U_mean, U_inf (U_inf = mean of last 5 U points)."""
    files = sorted_point_files(path)
    if not files:
        return None
    V0 = V0_MAP[label]
    y_mm, U_mean = [], []
    for pos, file in files:
        df = read_point_file(file)
        V_avg = df["V_hw"].mean()
        U_avg = U_from_hotwire(V_avg, V0)
        y_mm.append(pos * 0.1)  # 0.1 mm per index
        U_mean.append(U_avg)
    y_mm = np.asarray(y_mm); U_mean = np.asarray(U_mean)
    idx = np.argsort(y_mm)
    y_mm, U_mean = y_mm[idx], U_mean[idx]
    U_inf = float(np.mean(U_mean[-min(5, len(U_mean)):] ))
    return y_mm, U_mean, U_inf

def interp_y_at_U(y, U, target):
    """Linear interpolation for y where U = target."""
    y = np.asarray(y); U = np.asarray(U)
    above = np.where(U >= target)[0]
    if len(above) == 0:
        return np.nan
    j = above[0]
    if j == 0:
        if U[0] == target: return y[0]
        if len(U) > 1:
            return y[0] + (target - U[0]) * (y[1]-y[0]) / (U[1]-U[0])
        return np.nan
    y0, y1 = y[j-1], y[j]
    U0, U1 = U[j-1], U[j]
    if U1 == U0: return y1
    return y0 + (target - U0) * (y1 - y0) / (U1 - U0)

def compute_integral_params(y_mm, U):
    """
    Compute U_inf, δ99, δ*, θ, H for one profile.
    y_mm in mm, U in m/s.
    """
    y = np.asarray(y_mm, dtype=float)
    U = np.asarray(U, dtype=float)
    idx = np.argsort(y)
    y, U = y[idx], U[idx]

    k = min(5, len(U))
    U_inf = float(np.mean(U[-k:]))

    # δ99 (mm)
    delta99_mm = interp_y_at_U(y, U, 0.99*U_inf)
    if np.isnan(delta99_mm):
        delta99_mm = y[-1]

    # Subset up to δ99
    mask = y <= delta99_mm
    y_sub = y[mask]
    U_sub = U[mask]
    Uy = U_sub / U_inf

    # Integrate in meters
    y_m = y_sub * 1e-3
    delta_star = np.trapz((1 - Uy), y_m)       # [m]
    theta      = np.trapz(Uy * (1 - Uy), y_m)  # [m]
    H = delta_star / theta if theta > 0 else np.nan

    return dict(
        U_inf=U_inf,
        delta99_mm=delta99_mm,
        delta_star_mm=delta_star*1e3,
        theta_mm=theta*1e3,
        H=H
    )

# ============================================================
# LOAD + METRICS
# ============================================================
results = []          # [(label, df_profile)]
metrics_summary = []  # list of dicts with U_inf, δ99, δ*, θ, H

for label, path in FOLDERS.items():
    prof = load_profile(label, path)
    if prof is None:
        print(f"⚠️ No files in {path}")
        continue

    y_mm, U_mean, _ = prof
    m = compute_integral_params(y_mm, U_mean)

    df_profile = pd.DataFrame({"y_position_mm": y_mm, "U_mean_mps": U_mean})
    results.append((label, df_profile))
    metrics_summary.append({"Profile": label, **m})

# Pretty table for report
metrics_df = pd.DataFrame(metrics_summary, columns=["Profile","U_inf","delta99_mm","delta_star_mm","theta_mm","H"])
print("\n=== Boundary-layer integral parameters ===")
print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

# ============================================================
# PLOT 1 — Boundary-layer profiles with clean δ99 labels
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 5.2))

xmin, xmax = +1e9, -1e9
for label, dfp in results:
    c = COLORS.get(label, None)
    ax.plot(dfp["U_mean_mps"], dfp["y_position_mm"], "o-", ms=4, label=label, color=c)
    xmin = min(xmin, dfp["U_mean_mps"].min())
    xmax = max(xmax, dfp["U_mean_mps"].max())

for row in metrics_summary:
    y99 = float(row["delta99_mm"])
    prof = row["Profile"]
    c = COLORS.get(prof, "gray")
    ax.axhline(y99, color=c, ls="--", lw=1.0, alpha=0.9, zorder=1)

    y_offset = -0.4 if y99 < 15 else 0.0
    ax.annotate(
        f"δ₉₉ = {y99:.1f} mm",
        xy=(1.01, y99 + y_offset), xycoords=('axes fraction', 'data'),
        ha="left", va="center", color=c, fontsize=8, annotation_clip=False,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9)
    )

ax.set_xlim(xmin, xmax)
ax.set_xlabel("Velocity U [m/s]")
ax.set_ylabel("Wall-normal position y [mm]")
ax.set_title("Boundary-layer profiles (U vs y) with δ₉₉ markers")
ax.grid(True, alpha=0.35)
ax.legend(loc="upper left", frameon=True)
fig.subplots_adjust(right=0.86)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2 — Collapsed U/U∞ vs y/δ with 1/7th law
# ============================================================
def plot_powerlaw_collapsed(results, metrics_summary):
    meta = {row["Profile"]: (row["delta99_mm"], row["U_inf"]) for row in metrics_summary}
    plt.figure(figsize=(7.6, 5.0))
    for label, dfp in results:
        if label not in meta: continue
        delta99_mm, U_inf = meta[label]
        x = dfp["y_position_mm"].to_numpy() / delta99_mm
        y = dfp["U_mean_mps"].to_numpy() / U_inf
        plt.plot(x, y, "o", ms=3.6, alpha=0.9, label=label, color=COLORS.get(label))
    xi = np.linspace(0, 1.2, 400)
    plt.plot(xi, xi**(1/7), "k--", lw=2.0, label="$(y/\\delta)^{1/7}$")
    plt.xlim(0, 1.2); plt.ylim(0, 1.05)
    plt.xlabel(r"$y/\delta_{99}$"); plt.ylabel(r"$U/U_\infty$")
    plt.title("Collapsed profiles vs 1/7th power law")
    plt.grid(True, alpha=0.35)
    plt.legend(frameon=True, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()

plot_powerlaw_collapsed(results, metrics_summary)

# ============================================================
# PLOT 3 — Per-profile: U(y) vs 1/7th law TRUNCATED at U∞
# ============================================================
def plot_powerlaw_per_profile_truncated(results, metrics_summary):
    meta = {row["Profile"]: (row["delta99_mm"], row["U_inf"]) for row in metrics_summary}
    for label, dfp in results:
        if label not in meta: continue
        delta99_mm, U_inf = meta[label]
        y_mm  = dfp["y_position_mm"].to_numpy()
        U_meas= dfp["U_mean_mps"].to_numpy()
        # theoretical curve only up to δ99 (so U_theo ≤ U_inf)
        y_theo = np.linspace(0, delta99_mm, 350)
        U_theo = U_inf * (np.clip(y_theo / delta99_mm, 0, 1.0))**(1/7)
        # x-limits around measured velocities
        Umin, Umax = U_meas.min(), U_meas.max()
        pad = 0.05 * (Umax - Umin)
        xmin, xmax = Umin - pad, Umax + pad
        plt.figure(figsize=(7.0, 4.8))
        plt.plot(U_meas, y_mm, "o", ms=4, label="Measured", color=COLORS.get(label))
        plt.plot(U_theo, y_theo, "r--", lw=2.2, label="1/7 law fit")
        plt.axhline(delta99_mm, color="gray", ls="--", lw=1.0, alpha=0.6)
        plt.text(xmax, delta99_mm, f"  δ₉₉={delta99_mm:.1f} mm", va="center", ha="left",
                 color="gray", fontsize=8)
        plt.xlim(xmin, xmax); plt.ylim(0, max(y_mm)*1.05)
        plt.xlabel("Velocity U [m/s]"); plt.ylabel("y [mm]")
        plt.title(f"Profile vs 1/7th power law — {label}\n"
                  f"$U_\\infty$={U_inf:.2f} m/s,  $\\delta_{{99}}$={delta99_mm:.1f} mm")
        plt.grid(True, alpha=0.35); plt.legend()
        plt.tight_layout(); plt.show()

plot_powerlaw_per_profile_truncated(results, metrics_summary)
