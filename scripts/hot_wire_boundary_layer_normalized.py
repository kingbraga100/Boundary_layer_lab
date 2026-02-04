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

# Fluid/constants
RHO = 1.20
NU  = 1.50e-5
MU  = RHO * NU
KAPPA = 0.41
B_LOG = 5.0
BRADSHAW_CONST = 16.24  # 16.24, as requested

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
    return pd.DataFrame(data, columns=["time_s", "p_dyn", "V_hw", "p_static"])

def sorted_point_files(folder_path):
    all_files = [f for f in folder_path.glob("point*") if f.suffix.lower() in [".csv", ".txt"]]
    out = []
    for f in all_files:
        digits = ''.join(ch for ch in f.stem if ch.isdigit())
        if digits:
            out.append((int(digits), f))
    return sorted(out, key=lambda x: x[0])

def load_profile(label, path):
    files = sorted_point_files(path)
    if not files:
        return None
    V0 = V0_MAP[label]
    y_mm, U_mean = [], []
    for pos, file in files:
        df = read_point_file(file)
        V_avg = df["V_hw"].mean()
        U_avg = U_from_hotwire(V_avg, V0)
        y_mm.append(pos * 0.1)  # 0.1-mm steps
        U_mean.append(U_avg)
    y_mm = np.asarray(y_mm); U_mean = np.asarray(U_mean)
    idx = np.argsort(y_mm)
    y_mm, U_mean = y_mm[idx], U_mean[idx]
    U_inf = float(np.mean(U_mean[-min(5, len(U_mean)):] ))
    return y_mm, U_mean, U_inf

# ============================================================
# τw METHODS
# ============================================================
def tauw_viscous(y_mm, U):
    """
    Viscous sublayer slope using wall point (0,0) and first five near-wall points.
    Returns:
      tau_w, slope_dUdy, (y_fit[m], U_fit[m/s])
    """
    y = np.asarray(y_mm)*1e-3+0.0001  # Convert to meters
    U = np.asarray(U)
    
    if len(y) < 5:
        return np.nan, np.nan, (np.nan, np.nan)
    
    # Include wall point (0,0) and first five measured points
    y_wall = 0.0
    U_wall = 0.0
    
    # Use wall point and first five measured points for fitting
    n_points = min(5, len(y))
    y_fit = np.concatenate([[y_wall], y[:n_points]])
    U_fit = np.concatenate([[U_wall], U[:n_points]])
    
    m, c = np.polyfit(y_fit, U_fit, 1)  # U = m*y + c
    tau_w = MU * m
    return tau_w, m, (y_fit, U_fit)

def tauw_log_slope_ln1to2(y_mm, U, U_inf):
    """
    Log-layer slope from points with ln(y_mm) between 1 and 2.
    This corresponds to y between e¹ (2.718 mm) and e² (7.389 mm).
    Cf = 2*(slope/(2.5*U_inf))^2  (slope is dU/d(ln y_mm))
    Returns:
      tau_w, Cf, slope, (y_used[m], U_used, slope, intercept)
    """
    y_mm = np.asarray(y_mm)
    U    = np.asarray(U)

    # Filter out zero or negative values before taking log
    valid_mask = (y_mm > 0) & np.isfinite(U)
    y_mm_valid = y_mm[valid_mask]
    U_valid = U[valid_mask]
    
    if len(y_mm_valid) == 0:
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]), np.nan, np.nan)

    # Calculate ln(y_mm) for valid points
    ln_y = np.log(y_mm_valid)
    
    # Select points where ln(y_mm) is between 1 and 2
    mask = (ln_y >= 1.0) & (ln_y <= 2.0)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]), np.nan, np.nan)

    X = ln_y[mask]      # ln(y) with y in mm
    Y = U_valid[mask]
    slope, intercept = np.polyfit(X, Y, 1)
    Cf = 2.0 * (slope / (2.5*U_inf))**2
    tau_w = 0.5*RHO*U_inf**2 * Cf
    return tau_w, Cf, slope, (y_mm_valid[mask]*1e-3, U_valid[mask], slope, intercept)

def tauw_bradshaw(y_mm, U, U_inf):
    """
    Bradshaw intersection with U/U_inf = (16.24*100)*nu / (y[m] * U_inf).
    This should give intersection at y+ = 100 where U/U∞ = 16.24.
    Returns:
      tau_w, Cf, (y_star[m], U_star[m/s]), (y[m], meas, target)
    """
    y = np.asarray(y_mm)*1e-3
    U = np.asarray(U)
    mask = (y > 0) & np.isfinite(U)
    y, U = y[mask], U[mask]
    meas   = U / U_inf
    
    # DEBUG: Check what the theory curve should be
    print(f"\nBradshaw debug for U_inf={U_inf:.2f} m/s:")
    print("y[mm]\tMeasured U/U∞\tTarget U/U∞")
    for i in range(min(10, len(y))):
        target_val = (BRADSHAW_CONST*100.0)*NU/(y[i]*U_inf)
        print(f"{y[i]*1e3:.1f}\t{meas[i]:.3f}\t\t{target_val:.3f}")
    
    target = (BRADSHAW_CONST*100.0)*NU/(y*U_inf)

    diff = meas - target
    sgn  = np.sign(diff)
    idx  = np.where(sgn[:-1]*sgn[1:] < 0)[0]
    
    if len(idx) == 0:
        print("No intersection found!")
        return np.nan, np.nan, (np.nan, np.nan), (y, meas, target)

    i = idx[0]
    y0, y1 = y[i], y[i+1]
    d0, d1 = diff[i], diff[i+1]
    y_star = y0 - d0*(y1 - y0)/(d1 - d0)
    U_star = np.interp(y_star, y, U)
    
    # Calculate y+ and U+ at intersection for verification
    urU = (U_star/U_inf)/BRADSHAW_CONST
    Cf  = 2*(urU**2)
    tau_w = 0.5*RHO*U_inf**2 * Cf
    u_tau = np.sqrt(tau_w/RHO)
    y_plus_star = y_star * u_tau / NU
    
    print(f"Intersection: y*={y_star*1e3:.2f} mm, U*/U∞={U_star/U_inf:.3f}")
    print(f"Calculated: y+={y_plus_star:.1f}, should be 100")
    print(f"Cf={Cf:.5f}, u_tau={u_tau:.3f} m/s")
    
    return tau_w, Cf, (y_star, U_star), (y, meas, target)

# ============================================================
# MAIN LOOP
# ============================================================
rows = []
profiles = {}
for label, path in FOLDERS.items():
    prof = load_profile(label, path)
    if prof is None:
        continue
    y_mm, U_mean, U_inf = prof
    profiles[label] = (y_mm, U_mean, U_inf)

    tau_v, slope_v, data_v = tauw_viscous(y_mm, U_mean)
    tau_l, Cf_l, slope_l, data_l = tauw_log_slope_ln1to2(y_mm, U_mean, U_inf)
    tau_b, Cf_b, (y_star, U_star), data_b = tauw_bradshaw(y_mm, U_mean, U_inf)
    Cf_v = 2*tau_v/(RHO*U_inf**2) if np.isfinite(tau_v) else np.nan

    rows.append([label, U_inf, tau_v, Cf_v, tau_l, Cf_l, tau_b, Cf_b, y_star, U_star])
    # stash method data for plotting
    profiles[label] += (data_v, data_l, data_b)

df = pd.DataFrame(rows, columns=[
    "Profile","U_inf","tauw_visc","Cf_visc","tauw_log","Cf_log","tauw_brad","Cf_brad","y_star_m","U_star_mps"
])
pd.set_option("display.float_format", lambda x: f"{x:,.5f}")
print("\n=== Wall shear stress & Cf ===")
print(df.to_string(index=False))

# ============================================================
# PLOTS
# ============================================================
def plot_viscous_fit(y_mm, U, data, label, U_inf):
    """Velocity on x-axis, y[mm] on y-axis with wall point (0,0) included"""
    y_fit, U_fit = data
    
    # Create extended line for plotting
    m, c = np.polyfit(y_fit, U_fit, 1)
    y_line = np.linspace(0, max(y_fit), 200)
    U_line = m*y_line + c

    plt.figure(figsize=(6,4))
    
    # Plot measured points (excluding wall)
    plt.plot(U[1:], y_mm[1:], "bo", label="Measured")
    
    # Plot wall point separately
    plt.plot(0, 0, "ks", markersize=6, label="Wall (0,0)")
    
    # Plot fit line
    plt.plot(U_line, y_line*1e3, "r-", lw=2.5, label=f"Linear fit slope={m:.1f}")
   
    
    plt.xlabel("Velocity U [m/s]"); plt.ylabel("y [mm]")
    
    # Set x-axis limits to start near first velocity - 1 m/s
    first_velocity = min(U)
    x_min = max(0, first_velocity - 1.0)  # Don't go below 0
    plt.xlim(x_min, None)
    
    plt.title(f"Viscous sublayer fit — {label}")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()

def plot_log_fit(y_mm, U, data, label, U_inf):
    """
    Show the measured data vs ln(y_mm) and the fit using ln(y_mm) between 1 and 2.
    Simplified: slope only in legend, no annotation.
    """
    y_used_m, U_used, slope, intercept = data
    if len(U_used) == 0 or not np.isfinite(slope):
        return

    # All points for context (x = ln(y_mm) with y in mm)
    # Filter out zero or negative y values first
    valid_mask = y_mm > 0
    y_mm_valid = y_mm[valid_mask]
    U_valid = U[valid_mask]
    
    X_all = np.log(y_mm_valid)
    plt.figure(figsize=(6,4))
    plt.plot(X_all, U_valid, "bo", label="Measured data")

    # Fit segment limits in ln(mm) space (ln(y) between 1 and 2)
    x0 = 1.0
    x1 = 2.0

    # Stop the fit line when U reaches U_inf
    x_end = (U_inf - intercept)/slope
    x_line_hi = min(x1, x_end) if slope > 0 else max(x1, x_end)
    x_line = np.linspace(x0, x_line_hi, 100)
    U_line = slope*x_line + intercept

    plt.plot(x_line, U_line, "r-", lw=2, label=f"Log-fit slope={slope:.2f}")
    
    plt.xlabel("ln(y) [mm]"); plt.ylabel("U [m/s]")
    plt.title(f"Log-slope method — {label}")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()

def plot_bradshaw(y_mm, U, U_inf, data, label, y_star, U_star):
    """Fixed: Ensure theory curve matches exactly at y=100"""
    y, meas, target = data
    
    plt.figure(figsize=(6,4))
    plt.plot(y*1e3, meas, "bo-", label="Measured U/U∞")
    plt.plot(y*1e3, target, "m-", lw=1.8, label="Bradshaw theory")
    plt.scatter([y_star*1e3], [U_star/U_inf], color="red", marker="x", s=80, label="Intersection")

    # Annotate higher up - placed in upper part of plot
    x_annot = max(y)*1e3 * 0.3  # 70% across x-axis
    y_annot = 1.3  # Fixed high position
    
    plt.text(x_annot, y_annot,
             f"Intersection:\ny*={y_star*1e3:.1f} mm\nU/U∞={U_star/U_inf:.2f}",
             color="red", ha="left", va="center",
             bbox=dict(facecolor="white", edgecolor="red", alpha=0.8))
    
    # Axis markers
    plt.axvline(y_star*1e3, color="red", ls="--", lw=0.8, alpha=0.6)
    plt.axhline(U_star/U_inf, color="red", ls="--", lw=0.8, alpha=0.6)

    plt.ylim(0, 2)
    plt.xlabel("y [mm]"); plt.ylabel("U/U∞")
    plt.title(f"Bradshaw method — {label}")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()

# Generate per-profile diagnostic plots
for label, (y, U, U_inf, data_v, data_l, data_b) in profiles.items():
    plot_viscous_fit(y, U, data_v, label, U_inf)
    plot_log_fit(y, U, data_l, label, U_inf)
    y_star = df.loc[df["Profile"] == label, "y_star_m"].values[0]
    U_star = df.loc[df["Profile"] == label, "U_star_mps"].values[0]
    plot_bradshaw(y, U, U_inf, data_b, label, y_star, U_star)

# ============================================================
# LAW OF THE WALL (unchanged visuals)
# ============================================================
def plot_law_of_wall(df, profiles, method):
    method_map = {"Viscous":"tauw_visc","Log-law":"tauw_log","Bradshaw":"tauw_brad"}
    tau_col = method_map[method]
    color_map = {"Downstream 5 m/s":"tab:blue","Downstream 10 m/s":"tab:orange",
                 "Upstream 5 m/s":"tab:green","Upstream 10 m/s":"tab:red"}
    plt.figure(figsize=(7.5,5.5))
    for label, (y, U, U_inf, *_) in profiles.items():
        tau_w = df.loc[df["Profile"]==label, tau_col].values[0]
        if not np.isfinite(tau_w): continue
        u_tau = np.sqrt(tau_w/RHO)
        y_plus = (y*1e-3)*u_tau/NU
        u_plus = U/u_tau
        plt.scatter(y_plus, u_plus, s=16, alpha=0.8, color=color_map[label], label=label)
    # Theory segments
    yp_visc = np.linspace(0,5,100)
    yp_buff = np.linspace(5,30,100)
    yp_log  = np.linspace(30,4000,300)
    plt.plot(yp_visc, yp_visc, "k--", lw=1.2, label="Viscous: u+=y+ (0<y+<5)")
    plt.plot(yp_buff, -3.05+5*np.log(yp_buff), "gray", ls=":", lw=1.5, label="Buffer: -3.05+5ln(y+)")
    plt.plot(yp_log,  5+2.44*np.log(yp_log), "r-", lw=1.4, label="Log: 5.5+2.5ln(y+)")
    plt.xscale("log"); plt.xlim(0.5,4000); plt.ylim(0,35)
    plt.xlabel(r"$y^+$"); plt.ylabel(r"$u^+$")
    plt.title(f"Law of the Wall — All profiles ({method})")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(frameon=True, fontsize=8, loc="lower right")
    plt.tight_layout(); plt.show()

plot_law_of_wall(df, profiles, "Viscous")
plot_law_of_wall(df, profiles, "Log-law")
plot_law_of_wall(df, profiles, "Bradshaw")