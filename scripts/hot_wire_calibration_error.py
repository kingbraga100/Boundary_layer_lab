import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# Paths & constants
# -------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTWIRE_DIR = ROOT / "data" / "hot_wire_calibration"

A_VALID = 5.929716  # Validyne sensitivity [mmH2O/V], baseline removed with run 0

# Your static→U (order 4): x in mmH2O, returns U [m/s]
STATIC_TO_U_COEFFS = [-7.103e-04,  2.793e-02, -3.932e-01,  2.945e+00,  5.375e-01]

ANCHOR_DEGREE = 4    # hot-wire fit degree in ΔV (try 3 or 4)

# -------------------------------------------------
# Helper: read calibration files
# -------------------------------------------------
def read_tab_file(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = [ln.strip().replace(",", ".") for ln in f if ln.strip()]
    data = [list(map(float, ln.split())) for ln in lines]
    return pd.DataFrame(data, columns=["time_s", "p_dyn", "V_hw", "V_static_validyne"])

# -------------------------------------------------
# UNCERTAINTY CALCULATION FOR INDIVIDUAL POINTS
# -------------------------------------------------
def calculate_point_uncertainty(V_hw, U_true, coeffs, V0, sigma_fit):
    """
    Calculate uncertainty for individual calibration points
    Returns single uncertainty value
    """
    dV = V_hw - V0
    
    # 1. Fit uncertainty (constant)
    uncertainty_fit = 2 * sigma_fit
    
    # 2. Reference velocity uncertainty (pressure-based)
    delta_p = 0.1 * 9.81  # Pa
    rho = 1.225
    U_safe = max(U_true, 0.5)  # Avoid division by zero
    uncertainty_ref_pressure = (1 / (rho * U_safe)) * delta_p
    uncertainty_ref_constant = 0.1
    uncertainty_ref = np.sqrt(uncertainty_ref_pressure**2 + uncertainty_ref_constant**2)
    
    # 3. Voltage measurement uncertainty
    dUdV = 0.0
    for k, c in enumerate(coeffs):
        power = k + 1
        dUdV += c * power * (dV)**(power - 1)
    
    delta_V = 0.01
    uncertainty_voltage = np.abs(dUdV) * delta_V
    
    # Total uncertainty for this point
    total_uncertainty = np.sqrt(uncertainty_fit**2 + uncertainty_ref**2 + uncertainty_voltage**2)
    
    return total_uncertainty

# -------------------------------------------------
# Load runs and compute "true" U from static
# -------------------------------------------------
files = [f for f in HOTWIRE_DIR.glob("*.csv") if f.stem.isdigit()]
files = sorted(files, key=lambda f: int(f.stem))
if not files:
    raise RuntimeError("No numeric calibration files found.")

# Baseline (run 0): set P_ref = 0 by zeroing the Validyne voltage
df0 = read_tab_file(files[0])
V_ref = df0["V_static_validyne"].mean()
print(f"Baseline (run 0): V_ref = {V_ref:.6f} V  → P_ref := 0 mmH2O\n")

rows = []
for f in files:
    run_id = int(f.stem)
    df = read_tab_file(f)

    # Zero Validyne voltage using baseline and convert to mmH2O
    V_stat_corr = df["V_static_validyne"] - V_ref
    p_static_mmh2o = A_VALID * V_stat_corr

    # "True" velocity from your static→U polynomial
    U_true = np.polyval(STATIC_TO_U_COEFFS, p_static_mmh2o)

    rows.append({
        "Run": run_id,
        "V_hw_mean": df["V_hw"].mean(),
        "U_true_mean": U_true.mean(),
    })

df = pd.DataFrame(rows).sort_values("Run").reset_index(drop=True)
df.loc[df["Run"] == 0, "U_true_mean"] = 0.0  # enforce baseline zero

# -------------------------------------------------
# Remove unreliable low-speed points
# -------------------------------------------------
bad_runs = [1, 2, 3]  # skip unstable near-zero points
df_clean = df[~df["Run"].isin(bad_runs)].copy().reset_index(drop=True)
print(f"Removed runs {bad_runs} (unstable near-zero region)\n")

# -------------------------------------------------
# Anchored hot-wire fit
# -------------------------------------------------
V0 = df.loc[df["Run"] == 0, "V_hw_mean"].iloc[0]
x = df_clean["V_hw_mean"].to_numpy()
y = df_clean["U_true_mean"].to_numpy()
dV = x - V0

def design(delta_v, degree):
    return np.column_stack([delta_v**k for k in range(1, degree+1)])

X = design(dV, ANCHOR_DEGREE)
coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ coeffs

# Metrics
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
sigma = float(np.std(y - y_pred))
print(f"Anchored hot-wire fit (degree {ANCHOR_DEGREE}) in ΔV = V - V0:")
print(f"V0 = {V0:.6f} V  (U(V0)=0 enforced)")
print(f"R² = {r2:.4f},  σ = {sigma:.4f} m/s")

# -------------------------------------------------
# Calculate uncertainties for individual points
# -------------------------------------------------
point_uncertainties = []
for i, row in df_clean.iterrows():
    uncertainty = calculate_point_uncertainty(
        row["V_hw_mean"], row["U_true_mean"], coeffs, V0, sigma
    )
    point_uncertainties.append(uncertainty)

point_uncertainties = np.array(point_uncertainties)

# -------------------------------------------------
# Create smooth calibration curve (only positive velocities)
# -------------------------------------------------
V_plot = np.linspace(df["V_hw_mean"].min(), df["V_hw_mean"].max(), 400)
dV_plot = V_plot - V0
X_plot = design(dV_plot, ANCHOR_DEGREE)
U_plot = X_plot @ coeffs

# Filter only positive velocities for plotting
positive_mask = U_plot >= 0
V_plot_positive = V_plot[positive_mask]
U_plot_positive = U_plot[positive_mask]

# Calculate curve uncertainty for positive region only
curve_uncertainties = []
for V_point, U_point in zip(V_plot_positive, U_plot_positive):
    uncertainty = calculate_point_uncertainty(V_point, max(U_point, 0.1), coeffs, V0, sigma)
    curve_uncertainties.append(uncertainty)

curve_uncertainties = np.array(curve_uncertainties)

# -------------------------------------------------
# Clean single plot with error bars and positive velocities only
# -------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot calibration curve with uncertainty band (only positive velocities)
plt.plot(V_plot_positive, U_plot_positive, "r-", lw=2.5, 
         label=f'Hot-wire calibration (R² = {r2:.3f})')

# Uncertainty band for the curve
plt.fill_between(V_plot_positive, 
                 U_plot_positive - curve_uncertainties, 
                 U_plot_positive + curve_uncertainties, 
                 alpha=0.3, color='red', label='95% confidence interval')

# Individual calibration points with error bars
plt.errorbar(df_clean["V_hw_mean"], df_clean["U_true_mean"], 
             yerr=point_uncertainties, 
             fmt='o', color='black', markersize=6, capsize=4, capthick=2,
             label='Calibration points ±95% CI')

# Zero velocity point (V0)
plt.axvline(V0, color='gray', linestyle=':', alpha=0.7, label=f'$V_0$ = {V0:.3f} V')

plt.xlabel("Hot-wire voltage V [V]", fontsize=12)
plt.ylabel("Velocity U [m/s]", fontsize=12)
plt.title("Hot-wire Calibration with Uncertainty", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Set limits to show only positive velocities
plt.ylim(bottom=0)
# Set x-axis to start slightly before V0 for better visualization
plt.xlim(left=V0-0.1, right=V_plot_positive.max() + 0.1)

plt.tight_layout()
plt.show()

# Print uncertainty summary
print(f"\n--- UNCERTAINTY SUMMARY ---")
print(f"Calibration fit uncertainty: ±{2*sigma:.3f} m/s (95% CI)")
print(f"Point uncertainties range: ±{point_uncertainties.min():.3f} to ±{point_uncertainties.max():.3f} m/s")
print(f"Average point uncertainty: ±{point_uncertainties.mean():.3f} m/s")

# Show individual point uncertainties
print(f"\nIndividual point uncertainties:")
for i, row in df_clean.iterrows():
    print(f"Run {int(row['Run']):2d}: V = {row['V_hw_mean']:.3f} V, U = {row['U_true_mean']:5.2f} m/s, Uncertainty = ±{point_uncertainties[i]:.3f} m/s")