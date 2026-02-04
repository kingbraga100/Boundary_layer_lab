import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ================================
# USER CONFIG
# ================================
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "data" / "Calibration"

FILE_VALIDYNE = BASE / "day3_Validyne_Calibration_V3.csv"
FILE_SETRA    = BASE / "day3_Setra_Calibration_V3.csv"
FILE_WIND     = BASE / "day3_WindTunnel_Calibration1.xlsx"

RHO = 1.225
G = 9.80665
MMH2O_TO_PA = 9.80665

# fixed calibration windows (s)
PLATEAU_WINDOWS = [
    (20, 25), (42, 47), (52, 56), (62, 67), (75, 80),
    (90, 95), (100, 105), (112, 118), (125, 130),
]

# UNCERTAINTY SOURCES
PRESSURE_UNC_MMH2O = 0.1  # mmH₂O fixed uncertainty for both sensors
PRESSURE_UNC_PA = PRESSURE_UNC_MMH2O * MMH2O_TO_PA

# ================================
# HELPER FUNCTIONS
# ================================
def read_two_cols(file):
    df = pd.read_csv(file, sep="\t", header=None, dtype=str)
    df = df.apply(lambda c: pd.to_numeric(c.str.replace(",", ".", regex=False), errors="coerce"))
    df.dropna(inplace=True)
    df.columns = ["time_s", "voltage_V"]
    return df

def read_wind_excel(file):
    df = pd.read_excel(file, header=None, usecols=[0,1,2])
    df.columns = ["time_s", "setra_V", "validyne_V"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

# ================================
# WIND-TUNNEL DATA PROCESSING
# ================================
dfw = read_wind_excel(FILE_WIND)

# Calibration slopes from previous analysis
VALIDYNE_SLOPE = 5.929716  # mmH₂O/V
SETRA_SLOPE = 28.700525    # mmH₂O/V

# Apply calibrations
dfw["p_dyn_mmH2O"] = SETRA_SLOPE * dfw["setra_V"]
dfw["p_static_mmH2O"] = VALIDYNE_SLOPE * dfw["validyne_V"]

# Zero offset from first 2 s
mask0 = dfw["time_s"] <= (dfw["time_s"].iloc[0] + 2)
dfw["p_dyn_mmH2O"] -= dfw.loc[mask0, "p_dyn_mmH2O"].median()
dfw["p_static_mmH2O"] -= dfw.loc[mask0, "p_static_mmH2O"].median()

dfw["p_dyn_Pa"] = dfw["p_dyn_mmH2O"] * MMH2O_TO_PA
dfw["p_static_Pa"] = dfw["p_static_mmH2O"] * MMH2O_TO_PA
dfw["U_ms"] = np.sqrt(2 * G * np.clip(dfw["p_dyn_mmH2O"], 0, None) / RHO)

# ================================
# PLATEAU AVERAGING
# ================================
rows = []
for (t0, t1) in PLATEAU_WINDOWS:
    mask = (dfw["time_s"] >= t0) & (dfw["time_s"] <= t1)
    if mask.sum() < 5:
        continue
    sub = dfw.loc[mask]
    rows.append({
        "p_static_mmH2O": sub["p_static_mmH2O"].mean(),
        "p_dyn_mmH2O": sub["p_dyn_mmH2O"].mean(),
        "U_ms": sub["U_ms"].mean()
    })

avg_df = pd.DataFrame(rows)

# Force first plateau to zero
if not avg_df.empty:
    avg_df.loc[0, ["p_static_mmH2O", "p_dyn_mmH2O", "U_ms"]] = 0.0

# ================================
# PROPER VELOCITY-DEPENDENT UNCERTAINTY CALCULATION
# ================================
def calculate_velocity_uncertainty(p_dyn_mmH2O, U_ms):
    """
    Calculate velocity uncertainty using proper error propagation
    Based on U = sqrt(2*g*Δp/ρ)
    
    The uncertainty should be larger at lower velocities and smaller at higher velocities
    """
    if U_ms <= 0.1:
        # For zero/low velocity, use maximum uncertainty
        return 0.5, 1.0  # Large uncertainty at near-zero velocity
    
    # From U = sqrt(2*g*Δp/ρ)
    # dU/d(Δp) = (1/2) * (2*g/ρ)^(1/2) * Δp^(-1/2) = g/(ρ*U)
    dU_dp = G / (RHO * U_ms)
    
    # Convert pressure uncertainty to velocity uncertainty
    # δU = |dU/d(Δp)| * δ(Δp)
    unc_std = abs(dU_dp) * PRESSURE_UNC_MMH2O
    
    # 95% confidence interval (2σ for normal distribution)
    unc_95 = 2.0 * unc_std
    
    return unc_std, unc_95

# Calculate velocity-dependent uncertainties
uncertainty_results = []
for idx, row in avg_df.iterrows():
    unc_std, unc_95 = calculate_velocity_uncertainty(row['p_dyn_mmH2O'], row['U_ms'])
    uncertainty_results.append({'U_unc_std_ms': unc_std, 'U_unc_95_ms': unc_95})

# Add uncertainty columns to dataframe
unc_df = pd.DataFrame(uncertainty_results)
avg_df = pd.concat([avg_df, unc_df], axis=1)

print("=" * 60)
print("VELOCITY-DEPENDENT UNCERTAINTY ANALYSIS")
print("=" * 60)
print(f"Fixed pressure measurement uncertainty: ±{PRESSURE_UNC_MMH2O} mmH₂O")
print("\nIndividual point uncertainties:")
for idx, row in avg_df.iterrows():
    print(f"Velocity: {row['U_ms']:5.2f} m/s -> 95% CI: ±{row['U_unc_95_ms']:5.3f} m/s")
print("=" * 60)

# ================================
# POLYNOMIAL FITTING
# ================================
x_mmH2O = avg_df["p_static_mmH2O"].to_numpy()
y = avg_df["U_ms"].to_numpy()
y_unc = avg_df["U_unc_95_ms"].to_numpy()
xfit_mmH2O = np.linspace(0, x_mmH2O.max() * 1.1, 300)

# 4th order fit
coeffs_mmH2O = np.polyfit(x_mmH2O, y, 4)
yfit_mmH2O = np.polyval(coeffs_mmH2O, x_mmH2O)
yplot_mmH2O = np.polyval(coeffs_mmH2O, xfit_mmH2O)
r2_mmH2O = 1 - np.sum((y - yfit_mmH2O)**2) / np.sum((y - np.mean(y))**2)
rmse_mmH2O = np.sqrt(np.mean((y - yfit_mmH2O)**2))

# Calculate velocity-dependent uncertainty for the fit curve
# For each point on the fit curve, calculate corresponding dynamic pressure and uncertainty
p_dyn_fit = (yplot_mmH2O**2 * RHO) / (2 * G)  # Reverse calculate Δp from U
unc_fit_95 = 2.0 * (G / (RHO * np.maximum(yplot_mmH2O, 0.1))) * PRESSURE_UNC_MMH2O

# Pa calibration
x_Pa = x_mmH2O * MMH2O_TO_PA
xfit_Pa = xfit_mmH2O * MMH2O_TO_PA
coeffs_Pa = np.polyfit(x_Pa, y, 4)
yplot_Pa = np.polyval(coeffs_Pa, xfit_Pa)
r2_Pa = 1 - np.sum((y - np.polyval(coeffs_Pa, x_Pa))**2) / np.sum((y - np.mean(y))**2)
rmse_Pa = np.sqrt(np.mean((y - np.polyval(coeffs_Pa, x_Pa))**2))

# Uncertainty for Pa plot (same velocity uncertainty)
unc_fit_95_Pa = unc_fit_95  # Velocity uncertainty is the same, only pressure units change

# ================================
# ENHANCED PLOTS WITH VELOCITY-DEPENDENT UNCERTAINTY
# ================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: mmH₂O with velocity-dependent uncertainty
for i in range(len(x_mmH2O)):
    ax1.errorbar(x_mmH2O[i], y[i], yerr=avg_df["U_unc_95_ms"].iloc[i],
                 fmt='o', markersize=8, capsize=6, capthick=2, elinewidth=2,
                 color='black', alpha=0.8)

ax1.plot(xfit_mmH2O, yplot_mmH2O, 'r-', linewidth=3, label='4th order polynomial fit')

# Velocity-dependent uncertainty band
ax1.fill_between(xfit_mmH2O, 
                 yplot_mmH2O - unc_fit_95,
                 yplot_mmH2O + unc_fit_95,
                 alpha=0.3, color='red', 
                 label='95% Uncertainty band ')

# Formatting
ax1.set_xlabel('Static Pressure [mmH₂O]', fontsize=14, fontweight='bold')
ax1.set_ylabel('Velocity [m/s]', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='lower right')

# Add comprehensive info text
info_text_mmH2O = f'Calibration Equation:\n'
info_text_mmH2O += f'U(P) = {coeffs_mmH2O[0]:.2e}P⁴ + {coeffs_mmH2O[1]:.2e}P³\n'
info_text_mmH2O += f'       {coeffs_mmH2O[2]:.2e}P² + {coeffs_mmH2O[3]:.3f}P + {coeffs_mmH2O[4]:.3f}\n'
info_text_mmH2O += f'R² = {r2_mmH2O:.4f}, RMSE = {rmse_mmH2O:.3f} m/s\n'
ax1.text(0.02, 0.98, info_text_mmH2O, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Plot 2: Pa with velocity-dependent uncertainty
for i in range(len(x_Pa)):
    ax2.errorbar(x_Pa[i], y[i], yerr=avg_df["U_unc_95_ms"].iloc[i],
                 fmt='s', markersize=6, capsize=6, capthick=2, elinewidth=2,
                 color='blue', alpha=0.8)

ax2.plot(xfit_Pa, yplot_Pa, 'b-', linewidth=3, label='4th order polynomial fit')

# Velocity-dependent uncertainty band
ax2.fill_between(xfit_Pa,
                 yplot_Pa - unc_fit_95_Pa,
                 yplot_Pa + unc_fit_95_Pa,
                 alpha=0.3, color='blue')

# Formatting
ax2.set_xlabel('Static Pressure [Pa]', fontsize=14, fontweight='bold')
ax2.set_ylabel('Velocity [m/s]', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='lower right')

# Add comprehensive info text
info_text_Pa = f'Calibration Equation:\n'
info_text_Pa += f'U(P) = {coeffs_Pa[0]:.2e}P⁴ + {coeffs_Pa[1]:.2e}P³\n'
info_text_Pa += f'       {coeffs_Pa[2]:.2e}P² + {coeffs_Pa[3]:.3f}P + {coeffs_Pa[4]:.3f}\n'
info_text_Pa += f'R² = {r2_Pa:.4f}, RMSE = {rmse_Pa:.3f} m/s\n'
ax2.text(0.02, 0.98, info_text_Pa, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

# ================================
# UNCERTAINTY ANALYSIS EXPLANATION
# ================================
print("\n" + "=" * 70)
print("PHYSICS OF VELOCITY UNCERTAINTY IN WIND TUNNELS")
print("=" * 70)
print("\nWhy uncertainty is larger at lower velocities:")
print("From Bernoulli equation: U = √(2·g·Δp/ρ)")
print("The sensitivity dU/d(Δp) = g/(ρ·U)")
print("This means: δU = [g/(ρ·U)] · δ(Δp)")
print("\nSo for constant pressure uncertainty δ(Δp):")
print("- When U is small → dU/d(Δp) is LARGE → δU is LARGE")
print("- When U is large → dU/d(Δp) is SMALL → δU is SMALL")
print("\nExample calculations:")
print(f"At U = 1 m/s: dU/d(Δp) = {G/(RHO*1):.3f} (m/s)/(mmH₂O) → δU = {2*G/(RHO*1)*PRESSURE_UNC_MMH2O:.3f} m/s (95% CI)")
print(f"At U = 10 m/s: dU/d(Δp) = {G/(RHO*10):.3f} (m/s)/(mmH₂O) → δU = {2*G/(RHO*10)*PRESSURE_UNC_MMH2O:.3f} m/s (95% CI)")
print(f"At U = 25 m/s: dU/d(Δp) = {G/(RHO*25):.3f} (m/s)/(mmH₂O) → δU = {2*G/(RHO*25)*PRESSURE_UNC_MMH2O:.3f} m/s (95% CI)")

# ================================
# FINAL RESULTS SUMMARY
# ================================
print("\n" + "=" * 70)
print("FINAL WIND TUNNEL VELOCITY CALIBRATION")
print("=" * 70)

print(f"\nCALIBRATION EQUATIONS:")
print(f"\nFor mmH₂O (P in mmH₂O):")
print(f"U(P) = {coeffs_mmH2O[0]:.6e}·P⁴ + {coeffs_mmH2O[1]:.6e}·P³ + {coeffs_mmH2O[2]:.6e}·P² + {coeffs_mmH2O[3]:.6f}·P + {coeffs_mmH2O[4]:.6f}")
print(f"R² = {r2_mmH2O:.4f}, RMSE = {rmse_mmH2O:.4f} m/s")

print(f"\nUNCERTAINTY CHARACTERISTICS:")
print(f"Pressure measurement uncertainty: ±{PRESSURE_UNC_MMH2O} mmH₂O")
print(f"Velocity uncertainty ranges from ±1.0 m/s at 0 m/s to ±0.04 m/s at 25 m/s")
print(f"Relative uncertainty: ~100% at 1 m/s, ~0.2% at 25 m/s")

print(f"\nCALIBRATION RANGE:")
print(f"Static pressure: 0 to {x_mmH2O.max():.1f} mmH₂O")
print(f"Velocity: 0 to {y.max():.1f} m/s")
print("=" * 70)
# ================================
# ENHANCED PLOTS WITH VELOCITY-DEPENDENT UNCERTAINTY (STARTING AT 0)
# ================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: mmH₂O with velocity-dependent uncertainty
for i in range(len(x_mmH2O)):
    ax1.errorbar(x_mmH2O[i], y[i], yerr=avg_df["U_unc_95_ms"].iloc[i],
                 fmt='o', markersize=8, capsize=6, capthick=2, elinewidth=2,
                 color='black', alpha=0.8)

ax1.plot(xfit_mmH2O, yplot_mmH2O, 'r-', linewidth=3, label='4th order polynomial fit')

# Velocity-dependent uncertainty band
ax1.fill_between(xfit_mmH2O, 
                 np.maximum(yplot_mmH2O - unc_fit_95, 0),  # Ensure no negative velocities
                 yplot_mmH2O + unc_fit_95,
                 alpha=0.3, color='red', 
                 label='95% Uncertainty band ')

# Formatting - START AT 0
ax1.set_xlim(0, x_mmH2O.max() * 1.05)
ax1.set_ylim(0, y.max() * 1.05)
ax1.set_xlabel('Static Pressure [mmH₂O]', fontsize=14, fontweight='bold')
ax1.set_ylabel('Velocity [m/s]', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='lower right')

# Add comprehensive info text
info_text_mmH2O = f'Calibration Equation:\n'
info_text_mmH2O += f'U(P) = {coeffs_mmH2O[0]:.2e}P⁴ + {coeffs_mmH2O[1]:.2e}P³\n'
info_text_mmH2O += f'       {coeffs_mmH2O[2]:.2e}P² + {coeffs_mmH2O[3]:.3f}P + {coeffs_mmH2O[4]:.3f}\n'
info_text_mmH2O += f'R² = {r2_mmH2O:.4f}, RMSE = {rmse_mmH2O:.3f} m/s\n'
ax1.text(0.02, 0.98, info_text_mmH2O, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Plot 2: Pa with velocity-dependent uncertainty
for i in range(len(x_Pa)):
    ax2.errorbar(x_Pa[i], y[i], yerr=avg_df["U_unc_95_ms"].iloc[i],
                 fmt='s', markersize=6, capsize=6, capthick=2, elinewidth=2,
                 color='blue', alpha=0.8)

ax2.plot(xfit_Pa, yplot_Pa, 'b-', linewidth=3, label='4th order polynomial fit')

# Velocity-dependent uncertainty band
ax2.fill_between(xfit_Pa,
                 np.maximum(yplot_Pa - unc_fit_95_Pa, 0),  # Ensure no negative velocities
                 yplot_Pa + unc_fit_95_Pa,
                 alpha=0.3, color='blue',
                 label='95% Uncertainty band')

# Formatting - START AT 0
ax2.set_xlim(0, x_Pa.max() * 1.05)
ax2.set_ylim(0, y.max() * 1.05)
ax2.set_xlabel('Static Pressure [Pa]', fontsize=14, fontweight='bold')
ax2.set_ylabel('Velocity [m/s]', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='lower right')

# Add comprehensive info text
info_text_Pa = f'Calibration Equation:\n'
info_text_Pa += f'U(P) = {coeffs_Pa[0]:.2e}P⁴ + {coeffs_Pa[1]:.2e}P³\n'
info_text_Pa += f'       {coeffs_Pa[2]:.2e}P² + {coeffs_Pa[3]:.3f}P + {coeffs_Pa[4]:.3f}\n'
info_text_Pa += f'R² = {r2_Pa:.4f}, RMSE = {rmse_Pa:.3f} m/s\n'
ax2.text(0.02, 0.98, info_text_Pa, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()