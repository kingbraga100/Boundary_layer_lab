import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    (20, 25),
    (42, 47),
    (52, 56),
    (62, 67),
    (75, 80),
    (90, 95),
    (100, 105),
    (112, 118),
    (125, 130),
]

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

def fit_linear(x, y):
    a, b = np.polyfit(x, y, 1)
    yhat = a*x + b
    r2 = 1 - np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2)
    return a, b, r2

# ================================
# CALIBRATION READINGS
# ================================
df_valid = read_two_cols(FILE_VALIDYNE)
df_setra = read_two_cols(FILE_SETRA)
p_levels = np.arange(0, 11)

# 1️⃣ PLOT: Raw calibration voltage vs time
plt.figure(figsize=(10,4))
plt.plot(df_valid["time_s"], df_valid["voltage_V"], label="Validyne (static)", linewidth=1.5)
plt.plot(df_setra["time_s"], df_setra["voltage_V"], label="Setra (dynamic)", linewidth=1.5)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Voltage [V]", fontsize=14)
plt.title("Calibration recordings: Voltage vs Time", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# bin-average each plateau region
def bin_average(df, n=11):
    tmin, tmax = df["time_s"].min(), df["time_s"].max()
    bins = np.linspace(tmin, tmax, n+1)
    means = []
    for i in range(n):
        mask = (df["time_s"]>=bins[i]) & (df["time_s"]<bins[i+1])
        means.append(df.loc[mask,"voltage_V"].median())
    return np.array(means)

v_valid = bin_average(df_valid, 11)
v_setra = bin_average(df_setra, 11)
a_valid, b_valid, r2_valid = fit_linear(v_valid, p_levels)
a_setra, b_setra, r2_setra = fit_linear(v_setra, p_levels)

# 2️⃣ PLOT: Calibration curves (pressure vs voltage)
plt.figure(figsize=(8,6))
plt.scatter(v_valid, p_levels, color="tab:blue", s=60, label=f"Validyne data (R²={r2_valid:.4f})")
plt.scatter(v_setra, p_levels, color="tab:orange", s=60, label=f"Setra data (R²={r2_setra:.4f})")
vfit = np.linspace(min(v_valid.min(), v_setra.min()), max(v_valid.max(), v_setra.max()), 200)
plt.plot(vfit, a_valid*vfit + b_valid, "b--", linewidth=2, label="Validyne linear fit")
plt.plot(vfit, a_setra*vfit + b_setra, "r--", linewidth=2, label="Setra linear fit")
plt.xlabel("Voltage [V]", fontsize=14)
plt.ylabel("Pressure [mmH₂O]", fontsize=14)
plt.title("Pressure calibration: Voltage-to-Pressure fits", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

print(f"Validyne: P(mmH₂O) = 5.929716 * V   (offset removed)")
print(f"Setra:    P(mmH₂O) = 28.700525 * V  (offset removed)")

# ================================
# WIND-TUNNEL DATA
# ================================
dfw = read_wind_excel(FILE_WIND)
# Use only slope, no intercepts
a_valid, a_setra = 5.929716, 28.700525
dfw["p_dyn_mmH2O"]    = a_setra * dfw["setra_V"]
dfw["p_static_mmH2O"] = a_valid * dfw["validyne_V"]

# zero offset from first 2 s
mask0 = dfw["time_s"] <= (dfw["time_s"].iloc[0]+2)
dfw["p_dyn_mmH2O"]    -= dfw.loc[mask0,"p_dyn_mmH2O"].median()
dfw["p_static_mmH2O"] -= dfw.loc[mask0,"p_static_mmH2O"].median()

dfw["p_dyn_Pa"]    = dfw["p_dyn_mmH2O"]    * MMH2O_TO_PA
dfw["p_static_Pa"] = dfw["p_static_mmH2O"] * MMH2O_TO_PA
dfw["U_ms"]        = np.sqrt(2*G*np.clip(dfw["p_dyn_mmH2O"],0,None)/RHO)

# 3️⃣ PLOT: Wind-tunnel pressures vs time with plateaus
plt.figure(figsize=(10,5))
plt.plot(dfw["time_s"], dfw["p_static_Pa"], label="Static [Pa]", linewidth=1.5)
plt.plot(dfw["time_s"], dfw["p_dyn_Pa"], label="Dynamic [Pa]", linewidth=1.5)
for (t0, t1) in PLATEAU_WINDOWS:
    plt.axvspan(t0, t1, color="gray", alpha=0.25)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Pressure [Pa]", fontsize=14)
plt.title("Wind-tunnel pressures with selected plateau regions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# ================================
# FIXED PLATEAUS: Average each window
# ================================
rows, shades = [], []
for (t0, t1) in PLATEAU_WINDOWS:
    mask = (dfw["time_s"] >= t0) & (dfw["time_s"] <= t1)
    if mask.sum() < 5:
        continue
    sub = dfw.loc[mask]
    rows.append({
        "t_start_s": t0,
        "t_end_s": t1,
        "p_static_mmH2O": sub["p_static_mmH2O"].mean(),
        "p_dyn_mmH2O":    sub["p_dyn_mmH2O"].mean(),
        "U_ms":           sub["U_ms"].mean()
    })
    shades.append((t0, t1))

avg_df = pd.DataFrame(rows)

# Force the first plateau (20–25 s) to 0 velocity and pressure
if not avg_df.empty:
    avg_df.loc[0, ["p_static_mmH2O", "p_dyn_mmH2O", "U_ms"]] = 0.0

print("\nAveraged plateau values (first point forced to zero):")
print(avg_df.round(4))

# ================================
# 4TH ORDER POLYNOMIAL FIT (mmH2O)
# ================================
x_mmH2O = avg_df["p_static_mmH2O"].to_numpy()
y = avg_df["U_ms"].to_numpy()
xfit_mmH2O = np.linspace(x_mmH2O.min(), x_mmH2O.max(), 400)

# 4th order fit
n = 4
coeffs_mmH2O = np.polyfit(x_mmH2O, y, n)
yfit = np.polyval(coeffs_mmH2O, x_mmH2O)
yplot_mmH2O = np.polyval(coeffs_mmH2O, xfit_mmH2O)
r2 = 1 - np.sum((y - yfit)**2)/np.sum((y - np.mean(y))**2)
sigma = np.std(y - yfit)

# Create compact equation string for legend (mmH2O)
equation_mmH2O = f"U(P) = {coeffs_mmH2O[0]:.3e}P⁴ + {coeffs_mmH2O[1]:.3e}P³ + {coeffs_mmH2O[2]:.3e}P² + {coeffs_mmH2O[3]:.3f}P + {coeffs_mmH2O[4]:.3f}\nR² = {r2:.4f}"

# 4️⃣ PLOT: Velocity vs Static pressure with 4th order fit (mmH2O)
plt.figure(figsize=(10, 7))
plt.scatter(x_mmH2O, y, s=80, c="black", label="Experimental data")
plt.plot(xfit_mmH2O, yplot_mmH2O, 'r-', linewidth=3, label=equation_mmH2O)
plt.xlabel("Static pressure [mmH₂O]", fontsize=16)
plt.ylabel("Velocity [m/s]", fontsize=16)
plt.title("Velocity vs Static Pressure — 4th order polynomial calibration", fontsize=18)
plt.legend(fontsize=13, loc='upper left')
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# ================================
# 5TH PLOT: Velocity vs Static pressure in Pascals
# ================================
# Convert static pressure to Pascals
x_Pa = avg_df["p_static_mmH2O"].to_numpy() * MMH2O_TO_PA
xfit_Pa = np.linspace(x_Pa.min(), x_Pa.max(), 400)

# Fit 4th order polynomial to Pa data
coeffs_Pa = np.polyfit(x_Pa, y, n)
yplot_Pa = np.polyval(coeffs_Pa, xfit_Pa)
r2_Pa = 1 - np.sum((y - np.polyval(coeffs_Pa, x_Pa))**2)/np.sum((y - np.mean(y))**2)
sigma_Pa = np.std(y - np.polyval(coeffs_Pa, x_Pa))

# Create compact equation string for legend (Pa)
equation_Pa = f"U(P) = {coeffs_Pa[0]:.3e}P⁴ + {coeffs_Pa[1]:.3e}P³ + {coeffs_Pa[2]:.3e}P² + {coeffs_Pa[3]:.3f}P + {coeffs_Pa[4]:.3f}\nR² = {r2_Pa:.4f}"

plt.figure(figsize=(10, 7))
plt.scatter(x_Pa, y, s=80, c="black", label="Experimental data")
plt.plot(xfit_Pa, yplot_Pa, 'b-', linewidth=3, label=equation_Pa)
plt.xlabel("Static pressure [Pa]", fontsize=16)
plt.ylabel("Velocity [m/s]", fontsize=16)
plt.title("Velocity vs Static Pressure — 4th order polynomial calibration", fontsize=18)
plt.legend(fontsize=13, loc='upper left')
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# ================================
# 6TH PLOT: Combined plot with compact equations
# ================================
plt.figure(figsize=(16, 7))

# mmH2O plot
plt.subplot(1, 2, 1)
plt.scatter(x_mmH2O, y, s=80, c="black", label="Experimental data")
plt.plot(xfit_mmH2O, yplot_mmH2O, 'r-', linewidth=3, label=equation_mmH2O)
plt.xlabel("Static pressure [mmH₂O]", fontsize=16)
plt.ylabel("Velocity [m/s]", fontsize=16)
plt.title("Velocity vs Static Pressure (mmH₂O)", fontsize=18)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Pa plot
plt.subplot(1, 2, 2)
plt.scatter(x_Pa, y, s=80, c="black", label="Experimental data")
plt.plot(xfit_Pa, yplot_Pa, 'b-', linewidth=3, label=equation_Pa)
plt.xlabel("Static pressure [Pa]", fontsize=16)
plt.ylabel("Velocity [m/s]", fontsize=16)
plt.title("Velocity vs Static Pressure (Pa)", fontsize=18)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

# ================================
# PRINT FIT RESULTS (full precision for actual use)
# ================================
print(f"\n4th order polynomial fit (mmH2O):")
print(f"U(P) = {coeffs_mmH2O[0]:.6e}·P⁴ + {coeffs_mmH2O[1]:.6e}·P³ + {coeffs_mmH2O[2]:.6e}·P² + {coeffs_mmH2O[3]:.6f}·P + {coeffs_mmH2O[4]:.6f}")
print(f"Where P is in mmH₂O, R² = {r2:.4f}")

print(f"\n4th order polynomial fit (Pa):")
print(f"U(P) = {coeffs_Pa[0]:.6e}·P⁴ + {coeffs_Pa[1]:.6e}·P³ + {coeffs_Pa[2]:.6e}·P² + {coeffs_Pa[3]:.6f}·P + {coeffs_Pa[4]:.6f}")
print(f"Where P is in Pa, R² = {r2_Pa:.4f}")