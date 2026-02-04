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
        "U_true_mean": U_true.mean()
    })

df = pd.DataFrame(rows).sort_values("Run").reset_index(drop=True)
df.loc[df["Run"] == 0, "U_true_mean"] = 0.0  # enforce baseline zero

# -------------------------------------------------
# Remove unreliable low-speed points (runs 1 & 2)
# -------------------------------------------------
bad_runs = [1, 2,3]  # skip 2nd and 3rd points (too close to baseline)
df_clean = df[~df["Run"].isin(bad_runs)].copy().reset_index(drop=True)
print(f"Removed runs {bad_runs} (unstable near-zero region)\n")

# -------------------------------------------------
# Anchored hot-wire fit: U = Σ c_k (V - V0)^k, k=1..d
# Guarantees U(V0)=0 exactly
# -------------------------------------------------
V0 = df.loc[df["Run"] == 0, "V_hw_mean"].iloc[0]
x = df_clean["V_hw_mean"].to_numpy()
y = df_clean["U_true_mean"].to_numpy()
dV = x - V0

# Design matrix without constant term
def design(delta_v, degree):
    return np.column_stack([delta_v**k for k in range(1, degree+1)])

X = design(dV, ANCHOR_DEGREE)

# Fit using least squares
coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

# Predictions (U(V0)=0 by construction)
y_pred = X @ coeffs

# Metrics
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
sigma = float(np.std(y - y_pred))
print(f"Anchored hot-wire fit (degree {ANCHOR_DEGREE}) in ΔV = V - V0:")
print("U =", " + ".join([f"{c:+.3e}*(V-V0)^{k+1}" for k, c in enumerate(coeffs)]))
print(f"V0 = {V0:.6f} V  (U(V0)=0 enforced)")
print(f"R² = {r2:.4f},  σ = {sigma:.4f} m/s")

# -------------------------------------------------
# Plot
# -------------------------------------------------
V_plot = np.linspace(df["V_hw_mean"].min(), df["V_hw_mean"].max(), 400)
dV_plot = V_plot - V0
X_plot = design(dV_plot, ANCHOR_DEGREE)
U_plot = X_plot @ coeffs

plt.figure(figsize=(8,6))
plt.scatter(x, y, s=50, color="k", label="Used points (calibrated)")
plt.plot(V_plot, U_plot, "r--", lw=2,
         label=f"Anchored ΔV^{ANCHOR_DEGREE} fit\nR²={r2:.3f}, σ={sigma:.3f} m/s")
plt.axvline(V0, color="gray", ls=":", lw=1)
plt.xlabel("Hot-wire voltage V [V]")
plt.ylabel("Velocity U [m/s]")
plt.title("Hot-wire calibration: anchored polynomial in ΔV ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

