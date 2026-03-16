import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load exported data
# -------------------------
slice_df = pd.read_csv("slice_z0.csv")
cmp_df = pd.read_csv("xaxis_compare.csv")

# -------------------------
# Rebuild grid
# -------------------------
x_vals = np.sort(slice_df["x"].unique())
y_vals = np.sort(slice_df["y"].unique())

nx = len(x_vals)
ny = len(y_vals)

X, Y = np.meshgrid(x_vals, y_vals)

# Pivot into 2D arrays
V_num = slice_df.pivot(index="y", columns="x", values="V_num").values
V_ana = slice_df.pivot(index="y", columns="x", values="V_ana").values

# Domain size
L = np.max(np.abs(x_vals))

# -------------------------
# Choose contour levels
# Avoid the very high values near the center
# -------------------------
k = 1.0 / (4.0 * np.pi)   # because q=1, eps0=1 in the Rust code
candidate_levels = np.array([0.10, 0.13, 0.17, 0.23, 0.32, 0.45])

# Keep only circles that fit comfortably inside the box
valid_levels = []
for lev in candidate_levels:
    r = k / lev
    if r < 0.9 * L:
        valid_levels.append(lev)

levels = np.array(valid_levels)

# -------------------------
# Plot 1: equipotential contours + analytical circles
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Numerical contours
cs = ax.contour(X, Y, V_num, levels=levels, linewidths=1.8)
ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")

# Analytical circles at the same potential levels
theta = np.linspace(0, 2 * np.pi, 600)
for lev in levels:
    r = k / lev
    xa = r * np.cos(theta)
    ya = r * np.sin(theta)
    ax.plot(xa, ya, "--", linewidth=1.5, label=f"Analytical V={lev:.2f}")

# Mark some sample points on those circles
angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
for lev in levels:
    r = k / lev
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)
    ax.plot(xs, ys, "o", markersize=4)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("z=0 slice: numerical equipotentials (solid) vs analytical circles (dashed)")
ax.set_aspect("equal")
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.grid(True, alpha=0.3)

# Keep legend compact
handles, labels = ax.get_legend_handles_labels()
if handles:
    # remove duplicates
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h)
            l2.append(l)
            seen.add(l)
    ax.legend(h2, l2, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("equipotential_comparison.png", dpi=200)
plt.show()

# -------------------------
# Plot 2: x-axis verification
# -------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

cmp_df = cmp_df.sort_values("x")
ax2.plot(cmp_df["x"], cmp_df["V_ana"], "-", linewidth=2, label="Analytical")
ax2.plot(cmp_df["x"], cmp_df["V_num"], "o", markersize=4, label="Numerical (SOR)")

ax2.set_xlabel("x  (with y=0, z=0)")
ax2.set_ylabel("Potential V")
ax2.set_title("Numerical vs analytical potential along the x-axis")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("xaxis_verification.png", dpi=200)
plt.show()

# -------------------------
# Print a quick error summary
# -------------------------
max_rel_err = cmp_df["rel_err"].max()
rms_rel_err = np.sqrt(np.mean(cmp_df["rel_err"]**2))

print(f"Max relative error = {max_rel_err:.6e}")
print(f"RMS relative error = {rms_rel_err:.6e}")