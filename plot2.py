import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load exported data
# -------------------------
slice_df = pd.read_csv("slice_z0.csv")

# -------------------------
# Rebuild grid
# -------------------------
x_vals = np.sort(slice_df["x"].unique())
y_vals = np.sort(slice_df["y"].unique())

X, Y = np.meshgrid(x_vals, y_vals)

V_num = slice_df.pivot(index="y", columns="x", values="V_num").values

L = np.max(np.abs(x_vals))

# -------------------------
# Same analytical contour levels
# -------------------------
k = 1.0 / (4.0 * np.pi)   # q=1, eps0=1
candidate_levels = np.array([0.10, 0.13, 0.17, 0.23, 0.32, 0.45])

levels = []
for lev in candidate_levels:
    r = k / lev
    if r < 0.9 * L:
        levels.append(lev)

levels = np.array(levels)

# -------------------------
# Plot: analytical solid circles + sampled numerical dots
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Analytical equipotential circles as solid lines
theta = np.linspace(0, 2 * np.pi, 600)
for lev in levels:
    r = k / lev
    xa = r * np.cos(theta)
    ya = r * np.sin(theta)
    ax.plot(xa, ya, linewidth=2.0, label=f"Analytical V={lev:.2f}")

# Numerical sample points:
# take points from numerical contours and plot them as dots
angles = np.deg2rad([0, 30, 45, 60, 90, 120, 135, 150, 180,
                     210, 225, 240, 270, 300, 315, 330])

for lev in levels:
    r = k / lev

    xs = []
    ys = []

    for ang in angles:
        x_target = r * np.cos(ang)
        y_target = r * np.sin(ang)

        # nearest grid point
        i = np.argmin(np.abs(x_vals - x_target))
        j = np.argmin(np.abs(y_vals - y_target))

        xs.append(x_vals[i])
        ys.append(y_vals[j])

    ax.plot(xs, ys, "o", markersize=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Analytical equipotentials (solid) with sampled numerical points (dots)")
ax.set_aspect("equal")
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.grid(True, alpha=0.3)

handles, labels = ax.get_legend_handles_labels()
if handles:
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h)
            l2.append(l)
            seen.add(l)
    ax.legend(h2, l2, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("equipotential_comparison_points.png", dpi=200)
plt.show()