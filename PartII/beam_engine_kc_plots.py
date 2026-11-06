"""
ME 4133 Part II - Grasshopper beam engine: position and kinematic-coefficient (KC) plots.

What this script does
    1. Solves the linkage position in closed form for every crank angle theta6 = 0..360 deg.
    2. Computes the first-order KCs (h = d theta / d theta6, f = d position / d theta6).
    3. Computes the second-order KCs (h' = d2 theta / d theta6^2, f' = d2 position / d theta6^2).
    4. Plots link angles, piston-pin position, and both KC orders (six figures, saved as PNG).

Conventions
    Origin at the radius-link pivot O_B, x to the right, y up, lengths in inches.
    Every angle is measured counter-clockwise from +x.
    Link 2 = radius link O_B->C, 3 = beam (D-C-E, theta3 points C->E), 4 = rocking pillar O_A->E,
    5 = connecting rod S->C, 6 = crank G->S (the input).

Equations are the same as Appendix A of the report and the Excel workbook.
Requires: numpy, matplotlib.   Run:  python beam_engine_kc_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Geometry (Fusion assembly user parameters, inches)
# ---------------------------------------------------------------------------
r_OBC = 0.3047          # radius link, O_B to C
r_CD = 0.2915           # beam, C to piston pin D
r_CE = 1.0 - r_CD       # beam, C to E (beam D-E is 1 in long and D, C, E are collinear)
r_OAE = 0.8599          # rocking pillar, O_A to E
r_SC = 0.8633           # connecting rod, S to C
r_GS = 0.1392           # crank, G to S
Gx, Gy = 0.2802, -0.8443      # crank pivot G
OAx, OAy = 1.0258, -0.8625    # rocking-pillar pivot O_A

# Crank angle sweep: one full revolution in 0.5 deg steps
theta6_deg = np.arange(0.0, 360.0 + 0.5, 0.5)
th6 = np.radians(theta6_deg)


# ---------------------------------------------------------------------------
# 2. Position solution (closed form, circle intersection)
# ---------------------------------------------------------------------------
def circle_intersection(Px, Py, rP, Qx, Qy, rQ, sign=+1):
    """Point X that is rP from P and rQ from Q.
    a = distance from P along the line P->Q, h = perpendicular offset.
    sign = +1 picks the assembly branch used in the CAD model."""
    dx, dy = Qx - Px, Qy - Py
    L = np.hypot(dx, dy)
    a = (rP**2 - rQ**2 + L**2) / (2 * L)
    h = np.sqrt(rP**2 - a**2)
    Xx = Px + (a * dx - sign * h * dy) / L
    Xy = Py + (a * dy + sign * h * dx) / L
    return Xx, Xy


# Crank pin S is known directly from the input angle
xS = Gx + r_GS * np.cos(th6)
yS = Gy + r_GS * np.sin(th6)

# Loop A: C is r_OBC from O_B (the origin) and r_SC from S
xC, yC = circle_intersection(0.0, 0.0, r_OBC, xS, yS, r_SC)
th2 = np.arctan2(yC, xC)              # radius link angle
th5 = np.arctan2(yC - yS, xC - xS)    # connecting rod angle (S -> C)

# Loop B: E is r_CE from C and r_OAE from O_A
xE, yE = circle_intersection(xC, yC, r_CE, OAx, OAy, r_OAE)
th3 = np.arctan2(yE - yC, xE - xC)    # beam angle (C -> E)
th4 = np.arctan2(yE - OAy, xE - OAx)  # pillar angle (O_A -> E)

# Piston pin D is on the beam, on the other side of C from E
xD = r_OBC * np.cos(th2) - r_CD * np.cos(th3)
yD = r_OBC * np.sin(th2) - r_CD * np.sin(th3)

# Check: both vector loops must close (residual ~1e-16)
resA = np.hypot(Gx + r_GS*np.cos(th6) + r_SC*np.cos(th5) - r_OBC*np.cos(th2),
                Gy + r_GS*np.sin(th6) + r_SC*np.sin(th5) - r_OBC*np.sin(th2))
resB = np.hypot(r_OBC*np.cos(th2) + r_CE*np.cos(th3) - r_OAE*np.cos(th4) - OAx,
                r_OBC*np.sin(th2) + r_CE*np.sin(th3) - r_OAE*np.sin(th4) - OAy)
print(f"Max loop-closure residual: {max(resA.max(), resB.max()):.1e} in")

# ---------------------------------------------------------------------------
# 3. First-order KCs
#    Differentiate each loop with respect to theta6, multiply by e^(-i*theta)
#    of one unknown link, and keep the real part. Each formula isolates one KC.
# ---------------------------------------------------------------------------
s, c = np.sin, np.cos
h2 = r_GS * s(th6 - th5) / (r_OBC * s(th2 - th5))
h5 = r_GS * s(th6 - th2) / (r_SC * s(th2 - th5))
h3 = -r_OBC * h2 * s(th2 - th4) / (r_CE * s(th3 - th4))
h4 = r_OBC * h2 * s(th2 - th3) / (r_OAE * s(th4 - th3))

# Piston pin: differentiate x_D and y_D above with the chain rule
f_xD = -r_OBC * h2 * s(th2) + r_CD * h3 * s(th3)
f_yD = r_OBC * h2 * c(th2) - r_CD * h3 * c(th3)

# ---------------------------------------------------------------------------
# 4. Second-order KCs
#    Differentiate the first-order loop equations again (theta6'' = 0) and
#    apply the same e^(-i*theta) step.
# ---------------------------------------------------------------------------
h2p = (r_GS * c(th6 - th5) + r_SC * h5**2 - r_OBC * h2**2 * c(th2 - th5)) / (r_OBC * s(th2 - th5))
h5p = (r_OBC * h2**2 - r_GS * c(th6 - th2) - r_SC * h5**2 * c(th5 - th2)) / (r_SC * s(th5 - th2))
h3p = (r_OAE * h4**2 - r_OBC * h2p * s(th2 - th4) - r_OBC * h2**2 * c(th2 - th4)
       - r_CE * h3**2 * c(th3 - th4)) / (r_CE * s(th3 - th4))
h4p = (r_OBC * h2p * s(th2 - th3) + r_OBC * h2**2 * c(th2 - th3) + r_CE * h3**2
       - r_OAE * h4**2 * c(th4 - th3)) / (r_OAE * s(th4 - th3))

# Piston pin: product rule on each term of f_xD and f_yD
f_xDp = -r_OBC * (h2p * s(th2) + h2**2 * c(th2)) + r_CD * (h3p * s(th3) + h3**2 * c(th3))
f_yDp = r_OBC * (h2p * c(th2) - h2**2 * s(th2)) - r_CD * (h3p * c(th3) - h3**2 * s(th3))

# Validation: analytic KCs vs a numerical derivative of the curves above
d = np.radians(0.5)
err1 = np.max(np.abs(np.gradient(np.unwrap(th2), d)[1:-1] - h2[1:-1]))
err2 = np.max(np.abs(np.gradient(h2, d)[1:-1] - h2p[1:-1]))
print(f"h2 vs numerical derivative: max difference {err1:.1e}")
print(f"h2' vs numerical derivative of h2: max difference {err2:.1e}")

# Key results (printed for the report tables)
print(f"Piston stroke: {np.ptp(yD):.4f} in   side drift of D: {np.ptp(xD):.4f} in")

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # colorblind-checked series colors
STYLES = ["-", "--", "-.", ":"]                          # line styles so the plots also read in grayscale
plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.color": "#e4e3de", "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
    "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight",
})
XTICKS = np.arange(0, 361, 45)


def kc_plot(curves, labels, ylabel, title, filename):
    """One axis of up to four curves vs crank angle, legend below the plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (y, lab) in enumerate(zip(curves, labels)):
        ax.plot(theta6_deg, y, color=COLORS[i], ls=STYLES[i], label=lab)
    ax.axhline(0, color="#9a9a94", lw=0.8)
    ax.set_xlim(0, 360); ax.set_xticks(XTICKS)
    ax.set_xlabel("Crank angle θ6 (deg)"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(ncol=len(curves), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(filename); plt.close(fig)


# Link angles: four panels, since each link rocks over a different range
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
panels = [(th2, "θ2 radius link"), (th3, "θ3 beam"), (th4, "θ4 rocking pillar"), (th5, "θ5 connecting rod")]
for i, (ax, (ang, lab)) in enumerate(zip(axs.flat, panels)):
    ax.plot(theta6_deg, np.degrees(ang), color=COLORS[i], ls=STYLES[i])
    ax.set_title(lab, loc="left"); ax.set_ylabel("deg")
    ax.set_xlim(0, 360); ax.set_xticks(XTICKS)
for ax in axs[1]:
    ax.set_xlabel("Crank angle θ6 (deg)")
fig.suptitle("Link angles over one crank revolution", x=0.02, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig("B2_link_angles.png"); plt.close(fig)

# Piston pin D: vertical stroke and horizontal drift have very different scales, so two panels
fig, axs = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
axs[0].plot(theta6_deg, yD, color=COLORS[0])
axs[0].set_ylabel("y_D (in)"); axs[0].set_title(f"Piston pin D: vertical position (stroke {np.ptp(yD):.3f} in)", loc="left")
axs[1].plot(theta6_deg, xD, color=COLORS[1], ls="--")
axs[1].set_ylabel("x_D (in)"); axs[1].set_title(f"Piston pin D: horizontal drift ({np.ptp(xD):.4f} in total)", loc="left")
for ax in axs:
    ax.axhline(0, color="#9a9a94", lw=0.8); ax.set_xlim(0, 360); ax.set_xticks(XTICKS)
axs[1].set_xlabel("Crank angle θ6 (deg)")
fig.tight_layout(); fig.savefig("B3_piston_position.png"); plt.close(fig)

kc_plot([h2, h3, h4, h5], ["h2", "h3", "h4", "h5"], "rad/rad",
        "First-order angular KCs", "B4_KC1_angular.png")
kc_plot([f_yD, f_xD], ["f_yD (vertical)", "f_xD (horizontal)"], "in/rad",
        "First-order KCs of piston pin D", "B5_KC1_piston.png")
kc_plot([h2p, h3p, h4p, h5p], ["h2'", "h3'", "h4'", "h5'"], "rad/rad²",
        "Second-order angular KCs", "B6_KC2_angular.png")
kc_plot([f_yDp, f_xDp], ["f_yD' (vertical)", "f_xD' (horizontal)"], "in/rad²",
        "Second-order KCs of piston pin D", "B7_KC2_piston.png")

print("Saved: B2_link_angles.png, B3_piston_position.png, B4_KC1_angular.png, "
      "B5_KC1_piston.png, B6_KC2_angular.png, B7_KC2_piston.png")
