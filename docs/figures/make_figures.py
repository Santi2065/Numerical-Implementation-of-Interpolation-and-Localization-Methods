"""Regenerates the figures and numbers shown in the README.

    pip install numpy scipy matplotlib
    python docs/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.interpolate import BarycentricInterpolator, CubicSpline, griddata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


def f1(x):
    return -0.4 * np.tanh(50 * x) + 0.6


def f2(x1, x2):  # as defined in MNyO_TP01.pdf
    return (0.75 * np.exp(-(9 * x1 - 2) ** 2 / 4 - (9 * x2 - 2) ** 2 / 4)
            + 0.75 * np.exp(-(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) ** 2 / 10)
            + 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4 - (9 * x2 - 3) ** 2 / 4)
            - 0.2 * np.exp(-(9 * x1 - 7) ** 2 / 4 - (9 * x2 - 3) ** 2 / 4))


def nodes(kind, n):
    return np.linspace(-1, 1, n) if kind == "equi" else np.polynomial.chebyshev.chebpts2(n)


def interp1(method, xs, ys, x):
    if method == "linear":
        return np.interp(x, xs, ys)
    if method == "spline":
        return CubicSpline(xs, ys)(x)
    return BarycentricInterpolator(xs, ys)(x)  # Lagrange polynomial, numerically stable form


xd = np.linspace(-1, 1, 2001)
METHODS = [("lagrange", "Lagrange"), ("spline", "Cubic spline"), ("linear", "Piecewise linear")]

# ---- Figure 1: Runge phenomenon, 15 nodes
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, kind, title in zip(axes, ("equi", "cheb"), ("(a) Equispaced nodes", "(b) Chebyshev nodes")):
    xs = nodes(kind, 15)
    ax.plot(xd, f1(xd), color="#1a1a1a", lw=1.6, label="$f_1(x)$")
    for (m, lab), c in zip(METHODS, C):
        ax.plot(xd, interp1(m, xs, f1(xs), xd), lw=1, color=c, ls="--" if m == "lagrange" else "-", label=lab)
    ax.plot(xs, f1(xs), "o", color="#1a1a1a", ms=3.2, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylim(-0.6, 1.8)
axes[0].set_ylabel("$p_{15}(x)$")
axes[1].legend(loc="upper right")
save(fig, "fig1-runge.svg")

# ---- Figure 2: MSE vs number of nodes (1D)
ns = np.arange(5, 100)
mse1 = {(m, k): [np.mean((interp1(m, nodes(k, n), f1(nodes(k, n)), xd) - f1(xd)) ** 2) for n in ns]
        for m, _ in METHODS for k in ("equi", "cheb")}
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for (m, lab), c in zip(METHODS, C):
    ax.semilogy(ns, mse1[m, "equi"], color=c, ls="--", lw=1, label=f"{lab}, equispaced")
    ax.semilogy(ns, mse1[m, "cheb"], color=c, lw=1.2, label=f"{lab}, Chebyshev")
ax.set_xlabel("number of nodes $n$")
ax.set_ylabel("MSE on $[-1, 1]$")
ax.set_ylim(1e-9, 1e3)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
save(fig, "fig2-mse-1d.svg")

# ---- Figure 3: 2D interpolation of f2
g = np.linspace(-1, 1, 120)
G1, G2 = np.meshgrid(g, g)
truth = f2(G1, G2)


def interp2(kind, method, n):
    v = nodes(kind, n)
    P1, P2 = np.meshgrid(v, v)
    return griddata((P1.ravel(), P2.ravel()), f2(P1, P2).ravel(), (G1, G2), method=method)


ns2 = np.arange(5, 61, 1)
mse2 = {(m, k): [np.nanmean((interp2(k, m, n) - truth) ** 2) for n in ns2]
        for m in ("linear", "cubic") for k in ("equi", "cheb")}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.35]})
ext = (-1, 1, -1, 1)
axes[0].imshow(truth, origin="lower", extent=ext, cmap="viridis")
axes[0].set_title("(a) $f_2(x_1, x_2)$")
err = np.abs(interp2("equi", "cubic", 10) - truth)
im = axes[1].imshow(err, origin="lower", extent=ext, cmap="magma_r")
v = nodes("equi", 10)
P1, P2 = np.meshgrid(v, v)
axes[1].plot(P1, P2, ".", color="#1a1a1a", ms=1.2)
axes[1].set_title("(b) $|$error$|$, bicubic, $10^2$ nodes")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
for ax in axes[:2]:
    ax.set_xlabel("$x_1$")
    ax.tick_params(top=False, right=False)
axes[0].set_ylabel("$x_2$")
for (m, lab), c in zip((("linear", "Bilinear"), ("cubic", "Bicubic")), C):
    axes[2].semilogy(ns2, mse2[m, "equi"], color=c, ls="--", lw=1, label=f"{lab}, equi.")
    axes[2].semilogy(ns2, mse2[m, "cheb"], color=c, lw=1.2, label=f"{lab}, Cheb.")
axes[2].set_title("(c) MSE vs. nodes per axis")
axes[2].set_xlabel("nodes per axis $n$")
axes[2].legend(loc="upper right")
fig.tight_layout()
save(fig, "fig3-2d.svg")

# ---- Figure 4: trilateration with Newton's method
sensors = np.loadtxt(ROOT / "sensor_positions.txt", delimiter=",", skiprows=1)[:, 1:]
meas = np.loadtxt(ROOT / "measurements.txt", delimiter=",", skiprows=1)
gt = np.loadtxt(ROOT / "trajectory.txt", delimiter=",", skiprows=1)


def newton(p, d, tol=1e-10, it=100):
    for k in range(it):
        F = np.sum((p - sensors) ** 2, axis=1) - d ** 2
        step = np.linalg.solve(2 * (p - sensors), -F)
        p = p + step
        if np.linalg.norm(step) < tol:
            return p, k + 1
    raise RuntimeError("Newton did not converge")


p, est, iters = np.zeros(3), [], []
for row in meas:
    p, k = newton(p, row[1:])
    est.append(p)
    iters.append(k)
est = np.array(est)
t = meas[:, 0]
spl = CubicSpline(t, est)
rec = spl(gt[:, 0])
err3 = np.linalg.norm(rec - gt[:, 1:], axis=1)
node_err = np.linalg.norm(est - np.array([np.interp(t, gt[:, 0], gt[:, j]) for j in (1, 2, 3)]).T, axis=1)

fig = plt.figure(figsize=(7.2, 2.9))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(*gt[:, 1:].T, color="#1a1a1a", lw=1.4, label="ground truth")
ax.plot(*rec.T, color=C[1], lw=1, ls="--", label="spline through Newton fixes")
ax.plot(*est.T, "o", color=C[0], ms=3, label="Newton fixes ($\\Delta t = 0.5$ s)")
ax.set_xlabel("$x$ [m]", labelpad=-4)
ax.set_ylabel("$y$ [m]", labelpad=-4)
ax.set_zlabel("$z$ [m]", labelpad=-4)
ax.tick_params(pad=-2, labelsize=7)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.fill = False
    a.pane.set_edgecolor("#bdbdbd")
    a._axinfo["grid"].update(color="#e6e6e6", linewidth=0.5)
ax.set_title("(a) Reconstructed trajectory", y=1.02)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(gt[:, 0], err3, color=C[1], lw=1)
ax2.semilogy(t, np.maximum(node_err, 1e-16), "o", color=C[0], ms=3)
ax2.set_xlabel("time $t$ [s]")
ax2.set_ylabel("position error [m]")
ax2.set_title("(b) Euclidean error vs. time")
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig4-trilateration.svg")

# ---- numbers for the README tables
print("| Method | Nodes | MSE (equispaced) | MSE (Chebyshev) |")
for m, lab in METHODS:
    for n in (15, 50):
        i = n - ns[0]
        print(f"| {lab} | {n} | {mse1[m, 'equi'][i]:.2e} | {mse1[m, 'cheb'][i]:.2e} |")
for m in ("linear", "cubic"):
    for n in (10, 30):
        i = n - ns2[0]
        print(f"2D {m} n={n}: equi {mse2[m, 'equi'][i]:.2e} cheb {mse2[m, 'cheb'][i]:.2e}")
print(f"Newton iterations per fix: mean {np.mean(iters):.1f}, max {max(iters)}")
print(f"spline trajectory error: mean {err3.mean():.3e} m, max {err3.max():.3e} m, RMSE {np.sqrt((err3**2).mean()):.3e} m")
print(f"fix error max {node_err.max():.2e} m")
