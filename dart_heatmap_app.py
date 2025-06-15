
"""
Streamlit app: dart_heatmap_polar_numbers.py

▸ Heat‑map of expected score for every aim point
▸ Sector numbers & ring outlines
▸ Gold ★ at optimal aim
▸ Dashed 1 σ and 2 σ ellipses showing spread when aiming at that point
▸ Sliders for along‑arm and cross‑arm σ up to 100 mm
▸ Arm‑angle slider is *board‑relative*: 0° points straight up through 20
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle

# ---------------- Board geometry (mm) ----------------
R_INNER_BULL, R_OUTER_BULL = 6.35, 15.9
R_TRIPLE_IN, R_TRIPLE_OUT = 99.0, 107.0
R_DOUBLE_IN, R_DOUBLE_OUT = 162.0, 170.0

SECTOR_VALUES = np.array([
     20, 1,18, 4,13, 6,10,15, 2,17,
      3,19, 7,16, 8,11,14, 9,12, 5
])

# How much we rotate the *display* so 20 is vertical
ROT_SHIFT = np.pi/2 - np.pi/20         # +81°

# ---------------- Scoring -----------------------------
def score_polar(r, th):
    """Return dart score for arrays `r` (mm) and `th` (rad)."""
    th_mod = (th % (2*np.pi))
    idx = (th_mod // (np.pi/10)).astype(int)       # 0–19 sector index
    base = SECTOR_VALUES[idx]

    score = np.zeros_like(r, dtype=np.int16)
    score[r <= R_INNER_BULL] = 50
    ring = (r > R_INNER_BULL) & (r <= R_OUTER_BULL)
    score[ring] = 25
    ring = (r > R_TRIPLE_IN) & (r <= R_TRIPLE_OUT)
    score[ring] = 3 * base[ring]
    ring = (r > R_DOUBLE_IN) & (r <= R_DOUBLE_OUT)
    score[ring] = 2 * base[ring]
    ring = (score == 0) & (r <= R_DOUBLE_OUT)
    score[ring] = base[ring]            # singles
    return score

# ---------------- EV grid -----------------------------
def ev_grid(sig_y, sig_x, theta_arm_board_deg,
            n_samples, r_grid=None, th_grid=None, seed=1234):
    """Expected score for every aim point on a (r,θ) grid.

    theta_arm_board_deg = 0 means *up through 20* on the board.
    """
    if r_grid is None:
        r_grid = np.arange(0.0, R_DOUBLE_IN, 4.0)
    if th_grid is None:
        th_grid = np.arange(0.0, 360.0, 4.0)

    # Aim mesh in CARTESIAN world co‑ords (before display rotation)
    r_mesh, th_mesh = np.meshgrid(r_grid, np.deg2rad(th_grid), indexing='ij')
    x0 = r_mesh * np.cos(th_mesh)
    y0 = r_mesh * np.sin(th_mesh)

    # Convert board‑relative arm angle → world angle
    world_arm_rad = np.deg2rad(theta_arm_board_deg) - ROT_SHIFT

    ux, uy = np.cos(world_arm_rad), np.sin(world_arm_rad)   # along‑arm
    vx, vy = -np.sin(world_arm_rad), np.cos(world_arm_rad)  # cross‑arm

    rng = np.random.default_rng(seed)
    eps_y = rng.normal(0, sig_y, (n_samples, 1))
    eps_x = rng.normal(0, sig_x, (n_samples, 1))

    xs = x0.ravel()[None, :] + eps_y * ux + eps_x * vx
    ys = y0.ravel()[None, :] + eps_y * uy + eps_x * vy

    ev = score_polar(np.hypot(xs, ys), np.arctan2(ys, xs)).mean(0)
    ev = ev.reshape(r_grid.size, th_grid.size)
    return ev, r_grid, th_grid, world_arm_rad

# ---------------- Drawing helpers --------------------
def boundary_angle(idx):
    """Angle (rad) of sector boundary *after* display rotation."""
    return idx * np.pi/10 + ROT_SHIFT

def draw_board(ax):
    # Rings
    for r in [R_INNER_BULL, R_OUTER_BULL,
              R_TRIPLE_IN, R_TRIPLE_OUT,
              R_DOUBLE_IN, R_DOUBLE_OUT]:
        ax.add_patch(Circle((0, 0), r,
                            transform=ax.transData._b,
                            fill=False, lw=0.6, color='k', zorder=3))
    # Sector lines
    for k in range(20):
        ang = boundary_angle(k)
        ax.plot([ang, ang], [0, R_DOUBLE_OUT],
                lw=0.4, color='k', zorder=3)

    # Numbers
    offset = R_DOUBLE_OUT + 12
    for k, val in enumerate(SECTOR_VALUES):
        ang_mid = boundary_angle(k) + np.pi/20
        ax.text(ang_mid, offset, str(val),
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                zorder=4)

    ax.set_ylim(0, offset + 20)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

# ---------------- Streamlit UI -----------------------
st.set_page_config(layout="wide")
st.title("Dart‑board Expected‑Value Heat‑map with Spread Contours")

cols = st.columns(4)
with cols[0]:
    sig_y = st.slider("σₑ (along‑arm, mm)", 1, 100, 8, 1)
with cols[1]:
    sig_x = st.slider("σₓ (cross‑arm, mm)", 1, 100, 6, 1)
with cols[2]:
    theta_arm_board = st.slider(
        "Arm angle θ (deg, 0 = straight up)",
        0, 359, 0, 1,
        help="Measured clockwise from the 12 o’clock (20) direction.")
with cols[3]:
    N = st.number_input("Monte‑Carlo darts", 500, 20000, 4000, 500)

if st.button("Compute", type="primary"):
    with st.spinner("Simulating darts…"):
        ev, r_mm, th_deg, world_arm_rad = ev_grid(
            sig_y, sig_x, theta_arm_board, int(N)
        )

    # Build mesh edges in display co‑ords
    r_step = r_mm[1] - r_mm[0]
    th_step_deg = th_deg[1] - th_deg[0]
    r_edges = np.concatenate(([0], r_mm + r_step/2))
    th_edges = np.deg2rad(
        np.concatenate(
            (th_deg - th_step_deg/2, [th_deg[-1] + th_step_deg/2])
        )
    )
    th_edges_rot = (th_edges + ROT_SHIFT) % (2*np.pi)
    Th_e, R_e = np.meshgrid(th_edges_rot, r_edges, indexing='ij')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    pcm = ax.pcolormesh(Th_e, R_e, ev.T,
                        cmap='plasma', shading='auto',
                        alpha=0.75, zorder=2)

    draw_board(ax)

    # Optimal aim
    idx_best = np.unravel_index(np.argmax(ev), ev.shape)
    r_best = r_mm[idx_best[0]]
    th_best_world = np.deg2rad(th_deg[idx_best[1]])
    th_best_plot = (th_best_world + ROT_SHIFT) % (2*np.pi)
    ax.plot(th_best_plot, r_best, marker='*', markersize=18,
            color='gold', mec='black', mew=0.8, zorder=5)

    # 1σ & 2σ ellipses
    x0 = r_best * np.cos(th_best_world)
    y0 = r_best * np.sin(th_best_world)
    ux, uy = np.cos(world_arm_rad), np.sin(world_arm_rad)
    vx, vy = -np.sin(world_arm_rad), np.cos(world_arm_rad)

    t = np.linspace(0, 2*np.pi, 400)
    for n, lw in [(1, 1.1), (2, 0.9)]:
        dx = n * (sig_y * np.cos(t) * ux + sig_x * np.sin(t) * vx)
        dy = n * (sig_y * np.cos(t) * uy + sig_x * np.sin(t) * vy)
        x = x0 + dx
        y = y0 + dy
        r = np.hypot(x, y)
        th = (np.arctan2(y, x) + ROT_SHIFT) % (2*np.pi)
        ax.plot(th, r, color='k', lw=lw, ls='--', zorder=4)

    fig.colorbar(pcm, ax=ax, shrink=0.76, label="Expected points")
    ax.set_title("Expected score — ★ optimal aim  (dashed = 1σ, 2σ)")
    st.pyplot(fig)
