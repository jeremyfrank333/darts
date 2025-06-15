
"""
Streamlit app: dart_heatmap_polar_numbers.py

▸ Expected‑value heat‑map over a labelled dart board
▸ Gold ★ marks optimal aim
▸ Dashed ellipses (1 σ, 2 σ) show landing spread
▸ New: dashed radial line along the arm direction through the star
▸ Sliders:
    • Timing spread σₜ  (along‑arm, mm)
    • Lateral spread σₗ (across‑arm, mm)
    • Arm angle θ (0° = straight up through 20, clockwise positive)
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

# Rotate display so 20 is at 12 o'clock
ROT_SHIFT = np.pi / 2 - np.pi / 20    # 81°

# ---------------- Scoring -----------------------------
def score_polar(r, th):
    th_mod = np.mod(th, 2 * np.pi)
    idx = (th_mod // (np.pi / 10)).astype(int)
    base = SECTOR_VALUES[idx]

    score = np.zeros_like(r, dtype=np.int16)
    score[r <= R_INNER_BULL] = 50
    mask = (r > R_INNER_BULL) & (r <= R_OUTER_BULL)
    score[mask] = 25
    mask = (r > R_TRIPLE_IN) & (r <= R_TRIPLE_OUT)
    score[mask] = 3 * base[mask]
    mask = (r > R_DOUBLE_IN) & (r <= R_DOUBLE_OUT)
    score[mask] = 2 * base[mask]
    mask = (score == 0) & (r <= R_DOUBLE_OUT)
    score[mask] = base[mask]
    return score

# ---------------- EV grid -----------------------------
def ev_grid(sig_timing, sig_lateral, theta_board_deg,
            n_samples, r_grid=None, th_grid=None, seed=1234):
    if r_grid is None:
        r_grid = np.arange(0.0, R_DOUBLE_IN, 4.0)
    if th_grid is None:
        th_grid = np.arange(0.0, 360.0, 4.0)

    # Aim mesh in world coords
    r_mesh, th_mesh = np.meshgrid(r_grid,
                                  np.deg2rad(th_grid),
                                  indexing='ij')
    x0 = r_mesh * np.cos(th_mesh)
    y0 = r_mesh * np.sin(th_mesh)

    # Convert slider angle to world angle:
    # add 90° (π/2) then subtract display rotation
    world_arm_rad = np.deg2rad(theta_board_deg) + (np.pi/2 - ROT_SHIFT)

    ux, uy = np.cos(world_arm_rad), np.sin(world_arm_rad)    # along‑arm
    vx, vy = -np.sin(world_arm_rad), np.cos(world_arm_rad)   # across‑arm

    rng = np.random.default_rng(seed)
    eps_t = rng.normal(0, sig_timing, (n_samples, 1))
    eps_l = rng.normal(0, sig_lateral, (n_samples, 1))

    xs = x0.ravel()[None, :] + eps_t * ux + eps_l * vx
    ys = y0.ravel()[None, :] + eps_t * uy + eps_l * vy

    ev = score_polar(np.hypot(xs, ys), np.arctan2(ys, xs)).mean(0)
    ev = ev.reshape(r_grid.size, th_grid.size)
    return ev, r_grid, th_grid, world_arm_rad

# ---------------- Drawing helpers --------------------
def boundary_angle(k):
    return k * np.pi / 10 + ROT_SHIFT

def draw_board(ax):
    for r in [R_INNER_BULL, R_OUTER_BULL,
              R_TRIPLE_IN, R_TRIPLE_OUT,
              R_DOUBLE_IN, R_DOUBLE_OUT]:
        ax.add_patch(Circle((0, 0), r,
                            transform=ax.transData._b,
                            fill=False, lw=0.6, color='k', zorder=3))
    # sector lines
    for k in range(20):
        ang = boundary_angle(k)
        ax.plot([ang, ang], [0, R_DOUBLE_OUT],
                lw=0.4, color='k', zorder=3)

    # numbers
    offset = R_DOUBLE_OUT + 12
    for k, val in enumerate(SECTOR_VALUES):
        ang_mid = boundary_angle(k) + np.pi / 20
        ax.text(ang_mid, offset, str(val),
                ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=4)

    ax.set_ylim(0, offset + 20)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

# ---------------- Streamlit UI -----------------------
st.set_page_config(layout="wide")
st.title("Dart‑board Expected‑Value Heat‑map with Spread Contours")

col1, col2, col3, col4 = st.columns(4)
with col1:
    sig_timing = st.slider("Timing spread σₜ (along‑arm, mm)", 1, 100, 8, 1)
with col2:
    sig_lateral = st.slider("Lateral spread σₗ (across‑arm, mm)", 1, 100, 6, 1)
with col3:
    theta_board = st.slider("Arm angle θ (deg, 0 = up through 20)", 0, 359, 0, 1)
with col4:
    N = st.number_input("Monte‑Carlo darts", 500, 20000, 4000, 500)

if st.button("Compute", type="primary"):
    with st.spinner("Simulating darts…"):
        ev, r_mm, th_deg, world_arm_rad = ev_grid(
            sig_timing, sig_lateral, theta_board, int(N)
        )

    # build mesh edges
    r_edges = np.concatenate(([0], r_mm + (r_mm[1] - r_mm[0]) / 2))
    th_step_deg = th_deg[1] - th_deg[0]
    th_edges = np.deg2rad(np.concatenate(
        (th_deg - th_step_deg / 2, [th_deg[-1] + th_step_deg / 2])
    ))
    th_edges_rot = (th_edges + ROT_SHIFT) % (2 * np.pi)
    Th_e, R_e = np.meshgrid(th_edges_rot, r_edges, indexing='ij')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    pcm = ax.pcolormesh(Th_e, R_e, ev.T,
                        cmap='plasma', shading='auto',
                        alpha=0.75, zorder=2)

    draw_board(ax)

    # optimal aim
    idx_best = np.unravel_index(np.argmax(ev), ev.shape)
    r_best = r_mm[idx_best[0]]
    th_best_world = np.deg2rad(th_deg[idx_best[1]])
    th_best_plot = (th_best_world + ROT_SHIFT) % (2 * np.pi)
    ax.plot(th_best_plot, r_best, marker='*', markersize=18,
            color='gold', mec='black', mew=0.9, zorder=5)

    # arm-direction dashed radial line
    arm_display_angle = (world_arm_rad + ROT_SHIFT) % (2 * np.pi)
    ax.plot([arm_display_angle, arm_display_angle],
            [0, R_DOUBLE_OUT + 20],
            ls='--', lw=1.1, color='k', zorder=4)

    # 1σ & 2σ ellipses
    x0 = r_best * np.cos(th_best_world)
    y0 = r_best * np.sin(th_best_world)
    ux, uy = np.cos(world_arm_rad), np.sin(world_arm_rad)
    vx, vy = -np.sin(world_arm_rad), np.cos(world_arm_rad)
    t = np.linspace(0, 2*np.pi, 400)
    for n, lw in [(1, 1.1), (2, 0.9)]:
        dx = n * (sig_timing * np.cos(t) * ux + sig_lateral * np.sin(t) * vx)
        dy = n * (sig_timing * np.cos(t) * uy + sig_lateral * np.sin(t) * vy)
        r = np.hypot(x0 + dx, y0 + dy)
        th = (np.arctan2(y0 + dy, x0 + dx) + ROT_SHIFT) % (2 * np.pi)
        ax.plot(th, r, color='k', lw=lw, ls='--', zorder=4)

    fig.colorbar(pcm, ax=ax, shrink=0.76, label="Expected points")
    ax.set_title("Expected score — ★ optimal aim  (dashed = 1σ, 2σ & arm axis)")
    st.pyplot(fig)
