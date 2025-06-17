
"""
Streamlit app: dart_heatmap_polar_numbers.py

Interactive dart-board EV visualiser.

Toggles (in sidebar):
---------------------
▸ **Enable custom arm angle** – default OFF (arm axis fixed vertical).
▸ **Separate timing vs lateral spreads** – default OFF (single σ controls both).

Visual elements:
----------------
▸ Heat‑map of expected score for every aim point
▸ Gold ★ at optimal aim
▸ Dashed ellipses (1 σ, 2 σ) showing landing spread
▸ Dashed line along arm axis through the ★
▸ Sector numbers and ring outlines
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle

# ---------------- Board constants --------------------
R_INNER_BULL, R_OUTER_BULL = 6.35, 15.9
R_TRIPLE_IN, R_TRIPLE_OUT = 99.0, 107.0
R_DOUBLE_IN, R_DOUBLE_OUT = 162.0, 170.0

SECTOR_VALUES = np.array([
    20, 1,18, 4,13, 6,10,15, 2,17,
     3,19, 7,16, 8,11,14, 9,12, 5
])

ROT_SHIFT = np.pi/2 - np.pi/20   # rotate plot so 20 is at top

# ---------------- Scoring ----------------------------
def score_polar(r, th):
    th_mod = np.mod(th, 2*np.pi)
    idx = (th_mod // (np.pi/10)).astype(int)
    base = SECTOR_VALUES[idx]

    s = np.zeros_like(r, dtype=np.int16)
    s[r <= R_INNER_BULL] = 50
    ring = (r > R_INNER_BULL) & (r <= R_OUTER_BULL); s[ring] = 25
    ring = (r > R_TRIPLE_IN) & (r <= R_TRIPLE_OUT);  s[ring] = 3*base[ring]
    ring = (r > R_DOUBLE_IN) & (r <= R_DOUBLE_OUT);  s[ring] = 2*base[ring]
    ring = (s == 0) & (r <= R_DOUBLE_OUT);           s[ring] = base[ring]
    return s

# ---------------- EV grid ----------------------------
def ev_grid(sig_t, sig_l, theta_board_deg, N, seed=1234,
            r_grid=None, th_grid=None):
    if r_grid is None:
        r_grid = np.arange(0.0, R_DOUBLE_IN, 4.0)
    if th_grid is None:
        th_grid = np.arange(0.0, 360.0, 4.0)

    r_mesh, th_mesh = np.meshgrid(r_grid, np.deg2rad(th_grid), indexing='ij')
    x0, y0 = r_mesh*np.cos(th_mesh), r_mesh*np.sin(th_mesh)

    world_arm = np.deg2rad(theta_board_deg) + (np.pi/2 - ROT_SHIFT)
    ux, uy = np.cos(world_arm), np.sin(world_arm)
    vx, vy = -np.sin(world_arm), np.cos(world_arm)

    rng = np.random.default_rng(seed)
    eps_t = rng.normal(0, sig_t, (N, 1))
    eps_l = rng.normal(0, sig_l, (N, 1))

    xs = x0.ravel()[None, :] + eps_t*ux + eps_l*vx
    ys = y0.ravel()[None, :] + eps_t*uy + eps_l*vy

    ev = score_polar(np.hypot(xs, ys), np.arctan2(ys, xs)).mean(0)
    return ev.reshape(r_grid.size, th_grid.size), r_grid, th_grid, world_arm

# ---------------- Drawing helpers --------------------
def boundary_angle(k):
    return k*np.pi/10 + ROT_SHIFT

def draw_board(ax):
    # rings
    for r in [R_INNER_BULL, R_OUTER_BULL,
              R_TRIPLE_IN, R_TRIPLE_OUT,
              R_DOUBLE_IN, R_DOUBLE_OUT]:
        ax.add_patch(Circle((0,0), r, transform=ax.transData._b,
                            fill=False, lw=0.6, color='k', zorder=3))
    # sector lines
    for k in range(20):
        ang = boundary_angle(k)
        ax.plot([ang, ang], [0, R_DOUBLE_OUT], lw=0.4, color='k', zorder=3)
    # numbers
    offset = R_DOUBLE_OUT + 12
    for k, val in enumerate(SECTOR_VALUES):
        ang_mid = boundary_angle(k) + np.pi/20
        ax.text(ang_mid, offset, str(val),
                ha='center', va='center', fontsize=9,
                fontweight='bold', zorder=4)
    ax.set_ylim(0, offset+20)
    ax.set_yticklabels([]); ax.set_xticklabels([])

# ---------------- Streamlit UI -----------------------
st.set_page_config(layout="wide")
st.title("Dart‑board Expected‑Value Heat‑map")

# sidebar toggles
st.sidebar.header("Options")
custom_angle = st.sidebar.checkbox("Enable custom arm angle", value=False)
separate_spreads = st.sidebar.checkbox("Separate timing vs lateral spreads", value=False)

# main sliders
cols = st.columns(4)
with cols[0]:
    if separate_spreads:
        sig_t = st.slider("Timing σₜ (along‑arm)", 1, 100, 8, 1)
    else:
        sig_all = st.slider("Spread σ", 1, 100, 8, 1)
        sig_t = sig_all
with cols[1]:
    if separate_spreads:
        sig_l = st.slider("Lateral σₗ (across‑arm)", 1, 100, 8, 1)
    else:
        sig_l = sig_t
with cols[2]:
    if custom_angle:
        theta_board = st.slider("Arm axis θ° (0–180)", 0, 180, 0, 1)
    else:
        theta_board = 0
        st.markdown("Arm axis θ°: **0 (fixed)**")
with cols[3]:
    N = st.number_input("Monte‑Carlo darts", 500, 20000, 4000, 500)

if st.button("Compute", type="primary"):
    with st.spinner("Simulating darts…"):
        ev, r_mm, th_deg, world_arm = ev_grid(
            sig_t, sig_l, theta_board, int(N)
        )

    # mesh edges
    r_edges = np.concatenate(([0], r_mm + (r_mm[1]-r_mm[0])/2))
    th_step = th_deg[1] - th_deg[0]
    th_edges = np.deg2rad(np.concatenate(
        (th_deg - th_step/2, [th_deg[-1] + th_step/2])
    ))
    th_edges_rot = (th_edges + ROT_SHIFT) % (2*np.pi)
    Th_e, R_e = np.meshgrid(th_edges_rot, r_edges, indexing='ij')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')
    pcm = ax.pcolormesh(Th_e, R_e, ev.T, cmap='plasma',
                        shading='auto', alpha=0.75, zorder=2)

    draw_board(ax)

    # optimal aim
    idx_best = np.unravel_index(np.argmax(ev), ev.shape)
    r_best = r_mm[idx_best[0]]
    th_best_world = np.deg2rad(th_deg[idx_best[1]])
    th_best_plot = (th_best_world + ROT_SHIFT) % (2*np.pi)
    ax.plot(th_best_plot, r_best, marker='*', markersize=18,
            color='gold', mec='black', mew=1, zorder=5)

    # arm axis through star
    ux, uy = np.cos(world_arm), np.sin(world_arm)
    x0, y0 = r_best*np.cos(th_best_world), r_best*np.sin(th_best_world)
    L = R_DOUBLE_OUT + 40
    x_line = x0 + np.array([-L, L])*ux
    y_line = y0 + np.array([-L, L])*uy
    r_line = np.hypot(x_line, y_line)
    th_line = (np.arctan2(y_line, x_line) + ROT_SHIFT) % (2*np.pi)
    ax.plot(th_line, r_line, ls='--', lw=1.1, color='k')

    # ellipses
    vx, vy = -np.sin(world_arm), np.cos(world_arm)
    t = np.linspace(0, 2*np.pi, 400)
    for n, lw in [(1,1.1), (2,0.9)]:
        dx = n*(sig_t*np.cos(t)*ux + sig_l*np.sin(t)*vx)
        dy = n*(sig_t*np.cos(t)*uy + sig_l*np.sin(t)*vy)
        r = np.hypot(x0+dx, y0+dy)
        th = (np.arctan2(y0+dy, x0+dx) + ROT_SHIFT) % (2*np.pi)
        ax.plot(th, r, color='k', lw=lw, ls='--')

    fig.colorbar(pcm, ax=ax, shrink=0.75, label="Expected points")
    ax.set_title("Expected score — ★ optimal aim  (dashed: 1σ, 2σ, axis)")
    st.pyplot(fig)
