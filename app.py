# =============================================================================
# Deadbeat Escapement Simulation — Streamlit App
# ME444 Course Project · IIT Bombay
# =============================================================================

import io
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Escapement Simulation",
    page_icon="🕰️",
    layout="wide"
)

g = 9.81

# =============================================================================
# SIMULATION FUNCTIONS (unchanged)
# =============================================================================

def simulate_pendulum(L, theta0_deg, damping, impulse_strength,
                      t_end=60, escapement=True):
    omega0       = np.sqrt(g / L)
    theta0       = np.radians(theta0_deg)
    impulse_zone = np.radians(3)

    def ode(t, y):
        theta, omega = y
        gravity  = -omega0**2 * np.sin(theta)
        friction = -damping * omega
        kick = 0.0
        if escapement and abs(theta) < impulse_zone:
            kick = impulse_strength * np.sign(omega)
        return [omega, gravity + friction + kick]

    t_eval = np.linspace(0, t_end, t_end * 400)
    sol = solve_ivp(ode, (0, t_end), [theta0, 0.0],
                    t_eval=t_eval, method='RK45',
                    rtol=1e-9, atol=1e-11, max_step=0.005)
    return sol.t, sol.y[0], sol.y[1]


def compute_period(t, theta):
    half     = len(theta) // 2
    peaks, _ = find_peaks(theta[half:])
    if len(peaks) < 2:
        return float('nan')
    return float(np.mean(np.diff(t[half:][peaks])))


# =============================================================================
# ANIMATION DRAWING
# =============================================================================

# Colour palette — dark background, brass wheel, steel fork
BG      = '#12121F'
BRASS   = '#C9972A'
BRASS_L = '#E8C060'   # highlight
STEEL   = '#7FB3CC'
STEEL_D = '#4A7A95'
ROD_C   = '#D5D8DC'
BOB_C   = '#B03030'
BOB_E   = '#7A1515'
GOLD    = '#F0B429'
FRAME_C = '#2E2E45'
SCREW_C = '#555570'


def _draw_escape_wheel(ax, cx, cy, R, n_teeth, angle):
    """Brass escape wheel with asymmetric deadbeat teeth."""
    R_root = R * 0.70
    # Filled root disc
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(cx + R_root * np.cos(th),
            cy + R_root * np.sin(th),
            color=BRASS, zorder=3, alpha=0.92)

    # Spokes (4)
    for k in range(4):
        a = angle + k * np.pi / 2
        ax.plot([cx, cx + R_root * 0.86 * np.cos(a)],
                [cy, cy + R_root * 0.86 * np.sin(a)],
                color=BG, lw=2.8, zorder=4)

    # Teeth — asymmetric triangles (deadbeat profile)
    pitch = 2 * np.pi / n_teeth
    for i in range(n_teeth):
        a0 = angle + i * pitch              # trailing root
        a1 = angle + (i + 0.30) * pitch    # tip (offset = deadbeat asymmetry)
        a2 = angle + (i + 0.88) * pitch    # leading root
        xs = [cx + R_root * np.cos(a0),
              cx + R      * np.cos(a1),
              cx + R_root * np.cos(a2)]
        ys = [cy + R_root * np.sin(a0),
              cy + R      * np.sin(a1),
              cy + R_root * np.sin(a2)]
        ax.fill(xs, ys, color=BRASS_L, zorder=5, alpha=0.95)
        ax.plot(xs + [xs[0]], ys + [ys[0]], color=BG, lw=0.6, zorder=6)

    # Hub
    ax.add_patch(plt.Circle((cx, cy), 0.075, color='#1A1A2E', zorder=7))
    ax.add_patch(plt.Circle((cx, cy), 0.035, color=BRASS,     zorder=8))


def _draw_pallet_fork(ax, fx, fy, arm_len, spread_rad, theta):
    """Steel pallet fork (anchor) that rocks with the pendulum."""
    rock   = theta * 0.58          # coupling ratio to pendulum
    aL     = -np.pi / 2 + spread_rad + rock
    aR     = -np.pi / 2 - spread_rad + rock

    LTx = fx + arm_len * np.cos(aL)
    LTy = fy + arm_len * np.sin(aL)
    RTx = fx + arm_len * np.cos(aR)
    RTy = fy + arm_len * np.sin(aR)

    # Arms
    for tx, ty in [(LTx, LTy), (RTx, RTy)]:
        ax.plot([fx, tx], [fy, ty], color=STEEL_D, lw=5.5,
                solid_capstyle='round', zorder=8)
        ax.plot([fx, tx], [fy, ty], color=STEEL, lw=3.0,
                solid_capstyle='round', zorder=9)

    # Pallet faces (locking/impulse surfaces — gold highlight)
    pf = 0.14
    for tx, ty, a_perp in [
        (LTx, LTy, aL + np.pi / 2),
        (RTx, RTy, aR - np.pi / 2)
    ]:
        px, py = pf * np.cos(a_perp), pf * np.sin(a_perp)
        ax.plot([tx - px, tx + px], [ty - py, ty + py],
                color=GOLD, lw=5, solid_capstyle='round', zorder=11)

    # Crutch (connects fork body to pendulum rod)
    crutch_a   = -np.pi / 2 + rock
    crutch_len = 0.42
    CX = fx + crutch_len * np.cos(crutch_a)
    CY = fy + crutch_len * np.sin(crutch_a)
    ax.plot([fx, CX], [fy, CY], color=STEEL, lw=2,
            linestyle=(0, (4, 3)), zorder=7, alpha=0.65)

    # Fork pivot
    ax.add_patch(plt.Circle((fx, fy), 0.085, color=STEEL_D, zorder=12))
    ax.add_patch(plt.Circle((fx, fy), 0.042, color='#1A1A2E', zorder=13))


def _draw_pendulum(ax, px, py, length, theta):
    """Pendulum rod and bob."""
    bx = px + length * np.sin(theta)
    by = py - length * np.cos(theta)

    # Shadow / depth line
    ax.plot([px, bx], [py, by], color='#000', lw=4, zorder=6, alpha=0.3)
    # Rod
    ax.plot([px, bx], [py, by], color=ROD_C, lw=2.2, zorder=7, alpha=0.9)

    # Bob
    ax.add_patch(plt.Circle((bx, by), 0.27, color=BOB_E,   zorder=9))
    ax.add_patch(plt.Circle((bx, by), 0.25, color=BOB_C,   zorder=10))
    # Highlight
    ax.add_patch(plt.Circle((bx - 0.07, by + 0.07), 0.07,
                             color='#E05050', alpha=0.45, zorder=11))

    # Pivot
    ax.add_patch(plt.Circle((px, py), 0.055, color=ROD_C, zorder=12))

    return bx, by


def draw_frame(theta, esc_angle, frame_info=""):
    """
    Render one animation frame.
    Returns a PNG bytes object (fast for st.image).
    """
    # Layout constants
    EX, EY = 0.0,  0.10    # escape wheel centre
    ER     = 0.96           # escape wheel tip radius
    FX, FY = 0.0, EY + ER + 0.62   # fork pivot
    PL     = 2.75           # pendulum visual length

    fig, ax = plt.subplots(figsize=(4.2, 6.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-3.65, 1.95)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Mounting backplate ────────────────────────────────────────
    ax.fill_between([-1.55, 1.55], [FY + 0.28, FY + 0.28],
                    [FY + 0.60, FY + 0.60],
                    color=FRAME_C, alpha=0.9, zorder=1)
    ax.plot([-1.55, 1.55], [FY + 0.28, FY + 0.28],
            color='#3A3A55', lw=2.2, zorder=2)
    # Mounting screws
    for sx in [-1.30, 1.30]:
        ax.add_patch(plt.Circle((sx, FY + 0.44), 0.055,
                                color=SCREW_C, zorder=3))
        ax.add_patch(plt.Circle((sx, FY + 0.44), 0.022,
                                color='#888', zorder=4))

    # ── Escape wheel ─────────────────────────────────────────────
    _draw_escape_wheel(ax, EX, EY, ER, 30, esc_angle)

    # ── Pallet fork ──────────────────────────────────────────────
    _draw_pallet_fork(ax, FX, FY, arm_len=0.72,
                      spread_rad=np.radians(26), theta=theta)

    # ── Pendulum ─────────────────────────────────────────────────
    _draw_pendulum(ax, FX, FY, PL, theta)

    # ── Info text ────────────────────────────────────────────────
    ax.text(0, -3.50, frame_info,
            color='#888', ha='center', va='bottom',
            fontsize=8.5, fontfamily='monospace', zorder=15)
    ax.text(0, 1.85,
            "Deadbeat Escapement — Clock One",
            color='#9090B0', ha='center', va='top',
            fontsize=7.5, zorder=15)

    plt.tight_layout(pad=0.15)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90,
                bbox_inches='tight', facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# =============================================================================
# PLOT FUNCTIONS (unchanged)
# =============================================================================

def plot_time_series(t_free, theta_free, t_esc, theta_esc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    fig.suptitle("Pendulum Motion: Free Decay vs Escapement-Driven",
                 fontsize=12, fontweight='bold', y=1.01)

    ax1.plot(t_free, np.degrees(theta_free), color='tomato', lw=1.2)
    ax1.axhline(0, color='grey', lw=0.6, ls='--')
    ax1.set_title("Without Escapement → Motion Dies Out")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("θ (degrees)")
    ax1.grid(alpha=0.3)
    ax1.text(0.97, 0.95, "Damping wins —\nno energy input",
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=9, color='tomato',
             bbox=dict(boxstyle='round', fc='white', ec='tomato', alpha=0.7))

    ax2.plot(t_esc, np.degrees(theta_esc), color='steelblue', lw=1.2)
    ax2.axhline(0, color='grey', lw=0.6, ls='--')
    ax2.set_title("With Escapement → Sustained Oscillation")
    ax2.set_xlabel("Time (s)"); ax2.grid(alpha=0.3)
    ax2.text(0.97, 0.95, "Impulse replenishes\nenergy each tick",
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=9, color='steelblue',
             bbox=dict(boxstyle='round', fc='white', ec='steelblue', alpha=0.7))

    plt.tight_layout()
    return fig


def plot_phase_portrait(t_free, theta_free, omega_free,
                        t_esc,  theta_esc,  omega_esc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Phase Portrait  (θ vs θ̇)",
                 fontsize=12, fontweight='bold', y=1.01)

    ax1.plot(np.degrees(theta_free), np.degrees(omega_free),
             color='tomato', lw=1.0)
    ax1.plot(np.degrees(theta_free[0]),  np.degrees(omega_free[0]),
             'o', color='green', ms=7, label='Start', zorder=5)
    ax1.plot(np.degrees(theta_free[-1]), np.degrees(omega_free[-1]),
             's', color='black', ms=7, label='End',   zorder=5)
    ax1.set_title("Free Pendulum → Spiral to Rest")
    ax1.set_xlabel("θ (degrees)"); ax1.set_ylabel("θ̇ (degrees/s)")
    ax1.axhline(0, color='grey', lw=0.5); ax1.axvline(0, color='grey', lw=0.5)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    cut = len(theta_esc) // 4
    ax2.plot(np.degrees(theta_esc[:cut]), np.degrees(omega_esc[:cut]),
             color='lightsteelblue', lw=0.8, label='Transient')
    ax2.plot(np.degrees(theta_esc[cut:]), np.degrees(omega_esc[cut:]),
             color='steelblue', lw=1.4, label='Limit cycle')
    ax2.set_title("Escapement-Driven → Limit Cycle")
    ax2.set_xlabel("θ (degrees)"); ax2.set_ylabel("θ̇ (degrees/s)")
    ax2.axhline(0, color='grey', lw=0.5); ax2.axvline(0, color='grey', lw=0.5)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("🕰️ Deadbeat Escapement — Pendulum Simulation")
st.caption("ME444 · Analysis and Design of Mechanical Systems · IIT Bombay")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")
    st.caption("Adjust and click **Run Simulation**")

    L       = st.slider("Pendulum length  L (m)",  0.3,  1.5,  0.876, 0.001,
                        help="Clock One prototype: 876 mm")
    theta0  = st.slider("Initial angle  θ₀ (°)",   2.0, 30.0,  8.0,  0.5)
    impulse = st.slider("Impulse strength  K",      0.0,  0.5,  0.25, 0.01,
                        help="Set to 0 to see free decay only")
    damping = 0.06

    st.markdown("---")
    run = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Fixed parameters**")
    st.markdown(f"- Damping b = {damping}")
    st.markdown(f"- g = {g} m/s²")
    st.markdown(f"- Escape wheel = 30 teeth")

# ── Landing page ──────────────────────────────────────────────────────────────
if not run:
    st.info("👈  Adjust the sliders in the sidebar, then click **Run Simulation**.")
    st.markdown("""
    **What this simulation shows**

    | Plot | What to look for |
    |------|-----------------|
    | θ vs time (free)   | Oscillation decays — damping removes energy |
    | θ vs time (driven) | Oscillation sustained — escapement replenishes energy |
    | Phase portrait     | Free spiral inward vs driven limit cycle |

    **Prototype parameters** (Clock One, 3D printed PLA)
    - Pendulum length: 876 mm  ·  Escape wheel: 30 teeth  ·  Gear ratio: 16 : 1
    """)
    st.stop()

# ── Run simulation ─────────────────────────────────────────────────────────────
with st.spinner("Simulating..."):
    t_free, th_free, om_free = simulate_pendulum(
        L, theta0, damping, impulse, t_end=60, escapement=False)
    t_esc, th_esc, om_esc = simulate_pendulum(
        L, theta0, damping, impulse, t_end=60, escapement=True)

T_theory = 2 * np.pi * np.sqrt(L / g)
T_sim    = compute_period(t_esc, th_esc)
error    = abs(T_sim - T_theory) / T_theory * 100

# Pre-compute escape wheel angles (steps on each zero-crossing = 1 tick)
dtheta_tooth = 2 * np.pi / 30
esc_angles   = np.zeros(len(t_esc))
esc_cur      = 0.0
for k in range(1, len(t_esc)):
    if th_esc[k - 1] * th_esc[k] < 0:
        esc_cur += dtheta_tooth
    esc_angles[k] = esc_cur

# ── Validation metrics ─────────────────────────────────────────────────────────
st.subheader("📊 Period Validation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pendulum length",  f"{L * 1000:.0f} mm")
c2.metric("Theoretical period", f"{T_theory:.4f} s")
c3.metric("Simulated period",   f"{T_sim:.4f} s")
c4.metric("Error", f"{error:.3f} %", delta=f"{error:.3f}%", delta_color="inverse")

st.markdown("---")

# ── Layout: animation left, plots right ───────────────────────────────────────
col_anim, col_plots = st.columns([1, 2], gap="large")

with col_anim:
    st.subheader("🎞️ Live Animation")
    anim_slot = st.empty()
    tick_info = st.empty()

with col_plots:
    st.subheader("Plot 1 — Angular Displacement vs Time")
    fig1 = plot_time_series(t_free, th_free, t_esc, th_esc)
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Plot 2 — Phase Portrait  (θ vs θ̇)")
    fig2 = plot_phase_portrait(t_free, th_free, om_free,
                               t_esc,  th_esc,  om_esc)
    st.pyplot(fig2)
    plt.close(fig2)

# ── Animation loop ─────────────────────────────────────────────────────────────
# Show ~8 seconds of simulation at ~14 fps
FPS        = 14
ANIM_SECS  = 8
N_FRAMES   = FPS * ANIM_SECS                          # 112 frames

# Sample indices from first ANIM_SECS seconds of simulation
end_idx  = np.searchsorted(t_esc, ANIM_SECS)
end_idx  = min(end_idx, len(t_esc) - 1)
indices  = np.linspace(0, end_idx, N_FRAMES, dtype=int)

tick_count = 0
for idx in indices:
    theta_f = float(th_esc[idx])
    esc_f   = float(esc_angles[idx])
    t_f     = float(t_esc[idx])

    # Count ticks
    tick_count = int(esc_f / dtheta_tooth)

    info = f"t = {t_f:.2f}s   θ = {np.degrees(theta_f):+5.1f}°   ticks = {tick_count}"
    frame_bytes = draw_frame(theta_f, esc_f, info)

    anim_slot.image(frame_bytes, use_container_width=True)
    time.sleep(1.0 / FPS)

tick_info.caption(f"✅ Animation complete — {tick_count} ticks shown over {ANIM_SECS}s")
