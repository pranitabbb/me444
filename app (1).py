# =============================================================================
# Deadbeat Escapement Simulation — Streamlit App
# ME444 · Analysis and Design of Mechanical Systems · IIT Bombay
# Clock One — 3D Printed Prototype
# =============================================================================

import io
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
from matplotlib.path import Path
import matplotlib.patheffects as pe
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deadbeat Escapement · ME444 · IIT Bombay",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a45;
    }
    .main-header h1 { color: #e8c060; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { color: #8888aa; margin: 0.3rem 0 0; font-size: 0.85rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a45;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #8888aa; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: #e8c060; font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; }
    .metric-sub   { color: #5588aa; font-size: 0.75rem; margin-top: 0.15rem; }
    .section-header {
        color: #c0c0d8;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.2rem 0 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #2a2a45;
    }
    div[data-testid="stSidebar"] { background: #0e0e1c; }
    div[data-testid="stSidebar"] .stSlider > div > div > div { background: #c9972a; }
    .stButton > button {
        background: linear-gradient(135deg, #c9972a, #e8c060) !important;
        color: #0e0e1c !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.05em !important;
    }
    .info-box {
        background: #12121f;
        border: 1px solid #2a2a45;
        border-left: 3px solid #c9972a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
g        = 9.81
N_TEETH  = 30
DAMPING  = 0.06

# Colour palette
BG       = '#0f0f1a'
BG2      = '#1a1a2e'
BRASS    = '#C9972A'
BRASS_L  = '#F0C84A'
BRASS_D  = '#8B6914'
STEEL    = '#90BFD8'
STEEL_D  = '#3A6A85'
STEEL_HL = '#C8E8F8'
ROD_C    = '#D8DDE2'
BOB_C    = '#CC3333'
BOB_D    = '#881111'
BOB_HL   = '#FF6655'
GOLD     = '#FFD060'
FRAME_C  = '#1E1E38'
FRAME_HL = '#2E2E50'
SCREW_C  = '#484868'
WOOD_C   = '#5C3D1E'
WOOD_L   = '#7A5230'
TEXT_DIM = '#666688'


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def simulate_pendulum(L, theta0_deg, damping, impulse_strength,
                      t_end=60, escapement=True):
    omega0       = np.sqrt(g / L)
    theta0       = np.radians(theta0_deg)
    impulse_zone = np.radians(3.5)

    def ode(t, y):
        theta, omega = y
        gravity  = -omega0**2 * np.sin(theta)
        friction = -damping * omega
        kick = 0.0
        if escapement and abs(theta) < impulse_zone and abs(omega) > 0.01:
            kick = impulse_strength * np.sign(omega)
        return [omega, gravity + friction + kick]

    t_eval = np.linspace(0, t_end, t_end * 500)
    sol = solve_ivp(ode, (0, t_end), [theta0, 0.0],
                    t_eval=t_eval, method='RK45',
                    rtol=1e-9, atol=1e-12, max_step=0.004)
    return sol.t, sol.y[0], sol.y[1]


def compute_period(t, theta):
    half     = len(theta) // 2
    peaks, _ = find_peaks(theta[half:], height=0.01)
    if len(peaks) < 2:
        return float('nan')
    return float(np.mean(np.diff(t[half:][peaks])))


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def _circle_fill(ax, cx, cy, r, color, zorder=3, alpha=1.0, ec=None, lw=1):
    c = plt.Circle((cx, cy), r, color=color, zorder=zorder,
                   alpha=alpha, ec=ec if ec else color, lw=lw)
    ax.add_patch(c)


def _draw_wooden_frame(ax, fy):
    """Vertical wooden mounting board behind the mechanism."""
    board_w = 0.38
    board_h = 7.2
    board_top = fy + 0.7
    board_bot = board_top - board_h

    # Wood grain fill
    ax.fill_between([-board_w, board_w],
                    [board_bot, board_bot], [board_top, board_top],
                    color=WOOD_C, alpha=0.92, zorder=0)
    # Grain lines
    np.random.seed(42)
    for _ in range(18):
        x = np.random.uniform(-board_w + 0.04, board_w - 0.04)
        ax.plot([x, x + np.random.uniform(-0.05, 0.05)],
                [board_bot, board_top],
                color=WOOD_L, lw=0.5, alpha=0.35, zorder=1)
    # Edges
    ax.plot([-board_w, -board_w], [board_bot, board_top], color=BRASS_D, lw=1.5, zorder=2, alpha=0.7)
    ax.plot([ board_w,  board_w], [board_bot, board_top], color=BRASS_D, lw=1.5, zorder=2, alpha=0.7)


def _draw_mounting_bracket(ax, fy):
    """Top mounting plate with wood-screw details."""
    bx1, bx2 = -1.6, 1.6
    by1, by2 = fy + 0.22, fy + 0.58

    # Main plate
    bracket = mpatches.FancyBboxPatch((bx1, by1), bx2 - bx1, by2 - by1,
                                       boxstyle="round,pad=0.04",
                                       fc=FRAME_C, ec=FRAME_HL, lw=1.5, zorder=10)
    ax.add_patch(bracket)

    # Rivets / screws along plate
    for sx in [-1.35, -0.68, 0.0, 0.68, 1.35]:
        sy = (by1 + by2) / 2
        _circle_fill(ax, sx, sy, 0.055, SCREW_C, zorder=12)
        _circle_fill(ax, sx, sy, 0.024, '#1A1A2E', zorder=13)
        # Cross slot
        ax.plot([sx - 0.02, sx + 0.02], [sy, sy], color='#888', lw=0.8, zorder=14)
        ax.plot([sx, sx], [sy - 0.02, sy + 0.02], color='#888', lw=0.8, zorder=14)


def _draw_escape_wheel(ax, cx, cy, R, n_teeth, angle):
    """High-quality brass escape wheel with deadbeat tooth profile."""
    R_root = R * 0.68
    R_hub  = R * 0.18
    R_hub_i= R * 0.08
    n_spokes = 5

    # --- Outer ring ---
    th = np.linspace(0, 2 * np.pi, 360)
    ax.fill(cx + R_root * np.cos(th),
            cy + R_root * np.sin(th),
            color=BRASS, zorder=20, alpha=0.95)

    # Inner ring cutout for aesthetics
    ax.fill(cx + R_hub * 2.1 * np.cos(th),
            cy + R_hub * 2.1 * np.sin(th),
            color=BG2, zorder=21)

    # Spokes with taper
    for k in range(n_spokes):
        a = angle + k * (2 * np.pi / n_spokes)
        # Spoke as a thin trapezoid
        spoke_w_in  = 0.055
        spoke_w_out = 0.025
        perp = a + np.pi / 2
        r_in  = R_hub * 2.1
        r_out = R_root * 0.97
        pts_x = [
            cx + r_in  * np.cos(a) + spoke_w_in  * np.cos(perp),
            cx + r_out * np.cos(a) + spoke_w_out * np.cos(perp),
            cx + r_out * np.cos(a) - spoke_w_out * np.cos(perp),
            cx + r_in  * np.cos(a) - spoke_w_in  * np.cos(perp),
        ]
        pts_y = [
            cy + r_in  * np.sin(a) + spoke_w_in  * np.sin(perp),
            cy + r_out * np.sin(a) + spoke_w_out * np.sin(perp),
            cy + r_out * np.sin(a) - spoke_w_out * np.sin(perp),
            cy + r_in  * np.sin(a) - spoke_w_in  * np.sin(perp),
        ]
        ax.fill(pts_x, pts_y, color=BRASS, zorder=22)

    # Teeth — deadbeat profile (sharp leading, curved locking face)
    pitch = 2 * np.pi / n_teeth
    for i in range(n_teeth):
        a_root1 = angle + i * pitch
        a_tip   = angle + (i + 0.28) * pitch
        a_lock  = angle + (i + 0.55) * pitch
        a_root2 = angle + (i + 0.92) * pitch

        # Locking face is slightly curved (3 points)
        a_lock2 = angle + (i + 0.72) * pitch
        R_lock  = R * 0.92  # locking face radius (shorter than tip)

        xs = [
            cx + R_root * np.cos(a_root1),
            cx + R       * np.cos(a_tip),
            cx + R_lock  * np.cos(a_lock),
            cx + R_lock  * np.cos(a_lock2),
            cx + R_root  * np.cos(a_root2),
        ]
        ys = [
            cy + R_root * np.sin(a_root1),
            cy + R       * np.sin(a_tip),
            cy + R_lock  * np.sin(a_lock),
            cy + R_lock  * np.sin(a_lock2),
            cy + R_root  * np.sin(a_root2),
        ]
        ax.fill(xs, ys, color=BRASS_L, zorder=23, alpha=0.97)
        ax.plot(xs + [xs[0]], ys + [ys[0]], color=BRASS_D, lw=0.7, zorder=24)

    # Hub layers
    _circle_fill(ax, cx, cy, R_hub * 1.05, BRASS_D, zorder=25)
    _circle_fill(ax, cx, cy, R_hub,        BRASS,   zorder=26)
    _circle_fill(ax, cx, cy, R_hub_i,      '#0f0f1a', zorder=27)

    # Shaft hole
    _circle_fill(ax, cx, cy, R_hub_i * 0.55, STEEL_D, zorder=28)


def _draw_pallet_fork(ax, fx, fy, arm_len, spread_rad, theta):
    """Steel pallet fork (anchor) that rocks with the pendulum."""
    rock = theta * 0.55
    aL   = -np.pi / 2 + spread_rad + rock
    aR   = -np.pi / 2 - spread_rad + rock

    LTx = fx + arm_len * np.cos(aL)
    LTy = fy + arm_len * np.sin(aL)
    RTx = fx + arm_len * np.cos(aR)
    RTy = fy + arm_len * np.sin(aR)

    def draw_arm(x0, y0, x1, y1):
        # Shadow
        ax.plot([x0, x1], [y0, y1], color='#000', lw=10, alpha=0.18,
                solid_capstyle='round', zorder=28)
        # Dark core
        ax.plot([x0, x1], [y0, y1], color=STEEL_D, lw=7.5,
                solid_capstyle='round', zorder=29)
        # Mid highlight
        ax.plot([x0, x1], [y0, y1], color=STEEL, lw=4.5,
                solid_capstyle='round', zorder=30)
        # Top highlight (shimmer)
        ax.plot([x0, x1], [y0, y1], color=STEEL_HL, lw=1.5,
                solid_capstyle='round', zorder=31, alpha=0.5)

    draw_arm(fx, fy, LTx, LTy)
    draw_arm(fx, fy, RTx, RTy)

    # Pallet faces (impulse/locking surfaces)
    pf = 0.18
    for (tx, ty, aa, side) in [
        (LTx, LTy, aL + np.pi / 2, 'L'),
        (RTx, RTy, aR - np.pi / 2, 'R'),
    ]:
        px, py = pf * np.cos(aa), pf * np.sin(aa)
        # Gold pallet pad
        ax.plot([tx - px, tx + px], [ty - py, ty + py],
                color=BRASS_D, lw=9, solid_capstyle='round', zorder=32)
        ax.plot([tx - px, tx + px], [ty - py, ty + py],
                color=GOLD,   lw=6, solid_capstyle='round', zorder=33)
        ax.plot([tx - px, tx + px], [ty - py, ty + py],
                color=BRASS_L, lw=2, solid_capstyle='round', zorder=34, alpha=0.6)

    # Crutch rod going down to pendulum
    crutch_a   = -np.pi / 2 + rock
    crutch_len = 0.52
    CX = fx + crutch_len * np.cos(crutch_a)
    CY = fy + crutch_len * np.sin(crutch_a)
    ax.plot([fx, CX], [fy, CY], color=STEEL_D, lw=4,
            solid_capstyle='round', zorder=27, alpha=0.8)
    ax.plot([fx, CX], [fy, CY], color=STEEL, lw=2,
            solid_capstyle='round', zorder=28, alpha=0.7)

    # Fork pivot bearing
    _circle_fill(ax, fx, fy, 0.115, STEEL_D, zorder=35)
    _circle_fill(ax, fx, fy, 0.080, STEEL,   zorder=36)
    _circle_fill(ax, fx, fy, 0.038, '#0f0f1a', zorder=37)
    _circle_fill(ax, fx, fy, 0.018, STEEL_D,  zorder=38)


def _draw_pendulum(ax, px, py, length, theta):
    """High-quality pendulum rod with suspension spring and decorative bob."""
    bx = px + length * np.sin(theta)
    by = py - length * np.cos(theta)

    # Suspension spring (small zig-zag at top)
    spring_h = 0.22
    n_coils  = 5
    ys = np.linspace(py, py - spring_h, n_coils * 8)
    xs = 0.025 * np.sin(np.linspace(0, n_coils * 2 * np.pi, len(ys)))
    # Rotate by theta
    # (approximate — spring stays near vertical)
    ax.plot(px + xs, ys, color=STEEL, lw=1.5, zorder=40, alpha=0.9)

    rod_start_y = py - spring_h
    rod_start_x = px

    # Shadow rod
    ax.plot([rod_start_x, bx], [rod_start_y, by],
            color='#000', lw=6, zorder=37, alpha=0.22,
            solid_capstyle='round')
    # Outer rod (dark steel)
    ax.plot([rod_start_x, bx], [rod_start_y, by],
            color=STEEL_D, lw=4, zorder=38, solid_capstyle='round')
    # Inner rod highlight
    ax.plot([rod_start_x, bx], [rod_start_y, by],
            color=ROD_C, lw=2.0, zorder=39, solid_capstyle='round', alpha=0.85)
    # Shimmer
    ax.plot([rod_start_x, bx], [rod_start_y, by],
            color='#ffffff', lw=0.6, zorder=40, solid_capstyle='round', alpha=0.25)

    # Adjusting nut (small disc on rod, at 2/3 length)
    nut_x = rod_start_x + (bx - rod_start_x) * 0.72
    nut_y = rod_start_y + (by - rod_start_y) * 0.72
    _circle_fill(ax, nut_x, nut_y, 0.07, BRASS_D, zorder=42)
    _circle_fill(ax, nut_x, nut_y, 0.055, BRASS,  zorder=43)
    # Hex-ish nut indication
    for aa in range(6):
        ang = aa * np.pi / 3
        ax.plot([nut_x + 0.04 * np.cos(ang), nut_x + 0.055 * np.cos(ang)],
                [nut_y + 0.04 * np.sin(ang), nut_y + 0.055 * np.sin(ang)],
                color=BRASS_D, lw=0.8, zorder=44)

    # Bob — outer shadow
    _circle_fill(ax, bx, by, 0.33, '#000', zorder=44, alpha=0.35)
    # Bob body layers
    _circle_fill(ax, bx, by, 0.30, BOB_D, zorder=45)
    _circle_fill(ax, bx, by, 0.27, BOB_C, zorder=46)
    # Bob pattern — concentric rings (like the 3D printed bob)
    for r_ring, alpha_ring in [(0.20, 0.25), (0.14, 0.22), (0.08, 0.2)]:
        ring = plt.Circle((bx, by), r_ring, fill=False,
                           ec='#FF8866', lw=0.8, alpha=alpha_ring, zorder=47)
        ax.add_patch(ring)
    # Specular highlight
    _circle_fill(ax, bx - 0.09, by + 0.09, 0.085, BOB_HL, zorder=48, alpha=0.30)
    _circle_fill(ax, bx - 0.06, by + 0.06, 0.038, '#FFCCBB', zorder=49, alpha=0.25)

    # Bob rim
    rim = plt.Circle((bx, by), 0.28, fill=False,
                     ec=BOB_D, lw=1.8, zorder=50)
    ax.add_patch(rim)

    # Pivot cap at top
    _circle_fill(ax, px, py, 0.072, STEEL_D, zorder=41)
    _circle_fill(ax, px, py, 0.048, STEEL,   zorder=42)
    _circle_fill(ax, px, py, 0.020, '#0f0f1a', zorder=43)

    return bx, by


def draw_frame(theta, esc_angle, tick_count, t_now, theta0_max):
    """
    Render one high-quality animation frame. Returns PNG bytes.
    """
    # Layout constants
    EX, EY  = 0.0,  -0.30
    ER      = 1.00
    FX, FY  = 0.0,  EY + ER + 0.72
    PL      = 3.10    # pendulum visual length

    fig, ax = plt.subplots(figsize=(4.6, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-4.10, 2.10)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Wooden back board ─────────────────────────────────────────
    _draw_wooden_frame(ax, FY)

    # ── Mounting bracket ─────────────────────────────────────────
    _draw_mounting_bracket(ax, FY)

    # ── Escape wheel ─────────────────────────────────────────────
    _draw_escape_wheel(ax, EX, EY, ER, N_TEETH, esc_angle)

    # ── Pallet fork ──────────────────────────────────────────────
    _draw_pallet_fork(ax, FX, FY, arm_len=0.80,
                      spread_rad=np.radians(27), theta=theta)

    # ── Pendulum ─────────────────────────────────────────────────
    bx, by = _draw_pendulum(ax, FX, FY, PL, theta)

    # ── Pendulum arc trace (ghost trail) ─────────────────────────
    arc_r   = PL * 0.98
    arc_max = abs(theta0_max) * 1.1
    arc_th  = np.linspace(-arc_max, arc_max, 60)
    arc_x   = FX + arc_r * np.sin(arc_th)
    arc_y   = FY - arc_r * np.cos(arc_th)
    ax.plot(arc_x, arc_y, color='#ffffff', lw=0.6, alpha=0.06,
            zorder=35, linestyle='--')

    # ── Current position dot on arc ──────────────────────────────
    ax.plot(bx, by, 'o', color=GOLD, ms=3.5, alpha=0.5, zorder=55)

    # ── Tick counter display ──────────────────────────────────────
    # Mini digital readout area
    rd_x, rd_y, rd_w, rd_h = -1.78, -3.85, 3.56, 0.56
    readout = mpatches.FancyBboxPatch(
        (rd_x, rd_y), rd_w, rd_h,
        boxstyle="round,pad=0.06",
        fc='#0a0a14', ec='#2a2a45', lw=1.2, zorder=58
    )
    ax.add_patch(readout)
    ax.text(-1.60, -3.57,
            f"t",
            color=TEXT_DIM, ha='left', va='center',
            fontsize=7.5, fontfamily='monospace', zorder=60)
    ax.text(-1.35, -3.57,
            f"= {t_now:5.2f} s",
            color='#aaaacc', ha='left', va='center',
            fontsize=7.5, fontfamily='monospace', zorder=60)
    ax.text(0.20, -3.57,
            f"θ = {np.degrees(theta):+5.1f}°",
            color='#aaaacc', ha='left', va='center',
            fontsize=7.5, fontfamily='monospace', zorder=60)
    ax.text(-1.60, -3.73,
            f"ticks = {tick_count:3d}",
            color=GOLD, ha='left', va='center',
            fontsize=7.5, fontfamily='monospace', zorder=60, fontweight='bold')

    # ── Title ─────────────────────────────────────────────────────
    ax.text(0, 2.02,
            "Deadbeat Escapement  —  Clock One",
            color='#9090b8', ha='center', va='top',
            fontsize=7.8, zorder=65, fontstyle='italic')

    plt.tight_layout(pad=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100,
                bbox_inches='tight', facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

PLOT_STYLE = {
    'figure.facecolor': '#12121f',
    'axes.facecolor':   '#12121f',
    'axes.edgecolor':   '#2a2a45',
    'axes.labelcolor':  '#aaaacc',
    'xtick.color':      '#666688',
    'ytick.color':      '#666688',
    'grid.color':       '#1e1e36',
    'text.color':       '#aaaacc',
    'font.family':      'sans-serif',
}


def plot_time_series(t_free, theta_free, t_esc, theta_esc):
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        fig.suptitle("Pendulum Angular Displacement vs Time",
                     fontsize=12, fontweight='bold', color='#c8c8e8', y=1.02)

        # Free decay
        ax1.plot(t_free, np.degrees(theta_free),
                 color='#E05050', lw=1.3, alpha=0.9)
        ax1.axhline(0, color='#3a3a55', lw=0.8, ls='--')
        ax1.set_title("Without Escapement  →  Decay to Rest",
                      color='#c8c8e8', fontsize=10, pad=8)
        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("θ  (degrees)", fontsize=9)
        ax1.grid(alpha=0.35, lw=0.6)
        ax1.fill_between(t_free, np.degrees(theta_free), alpha=0.08, color='#E05050')
        ax1.text(0.97, 0.95, "Damping wins —\nno energy input",
                 transform=ax1.transAxes, ha='right', va='top',
                 fontsize=8.5, color='#E05050',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#1a0a0a', ec='#E05050', alpha=0.8))
        ax1.spines[:].set_edgecolor('#2a2a45')

        # Escapement driven
        ax2.plot(t_esc, np.degrees(theta_esc),
                 color='#5A9FD4', lw=1.3, alpha=0.9)
        ax2.axhline(0, color='#3a3a55', lw=0.8, ls='--')
        ax2.set_title("With Escapement  →  Sustained Oscillation",
                      color='#c8c8e8', fontsize=10, pad=8)
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.grid(alpha=0.35, lw=0.6)
        ax2.fill_between(t_esc, np.degrees(theta_esc), alpha=0.08, color='#5A9FD4')
        ax2.text(0.97, 0.95, "Impulse replenishes\nenergy each tick",
                 transform=ax2.transAxes, ha='right', va='top',
                 fontsize=8.5, color='#5A9FD4',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#0a0f1a', ec='#5A9FD4', alpha=0.8))
        ax2.spines[:].set_edgecolor('#2a2a45')

        plt.tight_layout()
        return fig


def plot_phase_portrait(t_free, theta_free, omega_free,
                        t_esc,  theta_esc,  omega_esc):
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
        fig.suptitle("Phase Portrait  (θ  vs  θ̇)",
                     fontsize=12, fontweight='bold', color='#c8c8e8', y=1.02)

        # Free spiral
        ax1.plot(np.degrees(theta_free), np.degrees(omega_free),
                 color='#E05050', lw=0.9, alpha=0.85)
        ax1.plot(np.degrees(theta_free[0]),  np.degrees(omega_free[0]),
                 'o', color='#44DD88', ms=8, label='Start', zorder=5)
        ax1.plot(np.degrees(theta_free[-1]), np.degrees(omega_free[-1]),
                 's', color='#ffffff', ms=7, label='End', zorder=5)
        ax1.set_title("Free Pendulum  →  Spiral to Rest",
                      color='#c8c8e8', fontsize=10, pad=8)
        ax1.set_xlabel("θ  (degrees)", fontsize=9)
        ax1.set_ylabel("θ̇  (degrees/s)", fontsize=9)
        ax1.axhline(0, color='#3a3a55', lw=0.6)
        ax1.axvline(0, color='#3a3a55', lw=0.6)
        ax1.legend(fontsize=8.5, facecolor='#1a1a2e', edgecolor='#2a2a45',
                   labelcolor='#aaaacc')
        ax1.grid(alpha=0.3, lw=0.6)
        ax1.spines[:].set_edgecolor('#2a2a45')

        # Limit cycle
        cut = len(theta_esc) // 4
        ax2.plot(np.degrees(theta_esc[:cut]), np.degrees(omega_esc[:cut]),
                 color='#3a5a7a', lw=0.8, label='Transient', alpha=0.7)
        ax2.plot(np.degrees(theta_esc[cut:]), np.degrees(omega_esc[cut:]),
                 color='#5A9FD4', lw=1.6, label='Limit cycle')
        ax2.set_title("Escapement-Driven  →  Limit Cycle",
                      color='#c8c8e8', fontsize=10, pad=8)
        ax2.set_xlabel("θ  (degrees)", fontsize=9)
        ax2.set_ylabel("θ̇  (degrees/s)", fontsize=9)
        ax2.axhline(0, color='#3a3a55', lw=0.6)
        ax2.axvline(0, color='#3a3a55', lw=0.6)
        ax2.legend(fontsize=8.5, facecolor='#1a1a2e', edgecolor='#2a2a45',
                   labelcolor='#aaaacc')
        ax2.grid(alpha=0.3, lw=0.6)
        ax2.spines[:].set_edgecolor('#2a2a45')

        plt.tight_layout()
        return fig


def plot_energy(t_esc, theta_esc, omega_esc, L):
    """Energy plot showing PE, KE, and total over time."""
    with plt.rc_context(PLOT_STYLE):
        m = 1.0  # normalised mass
        KE = 0.5 * m * (L * omega_esc)**2
        PE = m * g * L * (1 - np.cos(theta_esc))
        E_total = KE + PE

        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.plot(t_esc, KE,      color='#E8A030', lw=1.2, label='Kinetic Energy',   alpha=0.9)
        ax.plot(t_esc, PE,      color='#5A9FD4', lw=1.2, label='Potential Energy', alpha=0.9)
        ax.plot(t_esc, E_total, color='#88DD88', lw=1.6, label='Total Energy',     alpha=0.9)
        ax.fill_between(t_esc, E_total, alpha=0.06, color='#88DD88')
        ax.set_title("Energy vs Time  (normalised, m = 1 kg)",
                     color='#c8c8e8', fontsize=10, pad=8)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Energy  (J)", fontsize=9)
        ax.legend(fontsize=8.5, facecolor='#1a1a2e', edgecolor='#2a2a45',
                  labelcolor='#aaaacc')
        ax.grid(alpha=0.3, lw=0.6)
        ax.spines[:].set_edgecolor('#2a2a45')
        plt.tight_layout()
        return fig


# =============================================================================
# STREAMLIT UI
# =============================================================================

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🕰️ Deadbeat Escapement Simulation</h1>
  <p>ME444 · Analysis and Design of Mechanical Systems · IIT Bombay &nbsp;|&nbsp; Clock One — 3D Printed Prototype</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙️ Parameters</p>', unsafe_allow_html=True)
    st.caption("Adjust and click **Run Simulation**")

    L       = st.slider("Pendulum length  L (m)",  0.30, 1.50, 0.876, 0.001,
                        help="Clock One prototype: 876 mm (gives ~1.88 s period)")
    theta0  = st.slider("Initial angle  θ₀ (°)",   2.0,  30.0,  8.0,  0.5,
                        help="Starting displacement of the pendulum")
    impulse = st.slider("Impulse strength  K",     0.00,  0.60,  0.25, 0.01,
                        help="Energy replenished by escapement per tick. Set 0 = free decay only.")

    st.markdown('<p class="section-header">🔒 Fixed Parameters</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      <b>Damping</b> b = {DAMPING}<br>
      <b>g</b> = {g} m/s²<br>
      <b>Escape wheel</b> = {N_TEETH} teeth<br>
      <b>Gear ratio</b> = 16 : 1<br>
      <b>Material</b> = PLA (3D printed)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    run = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    st.markdown('<p class="section-header">🎞 Animation</p>', unsafe_allow_html=True)
    anim_speed = st.select_slider("Speed",
                                   options=["0.5×", "1×", "1.5×", "2×"],
                                   value="1×")
    speed_map  = {"0.5×": 0.5, "1×": 1.0, "1.5×": 1.5, "2×": 2.0}
    anim_dur   = st.slider("Duration to animate (s)", 4, 16, 8, 2)

# ── Landing ───────────────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    <div class="info-box">
      👈 &nbsp; Adjust the parameters in the sidebar, then click <b>Run Simulation</b>.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **🔧 What is a Deadbeat Escapement?**
        A precision clock mechanism that converts continuous weight-driven rotation into
        accurate, discrete time steps. It *eliminates recoil* — the escape wheel never
        moves backward, making it more accurate than earlier anchor escapements.
        """)
    with col_b:
        st.markdown("""
        **📐 Theoretical Period**
        For a simple pendulum:
        > **T = 2π √(L / g)**

        At L = 876 mm → T ≈ 1.878 s.
        The Clock One prototype has 30 escape-wheel teeth, so one full revolution
        takes 60 seconds — driving a 60-tooth minute gear.
        """)
    with col_c:
        st.markdown("""
        **📊 Plots Generated**
        - **θ vs t** — Free decay vs sustained oscillation
        - **Phase portrait** — Spiral to rest vs limit cycle
        - **Energy** — KE, PE and total over time

        The limit cycle in the phase portrait is the signature of a self-sustaining
        oscillator — energy in = energy out each tick.
        """)
    st.stop()

# ── Run simulation ─────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text="Running ODE solver…")

with st.spinner(""):
    t_free, th_free, om_free = simulate_pendulum(
        L, theta0, DAMPING, impulse, t_end=60, escapement=False)
    progress_bar.progress(40, text="Free decay done…")

    t_esc, th_esc, om_esc = simulate_pendulum(
        L, theta0, DAMPING, impulse, t_end=60, escapement=True)
    progress_bar.progress(80, text="Escapement simulation done…")

T_theory = 2 * np.pi * np.sqrt(L / g)
T_sim    = compute_period(t_esc, th_esc)
error    = abs(T_sim - T_theory) / T_theory * 100 if not np.isnan(T_sim) else float('nan')

# Pre-compute escape wheel angle (one tooth per zero-crossing)
dtheta_tooth = 2 * np.pi / N_TEETH
esc_angles   = np.zeros(len(t_esc))
esc_cur      = 0.0
for k in range(1, len(t_esc)):
    if th_esc[k - 1] * th_esc[k] < 0:
        esc_cur += dtheta_tooth
    esc_angles[k] = esc_cur

progress_bar.progress(100, text="Done!")
time.sleep(0.3)
progress_bar.empty()

# ── Metrics row ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Validation Metrics</p>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Pendulum Length</div>
      <div class="metric-value">{L * 1000:.0f} mm</div>
      <div class="metric-sub">prototype: 876 mm</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Theoretical Period</div>
      <div class="metric-value">{T_theory:.4f} s</div>
      <div class="metric-sub">T = 2π√(L/g)</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Simulated Period</div>
      <div class="metric-value">{T_sim:.4f} s</div>
      <div class="metric-sub">from ODE solver</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Error</div>
      <div class="metric-value">{error:.3f}%</div>
      <div class="metric-sub">theory vs simulation</div>
    </div>""", unsafe_allow_html=True)
with m5:
    ticks_60s = int(esc_angles[-1] / dtheta_tooth)
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Ticks in 60 s</div>
      <div class="metric-value">{ticks_60s}</div>
      <div class="metric-sub">expected: {int(60 / (T_theory / 2))}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Main layout: animation + plots ────────────────────────────────────────────
col_anim, col_plots = st.columns([1, 2.1], gap="large")

with col_anim:
    st.markdown('<p class="section-header">🎞 Live Animation</p>',
                unsafe_allow_html=True)
    anim_slot  = st.empty()
    tick_badge = st.empty()

with col_plots:
    st.markdown('<p class="section-header">Plot 1 — Angular Displacement vs Time</p>',
                unsafe_allow_html=True)
    fig1 = plot_time_series(t_free, th_free, t_esc, th_esc)
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown('<p class="section-header">Plot 2 — Phase Portrait (θ vs θ̇)</p>',
                unsafe_allow_html=True)
    fig2 = plot_phase_portrait(t_free, th_free, om_free,
                               t_esc,  th_esc,  om_esc)
    st.pyplot(fig2)
    plt.close(fig2)

# Energy plot (full width)
st.markdown('<p class="section-header">Plot 3 — Mechanical Energy vs Time</p>',
            unsafe_allow_html=True)
fig3 = plot_energy(t_esc, th_esc, om_esc, L)
st.pyplot(fig3)
plt.close(fig3)

# ── Animation loop ─────────────────────────────────────────────────────────────
FPS       = 18
spd       = speed_map[anim_speed]
end_idx   = np.searchsorted(t_esc, anim_dur)
end_idx   = min(end_idx, len(t_esc) - 1)
N_FRAMES  = FPS * anim_dur
indices   = np.linspace(0, end_idx, N_FRAMES, dtype=int)
theta0_rad = np.radians(theta0)

tick_count = 0
for idx in indices:
    theta_f = float(th_esc[idx])
    esc_f   = float(esc_angles[idx])
    t_f     = float(t_esc[idx])
    tick_count = int(round(esc_f / dtheta_tooth))

    frame_bytes = draw_frame(theta_f, esc_f, tick_count, t_f, theta0_rad)
    anim_slot.image(frame_bytes, use_container_width=True)
    time.sleep(1.0 / (FPS * spd))

tick_badge.success(f"✅ Animation complete — **{tick_count} ticks** shown over {anim_dur} s")
