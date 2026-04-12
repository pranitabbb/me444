# =============================================================================
# Deadbeat Escapement Simulation
# ME444 · Analysis and Design of Mechanical Systems · IIT Bombay
# =============================================================================

import io, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deadbeat Escapement · ME444",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global constants ──────────────────────────────────────────────────────────
G       = 9.81
N_TEETH = 30
DAMPING = 0.06

# ── One colour palette used everywhere ───────────────────────────────────────
C = dict(
    bg      = "#0D0D14",
    surface = "#13131E",
    border  = "#22223A",
    gold    = "#C8943A",
    gold_lt = "#E0B050",
    steel   = "#6A8FA8",
    text    = "#C8C8D8",
    dim     = "#555570",
    red     = "#C04040",
    blue    = "#4080B0",
    green   = "#4A9060",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [class*="css"] {{
      font-family: 'Inter','Segoe UI',sans-serif;
      background:{C['bg']}; color:{C['text']};
  }}
  [data-testid="stSidebar"] {{
      background:{C['surface']} !important;
      border-right:1px solid {C['border']};
  }}
  [data-testid="stSidebar"] * {{ color:{C['text']} !important; }}
  .stButton > button {{
      background:{C['gold']} !important; color:#0D0D14 !important;
      font-weight:700 !important; border:none !important;
      border-radius:6px !important; width:100%; padding:0.55rem 0 !important;
      letter-spacing:0.06em !important;
  }}
  .stButton > button:hover {{ background:{C['gold_lt']} !important; }}
  .eng-card {{
      background:{C['surface']}; border:1px solid {C['border']};
      border-top:2px solid {C['gold']}; border-radius:6px;
      padding:0.8rem 1rem; text-align:center;
  }}
  .eng-card .lbl {{
      font-size:0.62rem; text-transform:uppercase;
      letter-spacing:0.12em; color:{C['dim']}; margin-bottom:0.3rem;
  }}
  .eng-card .val {{
      font-size:1.4rem; font-weight:700;
      color:{C['gold_lt']}; line-height:1;
  }}
  .eng-card .sub {{
      font-size:0.68rem; color:{C['dim']}; margin-top:0.25rem;
  }}
  .sec {{
      font-size:0.60rem; text-transform:uppercase; letter-spacing:0.14em;
      color:{C['dim']}; border-bottom:1px solid {C['border']};
      padding-bottom:0.3rem; margin-bottom:0.6rem;
  }}
  #MainMenu, footer, [data-testid="stDecoration"] {{ display:none; visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIMULATION
# =============================================================================

def simulate(L, theta0_deg, b, K, t_end=60, driven=True):
    w0   = np.sqrt(G / L)
    th0  = np.radians(theta0_deg)
    zone = np.radians(3.5)

    def ode(t, y):
        th, om = y
        kick = K * np.sign(om) if driven and abs(th) < zone and abs(om) > 0.01 else 0.0
        return [om, -w0**2 * np.sin(th) - b * om + kick]

    sol = solve_ivp(ode, (0, t_end), [th0, 0.0],
                    t_eval=np.linspace(0, t_end, t_end * 500),
                    method='RK45', rtol=1e-9, atol=1e-12, max_step=0.004)
    return sol.t, sol.y[0], sol.y[1]


def measured_period(t, theta):
    half = len(theta) // 2
    peaks, _ = find_peaks(theta[half:], height=0.01)
    return float(np.mean(np.diff(t[half:][peaks]))) if len(peaks) >= 2 else float('nan')


# =============================================================================
# ANIMATION  — clean mechanical style
# =============================================================================

def draw_frame(theta, esc_angle, tick, t_now):
    EX, EY = 0.0, 0.0
    ER     = 1.0
    FX, FY = 0.0, EY + ER + 0.72
    PL     = 3.2

    fig, ax = plt.subplots(figsize=(4.0, 6.5))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-4.2, 2.2)
    ax.set_aspect('equal'); ax.axis('off')

    # palette
    WHL  = "#B8822A"; WHLL = "#D4A844"; TOOTH = "#C89A30"
    FRK  = "#5A7D96"; FRHL = "#7AADCC"; ROD   = "#8898A8"
    BOB  = "#2E2E48"; BOBH = "#4A4A68"; PAL   = "#D4A030"
    HUB  = "#0E0E18"

    # ── escape wheel ──────────────────────────────────────────────
    R_r = ER * 0.68; R_h = ER * 0.16
    th_c = np.linspace(0, 2*np.pi, 300)
    ax.fill(EX + R_r*np.cos(th_c), EY + R_r*np.sin(th_c),
            color=WHL, zorder=5, alpha=0.96)
    # spokes
    for k in range(5):
        a = esc_angle + k * 2*np.pi/5
        ax.plot([EX, EX + R_r*0.9*np.cos(a)],
                [EY, EY + R_r*0.9*np.sin(a)],
                color=C['bg'], lw=3.2, zorder=6, solid_capstyle='round')
    # teeth
    pitch = 2*np.pi / N_TEETH
    for i in range(N_TEETH):
        a0 = esc_angle + i*pitch
        a1 = esc_angle + (i + 0.26)*pitch
        a2 = esc_angle + (i + 0.90)*pitch
        xs = [EX+R_r*np.cos(a0), EX+ER*np.cos(a1),
              EX+ER*0.88*np.cos(a2), EX+R_r*np.cos(a2)]
        ys = [EY+R_r*np.sin(a0), EY+ER*np.sin(a1),
              EY+ER*0.88*np.sin(a2), EY+R_r*np.sin(a2)]
        ax.fill(xs, ys, color=TOOTH, zorder=7, alpha=0.97)
    ax.add_patch(plt.Circle((EX,EY), R_r, fill=False,
                             ec=WHLL, lw=1.0, zorder=8, alpha=0.4))
    ax.add_patch(plt.Circle((EX,EY), R_h*1.3, color=HUB,  zorder=9))
    ax.add_patch(plt.Circle((EX,EY), R_h,     color=WHL,  zorder=10))
    ax.add_patch(plt.Circle((EX,EY), R_h*0.4, color=HUB,  zorder=11))

    # ── pallet fork ───────────────────────────────────────────────
    rock  = theta * 0.52
    spread = np.radians(26)
    aL = -np.pi/2 + spread + rock
    aR = -np.pi/2 - spread + rock
    arm = 0.78
    LTx = FX + arm*np.cos(aL); LTy = FY + arm*np.sin(aL)
    RTx = FX + arm*np.cos(aR); RTy = FY + arm*np.sin(aR)

    for tx, ty in [(LTx,LTy),(RTx,RTy)]:
        ax.plot([FX,tx],[FY,ty], color=FRK, lw=6, zorder=14, solid_capstyle='round')
        ax.plot([FX,tx],[FY,ty], color=FRHL, lw=2.8, zorder=15, solid_capstyle='round')

    pf = 0.16
    for tx, ty, aa in [(LTx,LTy,aL+np.pi/2),(RTx,RTy,aR-np.pi/2)]:
        px, py = pf*np.cos(aa), pf*np.sin(aa)
        ax.plot([tx-px,tx+px],[ty-py,ty+py],
                color=PAL, lw=7, solid_capstyle='round', zorder=16)
        ax.plot([tx-px,tx+px],[ty-py,ty+py],
                color=WHLL, lw=2.5, solid_capstyle='round', zorder=17, alpha=0.6)

    ax.add_patch(plt.Circle((FX,FY), 0.10, color=FRK, zorder=18))
    ax.add_patch(plt.Circle((FX,FY), 0.05, color=HUB, zorder=19))

    # ── pendulum ──────────────────────────────────────────────────
    bx = FX + PL*np.sin(theta)
    by = FY - PL*np.cos(theta)

    ax.plot([FX,bx],[FY,by], color='#000', lw=5, zorder=20, alpha=0.22,
            solid_capstyle='round')
    ax.plot([FX,bx],[FY,by], color=ROD,   lw=2.2, zorder=21, solid_capstyle='round')

    # adjusting nut
    nx = FX + 0.68*(bx-FX); ny = FY + 0.68*(by-FY)
    ax.add_patch(plt.Circle((nx,ny), 0.062, color=WHL, zorder=22))
    ax.add_patch(plt.Circle((nx,ny), 0.028, color=HUB, zorder=23))

    # bob
    ax.add_patch(plt.Circle((bx,by), 0.30, color='#000', zorder=24, alpha=0.28))
    ax.add_patch(plt.Circle((bx,by), 0.28, color=BOB,   zorder=25))
    ax.add_patch(plt.Circle((bx,by), 0.26, color=BOBH,  zorder=26))
    ax.add_patch(plt.Circle((bx,by), 0.27, fill=False, ec=ROD, lw=1.2, zorder=27))
    ax.add_patch(plt.Circle((bx-0.08,by+0.08), 0.07,
                             color='#8888AA', alpha=0.15, zorder=28))

    # pivot
    ax.add_patch(plt.Circle((FX,FY), 0.065, color=ROD, zorder=29))
    ax.add_patch(plt.Circle((FX,FY), 0.025, color=HUB, zorder=30))

    # ── mounting rail ─────────────────────────────────────────────
    ax.fill_between([-1.7,1.7],[FY+0.20]*2,[FY+0.48]*2,
                    color='#181828', zorder=3, alpha=0.95)
    ax.plot([-1.7,1.7],[FY+0.20]*2, color=C['border'], lw=1.0, zorder=4)
    for sx in [-1.35,-0.68,0.0,0.68,1.35]:
        ax.add_patch(plt.Circle((sx,FY+0.34), 0.045, color='#2A2A44', zorder=5))
        ax.add_patch(plt.Circle((sx,FY+0.34), 0.018, color='#444466', zorder=6))

    # ── telemetry strip ───────────────────────────────────────────
    sy = -4.05
    ax.fill_between([-1.9,1.9],[sy]*2,[sy+0.55]*2,
                    color='#0A0A12', alpha=0.9, zorder=40)
    ax.plot([-1.9,1.9],[sy+0.55]*2, color=C['border'], lw=0.6, zorder=41)

    def tt(x, y, s, col=C['dim'], sz=7.0):
        ax.text(x, y, s, color=col, fontsize=sz, ha='left',
                va='center', fontfamily='monospace', zorder=42)

    tt(-1.80, sy+0.40, f"t  = {t_now:6.2f} s",          col=C['text'])
    tt(-1.80, sy+0.17, f"θ  = {np.degrees(theta):+6.1f}°", col=C['steel'])
    tt( 0.30, sy+0.40, f"ticks = {tick:3d}",             col=C['gold_lt'], sz=7.5)
    tt( 0.30, sy+0.17, f"Δω = 2π/{N_TEETH} rad/tick",   col=C['dim'])

    ax.text(0, 2.10, "Deadbeat Escapement  ·  Clock One",
            color=C['dim'], ha='center', va='top',
            fontsize=7.2, fontstyle='italic', zorder=43)

    plt.tight_layout(pad=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=105,
                bbox_inches='tight', facecolor=C['bg'])
    buf.seek(0); plt.close(fig)
    return buf.getvalue()


# =============================================================================
# PLOTS
# =============================================================================

RC = {
    'figure.facecolor': C['bg'],  'axes.facecolor':  C['bg'],
    'axes.edgecolor':   C['border'], 'axes.labelcolor': C['text'],
    'xtick.color':      C['dim'],    'ytick.color':      C['dim'],
    'grid.color':       C['border'], 'grid.linewidth':   0.6,
    'font.size':        9,           'axes.titlesize':   9.5,
    'axes.titlecolor':  C['text'],   'axes.labelsize':   8.5,
    'legend.facecolor': C['surface'],'legend.edgecolor': C['border'],
    'legend.fontsize':  8,
}


def _ax_clean(ax):
    ax.grid(True, alpha=0.40)
    for sp in ax.spines.values():
        sp.set_edgecolor(C['border']); sp.set_linewidth(0.8)


def plot_displacement(t_free, th_free, t_esc, th_esc):
    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
        fig.subplots_adjust(wspace=0.06)
        for ax, t, th, col, lbl in [
            (ax1, t_free, th_free, C['red'],  "Free  —  no escapement"),
            (ax2, t_esc,  th_esc,  C['blue'], "Driven  —  with escapement"),
        ]:
            ax.plot(t, np.degrees(th), color=col, lw=1.0, alpha=0.9)
            ax.fill_between(t, np.degrees(th), alpha=0.07, color=col)
            ax.axhline(0, color=C['border'], lw=0.7, ls='--')
            ax.set_xlabel("Time  (s)"); ax.set_title(lbl, pad=6)
            _ax_clean(ax)
        ax1.set_ylabel("θ  (degrees)")
        fig.suptitle("Angular Displacement vs Time",
                     color=C['text'], fontsize=10, fontweight='600', y=1.01)
        return fig


def plot_phase(t_free, th_free, om_free, t_esc, th_esc, om_esc):
    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
        fig.subplots_adjust(wspace=0.18)

        ax1.plot(np.degrees(th_free), np.degrees(om_free),
                 color=C['red'], lw=0.8, alpha=0.85)
        ax1.plot(np.degrees(th_free[0]),  np.degrees(om_free[0]),
                 'o', color=C['green'], ms=6, zorder=5, label='Start')
        ax1.plot(np.degrees(th_free[-1]), np.degrees(om_free[-1]),
                 's', color=C['dim'],   ms=6, zorder=5, label='End')
        ax1.set_title("Free  →  Spiral to rest")
        ax1.set_xlabel("θ  (°)"); ax1.set_ylabel("θ̇  (°/s)")
        ax1.legend(); _ax_clean(ax1)

        cut = len(th_esc) // 4
        ax2.plot(np.degrees(th_esc[:cut]), np.degrees(om_esc[:cut]),
                 color=C['border'], lw=0.7, alpha=0.6, label='Transient')
        ax2.plot(np.degrees(th_esc[cut:]), np.degrees(om_esc[cut:]),
                 color=C['blue'],   lw=1.4, label='Limit cycle')
        ax2.set_title("Driven  →  Limit cycle")
        ax2.set_xlabel("θ  (°)"); ax2.set_ylabel("θ̇  (°/s)")
        ax2.legend(); _ax_clean(ax2)

        fig.suptitle("Phase Portrait",
                     color=C['text'], fontsize=10, fontweight='600', y=1.01)
        return fig


def plot_energy(t_esc, th_esc, om_esc, L):
    with plt.rc_context(RC):
        KE = 0.5*(L*om_esc)**2
        PE = G*L*(1 - np.cos(th_esc))
        E  = KE + PE
        fig, ax = plt.subplots(figsize=(12, 3.0))
        ax.plot(t_esc, KE, color=C['gold'],  lw=1.0, label='Kinetic  Eₖ')
        ax.plot(t_esc, PE, color=C['blue'],  lw=1.0, label='Potential  Eₚ')
        ax.plot(t_esc, E,  color=C['green'], lw=1.5, label='Total  E')
        ax.fill_between(t_esc, E, alpha=0.05, color=C['green'])
        ax.set_xlabel("Time  (s)")
        ax.set_ylabel("Energy  (J · kg⁻¹)")
        ax.set_title("Mechanical Energy vs Time  [normalised,  m = 1 kg]")
        ax.legend(ncol=3); _ax_clean(ax)
        return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def _sec(label):
    return (f'<p style="font-size:0.60rem;letter-spacing:0.14em;color:{C["dim"]};'
            f'text-transform:uppercase;border-bottom:1px solid {C["border"]};'
            f'padding-bottom:0.3rem;margin:0 0 0.8rem;">{label}</p>')

with st.sidebar:
    st.markdown(_sec("Parameters"), unsafe_allow_html=True)
    L       = st.slider("Pendulum length  L  (m)",  0.30, 1.50, 0.876, 0.001,
                        help="Clock One prototype: 876 mm")
    theta0  = st.slider("Initial angle  θ₀  (°)",   2.0, 30.0, 8.0, 0.5)
    impulse = st.slider("Impulse strength  K",       0.00, 0.60, 0.25, 0.01,
                        help="Set 0 to observe free decay")
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶  Run Simulation")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_sec("Fixed"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.76rem;color:{C["dim"]};line-height:2.0;">'
        f'b = {DAMPING}&emsp;(damping)<br>'
        f'g = {G} m/s²<br>'
        f'Teeth = {N_TEETH}<br>'
        f'Gear ratio = 16 : 1<br>'
        f'Material = PLA (3D printed)'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_sec("Animation"), unsafe_allow_html=True)
    spd_label = st.select_slider("Speed", ["0.5×","1×","1.5×","2×"], value="1×")
    anim_dur  = st.slider("Duration (s)", 4, 16, 8, 2)
    spd_map   = {"0.5×": 0.5, "1×": 1.0, "1.5×": 1.5, "2×": 2.0}


# =============================================================================
# MAIN PAGE
# =============================================================================

st.markdown(
    f'<h2 style="color:{C["gold_lt"]};font-size:1.5rem;font-weight:700;'
    f'margin:0 0 0.1rem;">⚙  Deadbeat Escapement Simulation</h2>'
    f'<p style="color:{C["dim"]};font-size:0.80rem;margin:0 0 1.2rem;">'
    f'ME444 · Analysis and Design of Mechanical Systems · IIT Bombay</p>',
    unsafe_allow_html=True
)

if not run:
    st.markdown(
        f'<div style="color:{C["dim"]};font-size:0.83rem;padding:1rem 1.2rem;'
        f'border:1px solid {C["border"]};border-radius:6px;'
        f'background:{C["surface"]};">'
        f'Adjust parameters in the sidebar, then click '
        f'<b style="color:{C["gold"]};">Run Simulation</b>.</div>',
        unsafe_allow_html=True)
    st.stop()

# ── Solve ─────────────────────────────────────────────────────────────────────
prog = st.progress(0, text="Solving ODE…")
t_fr, th_fr, om_fr = simulate(L, theta0, DAMPING, impulse, driven=False)
prog.progress(45, text="Escapement solution…")
t_es, th_es, om_es = simulate(L, theta0, DAMPING, impulse, driven=True)
prog.progress(90, text="Preparing output…")

T_th  = 2*np.pi*np.sqrt(L/G)
T_sim = measured_period(t_es, th_es)
err   = abs(T_sim - T_th)/T_th*100 if not np.isnan(T_sim) else float('nan')

d_tooth    = 2*np.pi / N_TEETH
esc_angles = np.zeros(len(t_es))
cur = 0.0
for k in range(1, len(t_es)):
    if th_es[k-1]*th_es[k] < 0:
        cur += d_tooth
    esc_angles[k] = cur

ticks_60 = int(esc_angles[-1] / d_tooth)
prog.progress(100); time.sleep(0.15); prog.empty()

# ── Animation + Plot 1 ────────────────────────────────────────────────────────
col_anim, col_p1 = st.columns([1, 2.0], gap="large")

with col_anim:
    st.markdown(f'<p class="sec">Live Animation</p>', unsafe_allow_html=True)
    anim_slot  = st.empty()
    tick_badge = st.empty()

with col_p1:
    st.markdown(f'<p class="sec">Angular Displacement vs Time</p>',
                unsafe_allow_html=True)
    fig1 = plot_displacement(t_fr, th_fr, t_es, th_es)
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)

# ── Validation metrics ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
cols = st.columns(5)
cards = [
    ("Pendulum Length",    f"{L*1000:.0f} mm",   "prototype 876 mm"),
    ("Theoretical Period", f"{T_th:.4f} s",        "T = 2π√(L/g)"),
    ("Simulated Period",   f"{T_sim:.4f} s",        "from ODE solver"),
    ("Error",              f"{err:.3f}%",            "theory vs simulation"),
    ("Ticks / min",        f"{ticks_60}",
     f"expected {int(60/T_th*2)}"),
]
for col, (lbl, val, sub) in zip(cols, cards):
    col.markdown(
        f'<div class="eng-card">'
        f'<div class="lbl">{lbl}</div>'
        f'<div class="val">{val}</div>'
        f'<div class="sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Phase portrait ────────────────────────────────────────────────────────────
st.markdown(f'<p class="sec">Phase Portrait</p>', unsafe_allow_html=True)
fig2 = plot_phase(t_fr, th_fr, om_fr, t_es, th_es, om_es)
st.pyplot(fig2, use_container_width=True); plt.close(fig2)

st.markdown("<br>", unsafe_allow_html=True)

# ── Energy ────────────────────────────────────────────────────────────────────
st.markdown(f'<p class="sec">Mechanical Energy</p>', unsafe_allow_html=True)
fig3 = plot_energy(t_es, th_es, om_es, L)
st.pyplot(fig3, use_container_width=True); plt.close(fig3)

# ── Animation loop ────────────────────────────────────────────────────────────
FPS     = 18
spd     = spd_map[spd_label]
end_idx = min(np.searchsorted(t_es, anim_dur), len(t_es)-1)
indices = np.linspace(0, end_idx, FPS * anim_dur, dtype=int)

tick_count = 0
for idx in indices:
    th_f  = float(th_es[idx])
    esc_f = float(esc_angles[idx])
    t_now = float(t_es[idx])
    tick_count = int(round(esc_f / d_tooth))

    anim_slot.image(
        draw_frame(th_f, esc_f, tick_count, t_now),
        use_container_width=True
    )
    time.sleep(1.0 / (FPS * spd))

tick_badge.markdown(
    f'<p style="font-size:0.76rem;color:{C["dim"]};margin-top:0.3rem;">'
    f'✓ &nbsp;{tick_count} ticks · {anim_dur} s</p>',
    unsafe_allow_html=True
)
