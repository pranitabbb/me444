# =============================================================================
# Deadbeat Escapement Simulation — Streamlit App
# ME444 Course Project
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
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
# CORE FUNCTIONS (unchanged from your Colab file)
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
# STREAMLIT UI
# =============================================================================

st.title("🕰️ Deadbeat Escapement — Pendulum Simulation")
st.caption("ME444 · Analysis and Design of Mechanical Systems · IIT Bombay")
st.markdown("---")

# ── Sidebar sliders ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")
    st.caption("Adjust and click **Run Simulation**")

    L       = st.slider("Pendulum length  L (m)",  0.3,  1.5,  0.876, 0.001,
                        help="Clock One prototype: 876 mm")
    theta0  = st.slider("Initial angle  θ₀ (°)",   2.0, 30.0,  8.0,  0.5)
    impulse = st.slider("Impulse strength  K",      0.0,  0.5,  0.25, 0.01,
                        help="Escapement kick per tick. Set to 0 to see free decay.")
    damping = 0.06   # fixed — matches PLA friction estimate

    st.markdown("---")
    run = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Fixed parameters**")
    st.markdown(f"- Damping b = {damping}")
    st.markdown(f"- g = {g} m/s²")
    st.markdown(f"- Escape wheel = 30 teeth")

# ── Instructions before first run ─────────────────────────────────────────────
if not run:
    st.info("👈  Adjust the sliders in the sidebar, then click **Run Simulation**.")
    st.markdown("""
    **What this simulation shows**

    | Plot | What to look for |
    |------|-----------------|
    | θ vs time (free) | Oscillation decays — damping removes energy |
    | θ vs time (driven) | Oscillation sustained — escapement replenishes energy |
    | Phase portrait | Free spiral inward vs driven limit cycle |

    **Prototype parameters** (Clock One, 3D printed PLA)
    - Pendulum length: 876 mm
    - Escape wheel: 30 teeth
    - Gear ratio: 16 : 1
    """)
    st.stop()

# ── Run simulation ─────────────────────────────────────────────────────────────
with st.spinner("Simulating..."):
    t_free, th_free, om_free = simulate_pendulum(
        L, theta0, damping, impulse, t_end=60, escapement=False)
    t_esc,  th_esc,  om_esc  = simulate_pendulum(
        L, theta0, damping, impulse, t_end=60, escapement=True)

T_theory = 2 * np.pi * np.sqrt(L / g)
T_sim    = compute_period(t_esc, th_esc)
error    = abs(T_sim - T_theory) / T_theory * 100

# ── Validation metrics ────────────────────────────────────────────────────────
st.subheader("📊 Period Validation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pendulum length", f"{L*1000:.0f} mm")
c2.metric("Theoretical period",  f"{T_theory:.4f} s")
c3.metric("Simulated period",    f"{T_sim:.4f} s")
c4.metric("Error", f"{error:.3f} %",
          delta=f"{error:.3f}%", delta_color="inverse")

st.markdown("---")

# ── Plot 1: θ vs time ─────────────────────────────────────────────────────────
st.subheader("Plot 1 — Angular Displacement vs Time")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
fig1.suptitle("Free Decay  vs  Escapement-Driven", fontsize=12, fontweight='bold')

ax1.plot(t_free, np.degrees(th_free), color='tomato', lw=1.2)
ax1.axhline(0, color='grey', lw=0.6, ls='--')
ax1.set_title("Without Escapement → Motion Dies Out")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("θ (degrees)"); ax1.grid(alpha=0.3)
ax1.text(0.97, 0.95, "Damping wins —\nno energy input",
         transform=ax1.transAxes, ha='right', va='top', fontsize=9, color='tomato',
         bbox=dict(boxstyle='round', fc='white', ec='tomato', alpha=0.7))

ax2.plot(t_esc, np.degrees(th_esc), color='steelblue', lw=1.2)
ax2.axhline(0, color='grey', lw=0.6, ls='--')
ax2.set_title("With Escapement → Sustained Oscillation")
ax2.set_xlabel("Time (s)"); ax2.grid(alpha=0.3)
ax2.text(0.97, 0.95, "Impulse replenishes\nenergy each tick",
         transform=ax2.transAxes, ha='right', va='top', fontsize=9, color='steelblue',
         bbox=dict(boxstyle='round', fc='white', ec='steelblue', alpha=0.7))

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ── Plot 2: Phase portrait ────────────────────────────────────────────────────
st.subheader("Plot 2 — Phase Portrait  (θ vs θ̇)")

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 4))
fig2.suptitle("Phase Portrait", fontsize=12, fontweight='bold')

ax3.plot(np.degrees(th_free), np.degrees(om_free), color='tomato', lw=1.0)
ax3.plot(np.degrees(th_free[0]),  np.degrees(om_free[0]),  'o', color='green', ms=7, label='Start')
ax3.plot(np.degrees(th_free[-1]), np.degrees(om_free[-1]), 's', color='black', ms=7, label='End')
ax3.set_title("Free Pendulum → Spiral to Rest")
ax3.set_xlabel("θ (degrees)"); ax3.set_ylabel("θ̇ (degrees/s)")
ax3.axhline(0, color='grey', lw=0.5); ax3.axvline(0, color='grey', lw=0.5)
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

cut = len(th_esc) // 4
ax4.plot(np.degrees(th_esc[:cut]), np.degrees(om_esc[:cut]),
         color='lightsteelblue', lw=0.8, label='Transient')
ax4.plot(np.degrees(th_esc[cut:]), np.degrees(om_esc[cut:]),
         color='steelblue', lw=1.4, label='Limit cycle')
ax4.set_title("Escapement-Driven → Limit Cycle")
ax4.set_xlabel("θ (degrees)"); ax4.set_ylabel("θ̇ (degrees/s)")
ax4.axhline(0, color='grey', lw=0.5); ax4.axvline(0, color='grey', lw=0.5)
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)
