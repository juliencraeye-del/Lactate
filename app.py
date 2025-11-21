# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.9.1)
# - Timer MLSS autonome (composant HTML/JS) : Start/Stop/Reset, alertes 5’, bip ~2 s
# - Vitesse MLSS liée dynamiquement à SV2 (96 %) via case à cocher
# - Suggestion vitesse = f(pente, Δ10→30), bornée 0,2–0,8 km/h (+ priorité 96 % SV2 si cohérente)
# - Graphique MLSS : droites reliant les points (Lactate + FC), fond vert/rouge, badge Stable/Instable
# - Reset tableau MLSS
# - Onglet SRS complet (MRT + corrections)
# - Logo sur chaque onglet (fallback si logo.png absent)

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

VERSION = "0.9.1"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

LOGO_PATH = "logo.png"   # facultatif

# -------------------- Helpers --------------------
def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("### Wild Training")

def suggestion_step(slope, delta):
    """
    Amplitude d’ajustement en km/h selon pente et dérive lactate (Δ10→30), bornée [0.2 ; 0.8]
    - Base = |pente| * 10
    - Bonus dérive : +0.2 si |Δ| > 0.5 ; +0.3 si |Δ| > 1.0 (cumulatif max jusqu’au bornage)
    """
    base = abs(slope) * 10
    if delta is not None:
        if abs(delta) > 0.5:
            base += 0.2
        if abs(delta) > 1.0:
            base += 0.3
    return float(np.clip(base, 0.2, 0.8))

def mlss_stability_metrics(t, lac):
    """
    Calcule (pente, Δ10→30, stable).
    Critères stabilité : |pente| ≤ 0.02 mmol·L⁻¹·min⁻¹ ET |Δ10→30| ≤ 0.5 mmol/L
    """
    t = np.array(t, dtype=float); lac = np.array(lac, dtype=float)
    mask = np.isfinite(t) & np.isfinite(lac)
    t, lac = t[mask], lac[mask]
    slope, delta, stable = None, None, False
    if len(t) >= 2:
        slope = np.polyfit(t, lac, 1)[0]
    try:
        v10 = lac[np.where(t == 10)[0][0]]
        v30 = lac[np.where(t == 30)[0][0]]
        if np.isfinite(v10) and np.isfinite(v30):
            delta = float(v30 - v10)
    except Exception:
        delta = None
    if (slope is not None) and (delta is not None):
        stable = (abs(slope) <= 0.02) and (abs(delta) <= 0.5)
    return slope, delta, stable

# -------------------- Sidebar --------------------
show_logo()
st.sidebar.header("Paramètres")
with st.sidebar:
    show_logo()
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1, key="vma")
bsn = st.sidebar.number_input("Lactate Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1, key="bsn")
st.sidebar.caption(f"Version {VERSION}")

# -------------------- Tabs --------------------
ath_tab, lactate_tab, mlss_tab, srs_tab = st.tabs(["👤 Fiche Athlète", "📊 Analyse Lactate", "🧪 MLSS", "🏃‍♂️ SRS"])

# -------------------- Fiche Athlète --------------------
with ath_tab:
    show_logo()
    st.markdown("### Fiche signalétique")
    with st.form("ath_form"):
        nom    = st.text_input("Nom",    st.session_state.get("nom", ""),    key="nom")
        prenom = st.text_input("Prénom", st.session_state.get("prenom", ""), key="prenom")
        dob    = st.date_input("Date de naissance", key="dob")
        sexe   = st.selectbox("Sexe", ["Homme", "Femme"], index=0, key="sexe")
        poids  = st.number_input("Poids (kg)", 30.0, 150.0, st.session_state.get("poids", 70.0), key="poids")
        taille = st.number_input("Taille (cm)", 100.0, 220.0, st.session_state.get("taille", 175.0), key="taille")
        club   = st.text_input("Club",   st.session_state.get("club", ""),   key="club")
        email  = st.text_input("Email",  st.session_state.get("email", ""),  key="email")
        tel    = st.text_input("Téléphone", st.session_state.get("tel", ""), key="tel")
        sub    = st.form_submit_button("Enregistrer")
        if sub:
            st.success("Fiche enregistrée")

# -------------------- Analyse Lactate (lien) --------------------
with lactate_tab:
    show_logo()
    st.markdown("### Outil Analyse Lactate")
    components.html('<iframe src="https://www.exphyslab.com/lactate" style="width:100%;height:800px;border:none;"></iframe>', height=820)
st.info('Privilégier **Bsln+0.5** pour SL1 et **modDmax** pour SL2')

# -------------------- MLSS --------------------
with mlss_tab:
    show_logo()
    st.markdown("### MLSS – Palier 30 min")

    # --- Composant Timer (HTML/JS) : autonome, bip 2s, alertes à 5,10,15,20,25,30 min ---
    st.markdown("#### ⏱ Compte à rebours (30:00)")
    html_timer = """
    <div id="mlss-timer" style="padding:8px;border:1px solid #ddd;border-radius:8px;max-width:460px;">
      <div id="timer-display" style="font:600 28px/1.2 ui-sans-serif,system-ui; margin-bottom:8px;">30:00</div>
      <div id="alert-banner" style="display:none;padding:6px 10px;margin-bottom:8px;border-radius:6px;background:#ffe5e5;color:#b00020;font-weight:600;">⚠️ Point lactate !</div>
      <div style="display:flex;gap:8px;">
        <button id="btn-start" style="padding:6px 12px;border:1px solid #666;border-radius:6px;background:#e6f4ff;cursor:pointer;">▶️ Démarrer</button>
        <button id="btn-stop"  style="padding:6px 12px;border:1px solid #666;border-radius:6px;background:#f1f1f1;cursor:pointer;">⏹️ Arrêter</button>
        <button id="btn-reset" style="padding:6px 12px;border:1px solid #666;border-radius:6px;background:#fff;cursor:pointer;">🔄 Reset</button>
      </div>
      <div style="margin-top:6px;font:13px/1.3 ui-sans-serif,system-ui;color:#555;">Alertes à 25:00, 20:00, 15:00, 10:00, 05:00, 00:00 (bip ~2 s)</div>
    </div>
    <script>
    (function(){
      const marks = [1500,1200,900,600,300,0]; // secondes restantes: 25',20',15',10',5',0
      let duration = 1800; // 30 min en s
      let remaining = duration;
      let running = false;
      let intervalId = null;
      let fired = {}; marks.forEach(m => fired[m] = false);

      const disp = document.getElementById('timer-display');
      const banner = document.getElementById('alert-banner');
      const fmt = (s)=>{ const m=Math.floor(s/60), r=s%60; return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`};

      function showBanner(show){
        banner.style.display = show ? 'block' : 'none';
      }
      function beep2s(){
        // WebAudio beep ~2 s
        try{
          const ctx = new (window.AudioContext||window.webkitAudioContext)();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.frequency.value = 880;
          osc.connect(gain); gain.connect(ctx.destination);
          gain.gain.setValueAtTime(0.2, ctx.currentTime);
          osc.start();
          setTimeout(()=>{ osc.stop(); ctx.close(); }, 2000);
        }catch(e){}
      }
      function tick(){
        if(!running) return;
        remaining = Math.max(0, remaining-1);
        disp.textContent = fmt(remaining);
        if (marks.includes(remaining) && !fired[remaining]){
          fired[remaining] = true;
          showBanner(true);
          beep2s();
          setTimeout(()=>showBanner(false), 2200);
        }
        if(remaining === 0){ stop(); }
      }
      function start(){
        if(running) return;
        running = true;
        if(!intervalId){ intervalId = setInterval(tick, 1000); }
      }
      function stop(){
        running = false;
        if(intervalId){ clearInterval(intervalId); intervalId = null; }
      }
      function reset(){
        stop();
        remaining = duration;
        disp.textContent = fmt(remaining);
        Object.keys(fired).forEach(k=>fired[k]=false);
        showBanner(false);
      }
      document.getElementById('btn-start').onclick = start;
      document.getElementById('btn-stop').onclick  = stop;
      document.getElementById('btn-reset').onclick = reset;
    })();
    </script>
    """
    components.html(html_timer, height=190)

    st.markdown("---")

    # --- Lier v_target à SV2 (96 %) ---
    col_sv2 = st.columns([1,1,1.2])
    with col_sv2[0]:
        sv2 = st.number_input("SV2 (km/h)", 0.0, 30.0, float(st.session_state.get("sv2", 0.0)),
                              step=0.1, key="sv2")
    with col_sv2[1]:
        link_sv2 = st.checkbox("Lier v_target à 96 % SV2", value=st.session_state.get("link_sv2", True), key="link_sv2")
    with col_sv2[2]:
        sv2_text = f"{0.96*sv2:.1f} km/h" if sv2 > 0 else f"{0.85*vma:.1f} km/h (≈85% VMA)"
        st.caption(f"Vitesse initiale = **{sv2_text}**")

    if ("v_target" not in st.session_state) or link_sv2:
        st.session_state["v_target"] = round(0.96*sv2, 1) if sv2 > 0 else round(0.85*vma, 1)

    v_target = st.number_input("Vitesse cible MLSS (km/h)", 5.0, 30.0,
                               float(st.session_state["v_target"]), step=0.1, key="v_target")

    # --- Tableau MLSS ---
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame(
            {"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan}
        )

    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(st.session_state["df_mlss"], num_rows="fixed",
                                      hide_index=True, key="mlss_editor")
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit

    # --- Analyse + Graphique ---
    df = st.session_state["df_mlss"].copy()
    t  = pd.to_numeric(df["Temps (min)"], errors="coerce").to_numpy()
    lac = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce").to_numpy()
    hr  = pd.to_numeric(df["FC (bpm)"], errors="coerce").to_numpy()

    slope, delta_10_30, stable = mlss_stability_metrics(t, lac)

    # Message en haut si instable
    if slope is not None and delta_10_30 is not None and not stable:
        st.error(f"⚠️ Lactate instable (Δ10→30 = {delta_10_30:.2f} mmol/L ; pente = {slope:.3f} mmol·L⁻¹·min⁻¹)")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#d4edda" if stable else "#f8d7da")

    ax.plot(t, lac, "o-", color="#005a9e", linewidth=2.5, label="Lactate (mmol/L)")
    ax.axhline(bsn, color="#767676", linestyle="--", linewidth=1, label=f"Bsn ≈ {bsn:.1f} mmol/L")
    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("Lactate (mmol/L)")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if np.isfinite(hr).sum() >= 2:
        ax2 = ax.twinx()
        ax2.plot(t, hr, "s-", color="#d83b01", linewidth=2.0, label="FC (bpm)")
        ax2.set_ylabel("FC (bpm)")
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    ax.text(0.98, 0.04, "● Stable" if stable else "● Instable",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color="white",
            bbox=dict(boxstyle="round", facecolor=("#107c10" if stable else "#e81123"),
                      edgecolor="none", alpha=0.9))

    st.pyplot(fig)

    # --- Suggestion vitesse (pente + dérive) ---
    if slope is not None and delta_10_30 is not None:
        step = suggestion_step(slope, delta_10_30)      # 0.2 … 0.8
        vt   = float(st.session_state["v_target"])
        vs96 = round(0.96*sv2, 1) if sv2 > 0 else None

        if not stable:
            if slope > 0.02:   # lactate monte → vitesse trop élevée
                candidate = max(5.0, vt - step)
                suggestion = vs96 if (vs96 and vs96 < vt) else candidate
                rationale  = f"Lactate en hausse → baisse ~{step:.1f} km/h"
            elif slope < -0.02: # lactate baisse → vitesse trop faible
                candidate = min(30.0, vt + step)
                suggestion = vs96 if (vs96 and vs96 > vt) else candidate
                rationale  = f"Lactate en baisse → augmente ~{step:.1f} km/h"
            else:
                suggestion = vt
                rationale  = "Pente quasi nulle : conserver / affiner ±0,1–0,2 km/h"
            st.info(f"**Proposition vitesse ajustée** : **{suggestion:.1f} km/h**. {rationale} (Δ10→30={delta_10_30:.2f})")
        else:
            st.success("Lactate stable → conserver la vitesse testée.")

# -------------------- SRS --------------------
with srs_tab:
    show_logo()
    st.markdown("### Paramétrage SRS")
    slope_r   = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1, key="slope_r")
    sv1       = st.number_input("SV1 (km/h)",              0.0, 30.0, 0.0, step=0.1, key="sv1")
    sv2_srs   = st.number_input("SV2 (km/h)",              0.0, 30.0, 0.0, step=0.1, key="sv2_srs")
    delta_step2 = st.number_input("Delta Step2 (km/h)",   -5.0, 5.0, -0.8, step=0.1, key="delta_step2")
    step1     = st.number_input("Vitesse Step 1 (km/h)",   0.0, 30.0, 0.0, step=0.1, key="step1")
    vo2_1     = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1, key="vo2_1")
    v_equiv1  = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1, key="v_equiv1")
    vo2_2     = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1, key="vo2_2")
    v_equiv2  = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1, key="v_equiv2")

    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 else "—")
# --- Calcul MRT (Mean Response Time) et seuils corrigés ---
valid_slope = slope_r > 0
mrt_vals_min = []
try:
    if valid_slope and step1 > 0 and v_equiv1 > 0:
        mrt1_min = (v_equiv1 - step1) / slope_r
        mrt_vals_min.append(mrt1_min)
except Exception:
    pass
try:
    if valid_slope and sv2_srs > 0 and v_equiv2 > 0:
        step2_calc = sv2_srs + delta_step2
        if step2_calc > 0:
            mrt2_min = (v_equiv2 - step2_calc) / slope_r
            mrt_vals_min.append(mrt2_min)
except Exception:
    pass
if len(mrt_vals_min) > 0:
    mrt_min = sum(mrt_vals_min) / len(mrt_vals_min)
    mrt_sec = mrt_min * 60
    st.metric("MRT (s)", f"{mrt_sec:.1f}")
    if sv1 > 0:
        sv1_corr = sv1 + slope_r * mrt_min
        st.metric("SV1 corrigé (km/h)", f"{sv1_corr:.2f}")
    if sv2_srs > 0:
        sv2_corr = sv2_srs + slope_r * mrt_min
        st.metric("SV2 corrigé (km/h)", f"{sv2_corr:.2f}")
    st.caption("Correction appliquée : v_corr = v_mesurée + slope_r × MRT_min")
else:
    st.info("Renseignez slope_r, Step 1/2 et vitesses équivalentes (v_equiv1/v_equiv2) pour estimer le MRT et corriger les seuils.")
