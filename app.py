# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.9.0)
# - Timer MLSS autonome avec st_autorefresh (mise à jour chaque seconde)
# - Boutons Démarrer / Arrêter
# - Alerte visuelle + sonore (bip.wav en boucle 2s) toutes les 5 min
# - Vitesse MLSS liée dynamiquement à SV2 (96%) + affichage clair
# - Suggestion vitesse cohérente (priorité à SV2, ajustement borné)
# - Graphique MLSS : droites reliant les points, fond vert/rouge, badge Stable/Instable
# - Bouton Reset tableau MLSS
# - Onglet SRS complet
# - Logo sur chaque onglet avec fallback

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.9.0"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

LOGO_PATH = "logo.png"
BEEP_PATH = "beep.wav"

# -------------------- Helpers --------------------
def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("### Wild Training")

def suggestion_step_from_slope(slope):
    return float(np.clip(abs(slope) * 10, 0.2, 0.8))

def mlss_stability_metrics(t, lac):
    t = np.array(t, dtype=float)
    lac = np.array(lac, dtype=float)
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
    except:
        delta = None
    if (slope is not None) and (delta is not None):
        stable = (abs(slope) <= 0.02) and (abs(delta) <= 0.5)
    return slope, delta, stable

# -------------------- Sidebar --------------------
show_logo()
st.sidebar.header("Paramètres")
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
        nom = st.text_input("Nom", st.session_state.get("nom", ""), key="nom")
        prenom = st.text_input("Prénom", st.session_state.get("prenom", ""), key="prenom")
        dob = st.date_input("Date de naissance", key="dob")
        sexe = st.selectbox("Sexe", ["Homme", "Femme"], index=0, key="sexe")
        poids = st.number_input("Poids (kg)", 30.0, 150.0, st.session_state.get("poids", 70.0), key="poids")
        taille = st.number_input("Taille (cm)", 100.0, 220.0, st.session_state.get("taille", 175.0), key="taille")
        club = st.text_input("Club", st.session_state.get("club", ""), key="club")
        email = st.text_input("Email", st.session_state.get("email", ""), key="email")
        tel = st.text_input("Téléphone", st.session_state.get("tel", ""), key="tel")
        sub = st.form_submit_button("Enregistrer")
        if sub:
            st.success("Fiche enregistrée")

# -------------------- Analyse Lactate --------------------
with lactate_tab:
    show_logo()
    st.markdown("### Outil Analyse Lactate")
    st.markdown('https://www.exphyslab.com/lactate')

# -------------------- MLSS --------------------
with mlss_tab:
    show_logo()
    st.markdown("### MLSS – Palier 30 min")

    # Auto-refresh pour timer
    st_autorefresh = st.experimental_rerun  # simulate auto-refresh
    st.experimental_set_query_params()

    # Boutons Timer
    col_timer = st.columns(3)
    with col_timer[0]:
        if st.button("▶️ Démarrer", key="start_btn"):
            st.session_state["timer_running"] = True
            st.session_state["start_time"] = time.time()
    with col_timer[1]:
        if st.button("⏹️ Arrêter", key="stop_btn"):
            st.session_state["timer_running"] = False
    with col_timer[2]:
        if st.button("🔄 Reset tableau MLSS", key="reset_mlss"):
            st.session_state["df_mlss"] = pd.DataFrame(
                {"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan}
            )

    # Timer
    remaining = 1800
    if st.session_state.get("timer_running", False) and ("start_time" in st.session_state):
        elapsed = int(time.time() - st.session_state["start_time"])
        remaining = max(0, 1800 - elapsed)
    m, s = divmod(remaining, 60)
    st.markdown(f"⏱ Temps restant : **{m:02d}:{s:02d}**")

    # Alerte toutes les 5 min
    if remaining > 0 and remaining % 300 < 2:
        st.error("⚠️ Point lactate !")
        if os.path.exists(BEEP_PATH):
            st.audio(BEEP_PATH)

    # SV2 -> vitesse MLSS
    def update_v_target_from_sv2():
        sv2_val = float(st.session_state.get("sv2", 0.0))
        st.session_state["v_target"] = float(round(sv2_val * 0.96, 1)) if sv2_val > 0 else float(round(vma * 0.85, 1))

    sv2 = st.number_input("SV2 (km/h)", 0.0, 30.0, st.session_state.get("sv2", 0.0),
                          step=0.1, key="sv2", on_change=update_v_target_from_sv2)
    if "v_target" not in st.session_state:
        update_v_target_from_sv2()

    st.caption(f"Vitesse initiale = 96% SV2 ({st.session_state['v_target']:.1f} km/h)")
    v_target = st.number_input("Vitesse cible MLSS (km/h)", 5.0, 30.0,
                               st.session_state["v_target"], step=0.1, key="v_target")

    # Tableau MLSS
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame(
            {"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan}
        )

    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(st.session_state["df_mlss"], num_rows="fixed", hide_index=True, key="mlss_editor")
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit

    # Analyse + Graphique
    df = st.session_state["df_mlss"].copy()
    t = pd.to_numeric(df["Temps (min)"], errors="coerce").to_numpy()
    lac = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce").to_numpy()
    hr = pd.to_numeric(df["FC (bpm)"], errors="coerce").to_numpy()

    slope, delta_10_30, stable = mlss_stability_metrics(t, lac)

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

    # Suggestion vitesse
    if slope is not None and delta_10_30 is not None:
        step = suggestion_step_from_slope(slope)
        vt = float(st.session_state["v_target"])
        vs_96 = float(round(st.session_state.get("sv2", 0.0) * 0.96, 1)) if st.session_state.get("sv2", 0.0) > 0 else None

        if not stable:
            if slope > 0.02:
                candidate = max(5.0, vt - step)
                suggestion = vs_96 if vs_96 and vs_96 < vt else candidate
                rationale = "Lactate en hausse → baisse vitesse"
            elif slope < -0.02:
                candidate = min(30.0, vt + step)
                suggestion = vs_96 if vs_96 and vs_96 > vt else candidate
                rationale = "Lactate en baisse → augmente vitesse"
            else:
                suggestion = vt
                rationale = "Pente quasi nulle"
            st.info(f"Proposition vitesse ajustée : {suggestion:.1f} km/h. {rationale}")
        else:
            st.success("Lactate stable → conserver vitesse testée")

# -------------------- SRS --------------------
with srs_tab:
    show_logo()
    st.markdown("### Paramétrage SRS")
    slope_r = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1, key="slope_r")
    sv1 = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="sv1")
    sv2_srs = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="sv2_srs")
    delta_step2 = st.number_input("Delta Step2 (km/h)", -5.0, 5.0, -0.8, step=0.1, key="delta_step2")
    step1 = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="step1")
    vo2_1 = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1, key="vo2_1")
    v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1, key="v_equiv1")
    vo2_2 = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1, key="vo2_2")
    v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1, key="v_equiv2")

    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 else "—")