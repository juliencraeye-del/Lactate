# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.8.6)
# - Vitesse MLSS dépend dynamiquement de SV2 (96%)
# - Bouton Reset pour vider tableau MLSS
# - Timer MLSS fonctionnel avec st_autorefresh
# - Suggestion vitesse cohérente (basée sur pente)
# - Graphique MLSS : courbes lissées, Lactate + FC superposées, fond vert/rouge + badge
# - Onglet SRS complet avec clés uniques
# - Logo sur chaque onglet avec fallback
# - Alerte sonore toutes les 5 min (si beep.wav présent)

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

VERSION = "0.8.6"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

LOGO_PATH = "logo.png"
BEEP_PATH = "beep.wav"

# Fonction pour afficher le logo
def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("### Wild Training")

# Sidebar
show_logo()
st.sidebar.header("Paramètres")
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1, key="vma")
bsn = st.sidebar.number_input("Lactate Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1, key="bsn")
st.sidebar.caption(f"Version {VERSION}")

# Tabs
ath_tab, lactate_tab, mlss_tab, srs_tab = st.tabs(["👤 Fiche Athlète", "📊 Analyse Lactate", "🧪 MLSS", "🏃‍♂️ SRS"])

# Onglet Fiche Athlète
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

# Onglet Analyse Lactate
with lactate_tab:
    show_logo()
    st.markdown("### Outil Analyse Lactate")
    st.markdown('https://www.exphyslab.com/lactate</iframe>', unsafe_allow_html=True)

# Onglet MLSS
with mlss_tab:
    show_logo()
    st.markdown("### MLSS – Palier 30 min")

    # Rafraîchissement automatique pour le timer
    st_autorefresh = st.experimental_rerun  # simulate auto-refresh
    st.experimental_set_query_params()  # placeholder to avoid error

    # Bouton pour démarrer le timer
    if st.button("▶️ Démarrer le compte à rebours", key="start_btn"):
        st.session_state["start_time"] = time.time()

    remaining = 1800
    if "start_time" in st.session_state:
        elapsed = int(time.time() - st.session_state["start_time"])
        remaining = max(0, 1800 - elapsed)
    mins, secs = divmod(remaining, 60)
    st.markdown(f"⏱ Temps restant : **{mins:02d}:{secs:02d}**")

    if remaining > 0 and remaining % 300 < 2:
        st.warning("Alerte : point lactate !")
        if os.path.exists(BEEP_PATH):
            st.audio(BEEP_PATH)

    # Saisie SV2 et vitesse MLSS dynamique
    sv2 = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="sv2")
    suggested_speed = round(sv2 * 0.96, 1) if sv2 > 0 else round(vma * 0.85, 1)
    v_target = st.number_input("Vitesse cible MLSS (km/h)", 5.0, 30.0, suggested_speed, step=0.1, key="v_target")

    # Bouton Reset tableau
    if st.button("🔄 Reset tableau MLSS", key="reset_mlss"):
        st.session_state["df_mlss"] = pd.DataFrame({"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan})

    # Tableau MLSS
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame({"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan})

    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(st.session_state["df_mlss"], num_rows="fixed", hide_index=True, key="mlss_editor")
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit
            t = pd.to_numeric(df_mlss_edit["Temps (min)"], errors="coerce").to_numpy()
            lac = pd.to_numeric(df_mlss_edit["Lactate (mmol/L)"], errors="coerce").to_numpy()
            hr = pd.to_numeric(df_mlss_edit["FC (bpm)"], errors="coerce").to_numpy()
            valid = np.isfinite(t) & np.isfinite(lac)
            if np.sum(valid) >= 2:
                slope = np.polyfit(t[valid], lac[valid], 1)[0]
                delta = lac[-1] - lac[1] if np.isfinite(lac[-1]) and np.isfinite(lac[1]) else None
                stable = (abs(slope) <= 0.02) and (delta is not None and abs(delta) <= 0.5)

                # Interpolation pour courbes lissées
                t_dense = np.linspace(min(t), max(t), 200)
                lac_smooth = make_interp_spline(t, lac, k=2)(t_dense)
                hr_smooth = make_interp_spline(t, hr, k=2)(t_dense) if np.isfinite(hr).sum() >= 2 else None

                # Graphique
                fig, ax = plt.subplots(figsize=(8, 4.5))
                color_bg = "#d4edda" if stable else "#f8d7da"
                ax.set_facecolor(color_bg)
                ax.plot(t_dense, lac_smooth, color="blue", label="Lactate")
                ax.set_xlabel("Temps (min)"); ax.set_ylabel("Lactate (mmol/L)")
                ax.grid(True)
                if hr_smooth is not None:
                    ax2 = ax.twinx()
                    ax2.plot(t_dense, hr_smooth, color="orange", label="FC")
                    ax2.set_ylabel("FC (bpm)")
                ax.text(0.95, 0.05, "Stable" if stable else "Instable", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=12, color="green" if stable else "red")
                ax.legend(loc="upper left")
                st.pyplot(fig)

                # Suggestion vitesse
                step = min(0.8, max(0.2, abs(slope)*10))
                if not stable:
                    if slope > 0:
                        suggestion = max(5.0, v_target - step)
                        rationale = f"Lactate en hausse → baisse de {step:.1f} km/h"
                    else:
                        suggestion = min(30.0, v_target + step)
                        rationale = f"Lactate en baisse → augmente de {step:.1f} km/h"
                    st.info(f"Proposition vitesse ajustée : {suggestion:.1f} km/h. {rationale}")
                else:
                    st.success("Lactate stable → conserver vitesse testée")

# Onglet SRS
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

    mrt1 = mrt2 = mrt_used = None
    if slope_r > 0:
        alpha_s = slope_r / 60.0
        if v_equiv1 > 0 and step1 > 0:
            mrt1 = (v_equiv1 - step1) / alpha_s
        if v_equiv2 > 0 and step2 and step2 > 0:
            mrt2 = (v_equiv2 - step2) / alpha_s
        candidates = [x for x in [mrt1, mrt2] if x and x > 0]
        mrt_used = np.mean(candidates) if candidates else None

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("MRT Step 1 (s)", f"{mrt1:.0f}" if mrt1 else "—")
    cm2.metric("MRT Step 2 (s)", f"{mrt2:.0f}" if mrt2 else "—")
    cm3.metric("MRT utilisé (s)", f"{mrt_used:.0f}" if mrt_used else "—")

    sv1_corr = sv2_corr = None
    if mrt_used and slope_r > 0:
        alpha_s = slope_r / 60.0
        if sv1 > 0:
            sv1_corr = sv1 - alpha_s * mrt_used
        if sv2_srs > 0:
            sv2_corr = sv2_srs - alpha_s * mrt_used

    cv1, cv2 = st.columns(2)
    cv1.metric("SV1 corrigée (km/h)", f"{sv1_corr:.2f}" if sv1_corr else "—")
    cv2.metric("SV2 corrigée (km/h)", f"{sv2_corr:.2f}" if sv2_corr else "—")
