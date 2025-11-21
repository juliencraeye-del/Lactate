# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.8.7)
# - Vitesse MLSS dépend dynamiquement de SV2 (96%)
# - Boutons Démarrer/Arrêter + Reset pour MLSS
# - Timer autonome (sans interaction)
# - Graphique MLSS : courbes lissées (polyfit NumPy), Lactate + FC superposées, légende complète
# - Alerte couleur (vert/rouge) + badge "Stable/Instable" dans le graphique
# - Suggestion de vitesse cohérente (amplitude ∝ pente, bornée)
# - Onglet SRS complet avec clés uniques
# - Logo sur chaque onglet avec fallback
# - Alerte sonore toutes les 5 min (si beep.wav présent)

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.8.7"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

LOGO_PATH = "logo.png"   # place ton logo ici
BEEP_PATH = "beep.wav"   # place ton bip ici (optionnel)

# -------------------- Helpers --------------------
def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("### Wild Training")

def poly_smooth(x, y, n_points=200):
    """
    Lissage simple sans SciPy :
    - Si >=4 points : polyfit degré 3
    - Si 3 points   : polyfit degré 2
    - Si 2 points   : polyfit degré 1
    - Sinon         : retourne (None, None)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None, None
    deg = 1 if len(x) == 2 else (2 if len(x) == 3 else 3)
    try:
        coeffs = np.polyfit(x, y, deg)
        p = np.poly1d(coeffs)
        x_dense = np.linspace(np.min(x), np.max(x), n_points)
        y_hat = p(x_dense)
        return x_dense, y_hat
    except Exception:
        return None, None

def suggestion_step_from_slope(slope):
    """
    Amplitude de l'ajustement de vitesse basée sur la pente :
    0.2 à 0.8 km/h en fonction de |pente|
    """
    return float(np.clip(abs(slope) * 10, 0.2, 0.8))

def mlss_stability_metrics(t, lac):
    """
    Calcule pente globale et Δ10→30 ; renvoie (slope, delta, stable)
    Stabilité si |pente| ≤ 0.02 mmol·L⁻¹·min⁻¹ ET |Δ10→30| ≤ 0.5 mmol/L
    """
    t = np.array(t, dtype=float)
    lac = np.array(lac, dtype=float)
    mask = np.isfinite(t) & np.isfinite(lac)
    t, lac = t[mask], lac[mask]
    slope = None
    delta = None
    stable = False
    if len(t) >= 2:
        slope = np.polyfit(t, lac, 1)[0]
    # Δ10→30 si les points existent
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

def beep_if_needed(remaining_sec):
    """
    Joue le bip si présent à chaque multiple de 5 min.
    """
    if remaining_sec > 0 and remaining_sec % 300 == 0 and os.path.exists(BEEP_PATH):
        st.audio(BEEP_PATH)

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

# -------------------- Analyse Lactate (iframe) --------------------
with lactate_tab:
    show_logo()
    st.markdown("### Outil Analyse Lactate")
    # L'intégration directe d'un iframe peut être bloquée par CORS selon l'hébergeur ;
    # on affiche simplement le lien pour ouverture dans un nouvel onglet.
    st.markdown('https://www.exphyslab.com/lactate')

# -------------------- MLSS --------------------
with mlss_tab:
    show_logo()
    st.markdown("### MLSS – Palier 30 min")

    # 0) Contrôles Timer
    ctimer = st.columns(3)
    with ctimer[0]:
        if st.button("▶️ Démarrer", key="start_btn"):
            st.session_state["timer_running"] = True
            st.session_state["start_time"] = time.time()
    with ctimer[1]:
        if st.button("⏹️ Arrêter", key="stop_btn"):
            st.session_state["timer_running"] = False
    with ctimer[2]:
        if st.button("🔄 Reset tableau MLSS", key="reset_mlss"):
            st.session_state["df_mlss"] = pd.DataFrame(
                {"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan}
            )

    # Timer autonome (boucle contrôlée)
    placeholder_timer = st.empty()
    remaining = 1800
    if st.session_state.get("timer_running", False) and ("start_time" in st.session_state):
        # boucle de rafraîchissement – s'arrête si l'utilisateur change d'onglet
        # ou clique sur Arrêter ; mise à jour chaque seconde
        while st.session_state.get("timer_running", False):
            elapsed = int(time.time() - st.session_state["start_time"])
            remaining = max(0, 1800 - elapsed)
            m, s = divmod(remaining, 60)
            placeholder_timer.markdown(f"⏱ Temps restant : **{m:02d}:{s:02d}**")
            beep_if_needed(remaining)
            if remaining <= 0:
                st.session_state["timer_running"] = False
                break
            time.sleep(1)
    else:
        # affichage statique si le timer n'est pas actif
        placeholder_timer.markdown(f"⏱ Temps restant : **{remaining//60:02d}:{remaining%60:02d}**")

    # 1) Saisie SV2 → met à jour v_target dynamiquement à 96% SV2
    def update_v_target_from_sv2():
        sv2_val = float(st.session_state.get("sv2", 0.0))
        if sv2_val > 0:
            st.session_state["v_target"] = float(round(sv2_val * 0.96, 1))
        else:
            st.session_state["v_target"] = float(round(vma * 0.85, 1))

    sv2 = st.number_input("SV2 (km/h)", 0.0, 30.0, st.session_state.get("sv2", 0.0),
                          step=0.1, key="sv2", on_change=update_v_target_from_sv2)

    # valeur par défaut si pas encore définie
    if "v_target" not in st.session_state:
        update_v_target_from_sv2()

    v_target = st.number_input("Vitesse cible MLSS (km/h)", 5.0, 30.0,
                               st.session_state["v_target"], step=0.1,
                               key="v_target_input")
    # synchronise à chaque changement manuel de v_target
    st.session_state["v_target"] = float(st.session_state["v_target_input"])

    # 2) Tableau MLSS
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame(
            {"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan}
        )

    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(st.session_state["df_mlss"],
                                      num_rows="fixed", hide_index=True, key="mlss_editor")
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit

    # 3) Analyse + Graphique
    df = st.session_state["df_mlss"].copy()
    t = pd.to_numeric(df["Temps (min)"], errors="coerce").to_numpy()
    lac = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce").to_numpy()
    hr = pd.to_numeric(df["FC (bpm)"], errors="coerce").to_numpy()

    slope, delta_10_30, stable = mlss_stability_metrics(t, lac)

    # courbes lissées (polyfit-numPy)
    x_lac, y_lac = poly_smooth(t, lac, n_points=300)
    x_hr, y_hr = poly_smooth(t, hr, n_points=300) if np.isfinite(hr).sum() >= 2 else (None, None)

    fig, ax = plt.subplots(figsize=(9, 5))
    color_bg = "#d4edda" if stable else "#f8d7da"
    ax.set_facecolor(color_bg)

    # Lactate
    if x_lac is not None:
        ax.plot(x_lac, y_lac, color="#005a9e", linewidth=2.5, label="Lactate (mmol/L)")
    # points bruts visibles pour repères
    ax.scatter(t, lac, color="#005a9e", s=35, zorder=3)

    # Bsn
    ax.axhline(bsn, color="#767676", linestyle="--", linewidth=1, label=f"Bsn ≈ {bsn:.1f} mmol/L")
    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("Lactate (mmol/L)")
    ax.grid(True, alpha=0.3)

    # FC superposée (axe droit)
    handles, labels = ax.get_legend_handles_labels()
    if (x_hr is not None) and (y_hr is not None):
        ax2 = ax.twinx()
        ax2.plot(x_hr, y_hr, color="#d83b01", linewidth=2.0, label="FC (bpm)")
        ax2.scatter(t, hr, color="#d83b01", s=30, zorder=3)
        ax2.set_ylabel("FC (bpm)")
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    # Badge "Stable/Instable"
    ax.text(0.98, 0.04, "● Stable" if stable else "● Instable",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color="white",
            bbox=dict(boxstyle="round", facecolor=("#107c10" if stable else "#e81123"),
                      edgecolor="none", alpha=0.9))

    st.pyplot(fig)

    # 4) Suggestion vitesse (cohérente)
    if slope is not None and delta_10_30 is not None:
        step = suggestion_step_from_slope(slope)
        vt = float(st.session_state["v_target"])
        vs_96 = float(round(st.session_state.get("sv2", 0.0) * 0.96, 1)) if st.session_state.get("sv2", 0.0) > 0 else None

        if not stable:
            if slope > 0.02:   # lactate monte → trop vite
                candidate = max(5.0, vt - step)
                # si 96% SV2 < vt, suggérer d'abord de tendre vers 96%
                if vs_96 and vs_96 < vt:
                    suggestion = max(5.0, vs_96) if abs(vt - vs_96) >= step else candidate
                    rationale = "Lactate en hausse → rapproche-toi de ~96% SV2 ou baisse la vitesse."
                else:
                    suggestion = candidate
                    rationale = f"Lactate en hausse → baisse ≈ {step:.1f} km/h."
            elif slope < -0.02: # lactate baisse → vitesse trop faible
                candidate = min(30.0, vt + step)
                if vs_96 and vs_96 > vt:
                    suggestion = min(30.0, vs_96) if abs(vs_96 - vt) >= step else candidate
                    rationale = "Lactate en baisse → rapproche-toi de ~96% SV2 ou augmente la vitesse."
                else:
                    suggestion = candidate
                    rationale = f"Lactate en baisse → augmente ≈ {step:.1f} km/h."
            else:               # pente proche de zéro mais Δ non OK
                suggestion = vt
                rationale = "Pente quasi nulle : conserver et affiner ±0,1–0,2 km/h."
            st.info(f"**Proposition vitesse ajustée** : **{suggestion:.1f} km/h**. {rationale}")
        else:
            st.success("Lactate stable → conserver la vitesse testée.")

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

    mrt1 = mrt2 = mrt_used = None
    if slope_r > 0:
        alpha_s = slope_r / 60.0
        if v_equiv1 > 0 and step1 > 0:
            mrt1 = (v_equiv1 - step1) / alpha_s
        if v_equiv2 > 0 and step2 and step2 > 0:
            mrt2 = (v_equiv2 - step2) / alpha_s
        candidates = [x for x in [mrt1, mrt2] if x and x > 0]
        mrt_used = float(np.mean(candidates)) if candidates else None

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