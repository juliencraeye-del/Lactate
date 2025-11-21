
import os
import time
import base64
import io
import wave
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "2.2"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")
LOGO_PATH = "logo.png"

# -------------------- Helpers --------------------
def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)
    else:
        st.markdown("### Wild Training")

def suggestion_step(slope, delta):
    """Amplitude d'ajustement (km/h) en f(pente, dérive Δ10→30), bornée [0.2 ; 0.8]."""
    base = abs(float(slope)) * 10.0 if slope is not None else 0.2
    if delta is not None:
        if abs(delta) > 0.5:
            base += 0.2
        if abs(delta) > 1.0:
            base += 0.3
    return float(np.clip(base, 0.2, 0.8))

def df_get_value(df, col_time, col_value, t_target):
    try:
        row = df.loc[df[col_time] == t_target, col_value]
        if len(row) == 1:
            val = pd.to_numeric(row.iloc[0], errors='coerce')
            return float(val) if np.isfinite(val) else None
    except Exception:
        pass
    return None

def make_beep_wav_base64(freq=880.0, duration_s=0.25, rate=22050):
    """Génère un beep WAV (base64) court pour les alertes; fallback si autoplay bloqué."""
    t = np.linspace(0, duration_s, int(rate*duration_s), endpoint=False)
    data = (0.3*np.sin(2*np.pi*freq*t)).astype(np.float32)
    # Convert float32 -> int16 for PCM WAV
    data_i16 = (data * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data_i16.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return b64

BEEP_B64 = make_beep_wav_base64()

# -------------------- En-tête (logo au-dessus des onglets) --------------------
show_logo()
st.write("")

# -------------------- Sidebar --------------------
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
    st.markdown("### Fiche signalétique")
    with st.form("ath_form"):
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input("Nom", key="ath_nom")
            dob = st.date_input("Date de naissance", key="ath_dob")
            poids = st.number_input("Poids (kg)", 30.0, 150.0, 70.0, key="ath_poids")
            club = st.text_input("Club", key="ath_club")
        with col2:
            prenom = st.text_input("Prénom", key="ath_prenom")
            sexe = st.selectbox("Sexe", ["Homme", "Femme"], key="ath_sexe")
            taille = st.number_input("Taille (cm)", 100.0, 220.0, 175.0, key="ath_taille")
            email = st.text_input("Email", key="ath_email")
        tel = st.text_input("Téléphone", key="ath_tel")
        sub = st.form_submit_button("Enregistrer")
        if sub:
            # Sauvegarde CSV/Excel (append)
            row = pd.DataFrame([
                {
                    "Nom": nom, "Prénom": prenom, "DateNaissance": pd.to_datetime(dob),
                    "Sexe": sexe, "Poids(kg)": poids, "Taille(cm)": taille,
                    "Club": club, "Email": email, "Téléphone": tel,
                }
            ])
            # CSV
            csv_path = "athletes.csv"
            if os.path.exists(csv_path):
                row.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                row.to_csv(csv_path, index=False)
            # Excel
            xlsx_path = "athletes.xlsx"
            if os.path.exists(xlsx_path):
                try:
                    df_old = pd.read_excel(xlsx_path, engine='openpyxl')
                    df_new = pd.concat([df_old, row], ignore_index=True)
                    df_new.to_excel(xlsx_path, index=False, engine='openpyxl')
                except Exception as e:
                    st.warning(f"Échec mise à jour Excel: {e}")
            else:
                try:
                    row.to_excel(xlsx_path, index=False, engine='openpyxl')
                except Exception as e:
                    st.warning(f"Échec création Excel: {e}")
            st.success("Fiche enregistrée → athletes.csv / athletes.xlsx")

# -------------------- Analyse Lactate --------------------
with lactate_tab:
    st.markdown("### Outil Analyse Lactate")
    st.components.v1.html('<iframe src="https://www.exphyslab.com/lactate" style="width:100%;height:800px;border:none;"></iframe>', height=820)
    st.info('Privilégier **Bsln+0.5** pour SL1 et **modDmax** pour SL2')

# -------------------- MLSS --------------------
with mlss_tab:
    st.markdown("### MLSS – Palier 30 min")

    # ---- Timer 30:00 avec alertes 25/20/15/10/5/0 + beep ----
    if 'mlss_running' not in st.session_state:
        st.session_state.mlss_running = False
    if 'mlss_start_time' not in st.session_state:
        st.session_state.mlss_start_time = None
    if 'mlss_elapsed' not in st.session_state:
        st.session_state.mlss_elapsed = 0.0
    if 'mlss_duration' not in st.session_state:
        st.session_state.mlss_duration = 30 * 60  # sec
    if 'mlss_marks_fired' not in st.session_state:
        st.session_state.mlss_marks_fired = {1500: False, 1200: False, 900: False, 600: False, 300: False, 0: False}

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("▶️ Démarrer"):
            if not st.session_state.mlss_running:
                st.session_state.mlss_running = True
                st.session_state.mlss_start_time = time.time() - st.session_state.mlss_elapsed
    with colB:
        if st.button("⏹️ Arrêter"):
            if st.session_state.mlss_running:
                st.session_state.mlss_running = False
                st.session_state.mlss_elapsed = time.time() - st.session_state.mlss_start_time
    with colC:
        if st.button("🔄 Reset"):
            st.session_state.mlss_running = False
            st.session_state.mlss_start_time = None
            st.session_state.mlss_elapsed = 0.0
            st.session_state.mlss_marks_fired = {1500: False, 1200: False, 900: False, 600: False, 300: False, 0: False}

    # Mise à jour du temps écoulé
    if st.session_state.mlss_running and st.session_state.mlss_start_time is not None:
        st.session_state.mlss_elapsed = time.time() - st.session_state.mlss_start_time

    remaining = max(0, int(st.session_state.mlss_duration - st.session_state.mlss_elapsed))
    m, s = divmod(remaining, 60)
    st.metric("⏱ Temps restant", f"{m:02d}:{s:02d}")

    # Alertes visuelles/sonores aux jalons
    alert_placeholder = st.empty()
    if remaining in st.session_state.mlss_marks_fired and not st.session_state.mlss_marks_fired[remaining]:
        st.session_state.mlss_marks_fired[remaining] = True
        mm = remaining // 60
        if remaining == 0:
            alert_placeholder.warning("⏱️ 00:00 — Fin de palier, point lactate !")
        else:
            alert_placeholder.warning(f"⏱️ {mm:02d}:00 — Point lactate !")
        # beep (peut être bloqué par le navigateur)
        st.components.v1.html(f"""
            <audio autoplay>
              <source src="data:audio/wav;base64,{BEEP_B64}" type="audio/wav">
            </audio>
        """, height=0)

    # Rafraîchissement pendant l'exécution
    if st.session_state.mlss_running and remaining > 0:
        st.experimental_rerun()

    st.markdown("---")

    # ---- Tableau MLSS ----
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame({
            "Temps (min)": [5,10,15,20,25,30],
            "Lactate (mmol/L)": [np.nan]*6,
            "FC (bpm)": [np.nan]*6
        })

    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(
            st.session_state["df_mlss"], num_rows="fixed", hide_index=True,
            column_config={
                "Temps (min)": st.column_config.Column(disabled=True)
            }
        )
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit

    df = st.session_state["df_mlss"].copy()
    t = pd.to_numeric(df["Temps (min)"], errors="coerce").to_numpy()
    lac = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce").to_numpy()
    hr = pd.to_numeric(df["FC (bpm)"], errors="coerce").to_numpy()

    slope, delta_10_30 = None, None
    if np.isfinite(lac).sum() >= 2:
        try:
            slope = np.polyfit(t, lac, 1)[0]
        except Exception:
            slope = None
    # Δ10→30 si valeurs présentes
    l10 = df_get_value(df, "Temps (min)", "Lactate (mmol/L)", 10)
    l30 = df_get_value(df, "Temps (min)", "Lactate (mmol/L)", 30)
    if l10 is not None and l30 is not None:
        delta_10_30 = l30 - l10

    # Graphique
    if np.isfinite(lac).sum() >= 2:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(t, lac, 'o-', color='#005a9e', linewidth=2.5, label='Lactate (mmol/L)')
        ax.axhline(bsn, color='#767676', linestyle='--', linewidth=1, label=f'Bsn ≈ {bsn:.1f} mmol/L')
        ax.set_xlabel('Temps (min)')
        ax.set_ylabel('Lactate (mmol/L)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if np.isfinite(hr).sum() >= 2:
            ax2 = ax.twinx()
            ax2.plot(t, hr, 's-', color='#d83b01', linewidth=2.0, label='FC (bpm)')
            ax2.set_ylabel('FC (bpm)')
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(handles + h2, labels + l2, loc='upper left')
        else:
            ax.legend(loc='upper left')
        st.pyplot(fig)
        if slope is not None:
            st.caption(f"Pente lactate: {slope:.3f} mmol·L⁻¹·min⁻¹")

    # Suggestion vitesse MLSS (auto)
    if slope is not None and delta_10_30 is not None:
        step = suggestion_step(slope, delta_10_30)
        rationale = ""
        if slope > 0.02:
            rationale = f"Lactate en hausse → baisse ~{step:.1f} km/h"
        elif slope < -0.02:
            rationale = f"Lactate en baisse → augmente ~{step:.1f} km/h"
        else:
            rationale = "Pente quasi nulle : affiner ±0,1–0,2 km/h"
        st.info(f"**Suggestion MLSS** : ajuster la vitesse de **±{step:.1f} km/h**. {rationale} (Δ10→30={delta_10_30:.2f})")

# -------------------- SRS --------------------
with srs_tab:
    st.markdown("### Paramétrage SRS")
    slope_r = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1, key="slope_r")
    sv1 = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="sv1")
    sv2_srs = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="sv2_srs")
    delta_step2 = st.number_input("Delta Step2 (km/h)", -5.0, 5.0, -0.8, step=0.1, key="delta_step2")
    step1 = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1, key="step1")
    v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1, key="v_equiv1")
    v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1, key="v_equiv2")

    # Validation des champs SRS
    errors = []
    if slope_r == 0:
        errors.append("La pente de rampe (slope_r) doit être > 0 pour estimer le MRT.")
    if (step1 > 0 and v_equiv1 == 0) or (v_equiv1 > 0 and step1 == 0):
        errors.append("Renseignez à la fois Step 1 et Vitesse équivalente 1, ou laissez les deux à 0.")
    if (sv2_srs > 0 and v_equiv2 == 0) or (v_equiv2 > 0 and sv2_srs == 0):
        errors.append("Renseignez à la fois SV2 (Step 2) et Vitesse équivalente 2, ou laissez les deux à 0.")

    for e in errors:
        st.error(e)

    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 else "—")

    # --- Calcul MRT et seuils corrigés (uniquement si pas d'erreur) ---
    if len(errors) == 0 and slope_r > 0:
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
            mrt_min = float(np.clip(mrt_min, 0.167, 1.0))  # bornage 10s–60s pour robustesse
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
