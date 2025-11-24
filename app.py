
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

VERSION = "2.0"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")
LOGO_PATH = "logo.png"

def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("### Wild Training")

def suggestion_step(slope, delta):
    base = abs(slope) * 10
    if delta is not None:
        if abs(delta) > 0.5:
            base += 0.2
        if abs(delta) > 1.0:
            base += 0.3
    return float(np.clip(base, 0.2, 0.8))

def mlss_stability_metrics(t, lac):
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

# En-tête global
col_header = st.columns([0.12, 0.88])
with col_header[0]:
    show_logo()
with col_header[1]:
    st.markdown("## Seuils Lactate – VMA")

# Sidebar
st.sidebar.header("Paramètres")
with st.sidebar:
    show_logo()
    st.caption(f"Version {VERSION}")
    vma = st.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
    bsn = st.number_input("Lactate Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)

ath_tab, lactate_tab, mlss_tab, srs_tab = st.tabs(["👤 Fiche Athlète", "📊 Analyse Lactate", "🧪 MLSS", "🏃‍♂️ SRS"])

with ath_tab:
    st.markdown("### Fiche signalétique")
    with st.form("ath_form"):
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        dob = st.date_input("Date de naissance")
        sexe = st.selectbox("Sexe", ["Homme", "Femme"])
        poids = st.number_input("Poids (kg)", 30.0, 150.0, 70.0)
        taille = st.number_input("Taille (cm)", 100.0, 220.0, 175.0)
        club = st.text_input("Club")
        email = st.text_input("Email")
        tel = st.text_input("Téléphone")
        sub = st.form_submit_button("Enregistrer")
        if sub:
            st.success("Fiche enregistrée")

with lactate_tab:
    st.markdown("### Outil Analyse Lactate")
    components.html('<iframe src="https://www.exphyslab.com/lactate" style="width:100%;height:800px;border:none;"></iframe>', height=820)
    st.info('Privilégier **Bsln+0.5** pour SL1 et **modDmax** pour SL2')

with mlss_tab:
    st.markdown("### MLSS – Palier 30 min")
    st.markdown("#### ⏱ Timer 30 min")
    # Graphique MLSS
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame({"Temps (min)": [5,10,15,20,25,30], "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan})
    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(st.session_state["df_mlss"], num_rows="fixed", hide_index=True)
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit
    df = st.session_state["df_mlss"].copy()
    t = pd.to_numeric(df["Temps (min)"], errors="coerce").to_numpy()
    lac = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce").to_numpy()
    hr = pd.to_numeric(df["FC (bpm)"], errors="coerce").to_numpy()
    if np.isfinite(lac).sum() >= 2:
        slope = np.polyfit(t, lac, 1)[0]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(t, lac, 'o-', color='blue', label='Lactate')
        ax.axhline(bsn, color='gray', linestyle='--', label=f'Bsn {bsn:.1f}')
        ax.set_xlabel('Temps (min)')
        ax.set_ylabel('Lactate (mmol/L)')
        if np.isfinite(hr).sum() >= 2:
            ax2 = ax.twinx()
            ax2.plot(t, hr, 's-', color='red', label='FC')
            ax2.set_ylabel('FC (bpm)')
        st.pyplot(fig)
        if st.button("Exporter le graphique MLSS en PNG"):
            fig.savefig("mlss_graph.png")
            with open("mlss_graph.png", "rb") as img_file:
                st.download_button("Télécharger MLSS.png", img_file, "MLSS.png", "image/png")

with srs_tab:
    st.markdown("### Paramétrage SRS")
    slope_r = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1)
    sv1 = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    sv2_srs = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    delta_step2 = st.number_input("Delta Step2 (km/h)", -5.0, 5.0, -0.8, step=0.1, key="delta_step2")
    step1 = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
    v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 else "—")
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
    if slope_r > 0:
        mrt_for_calc = mrt_min if 'mrt_min' in locals() and mrt_min is not None else 0.5
        delta_suggestion = round(np.clip(-slope_r * mrt_for_calc, -1.2, -0.2), 1)
        st.caption(f"Suggestion Delta Step2 auto ≈ {delta_suggestion:+.1f} km/h")
        if st.button("Appliquer la suggestion"):
            st.session_state["delta_step2"] = delta_suggestion
            st.rerun()
    if st.button("Exporter le rapport complet en PDF"):
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate("rapport_complet.pdf", pagesize=A4)
        elements = []
        if os.path.exists(LOGO_PATH):
            elements.append(Image(LOGO_PATH, width=6*cm, height=6*cm))
        elements.append(Paragraph("Rapport complet MLSS + SRS + Fiche Athlète", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Fiche Athlète", styles['Heading2']))
        elements.append(Paragraph(f"Nom: {nom}<br/>Prénom: {prenom}<br/>Date de naissance: {dob}<br/>Sexe: {sexe}<br/>Poids: {poids} kg<br/>Taille: {taille} cm<br/>Club: {club}", styles['Normal']))
        elements.append(Spacer(1, 12))
        fig.savefig("mlss_graph.png")
        elements.append(Paragraph("Graphique MLSS", styles['Heading2']))
        elements.append(Image("mlss_graph.png", width=14*cm, height=8*cm))
        elements.append(Spacer(1, 12))
        rapport_text = f"Seuils initiaux : SV1 = {sv1:.2f} km/h, SV2 = {sv2_srs:.2f} km/h<br/>MRT estimé : {mrt_sec:.1f} s<br/>Seuils corrigés : SV1 = {sv1_corr:.2f} km/h, SV2 = {sv2_corr:.2f} km/h<br/>Pourquoi ces corrections ? Le MRT reflète le retard de la VO₂ lors des paliers. Les seuils corrigés tiennent compte de ce décalage cinétique, offrant une estimation plus réaliste des intensités physiologiques." if len(mrt_vals_min) > 0 else "Aucune donnée suffisante pour générer un rapport."
        elements.append(Paragraph("Rapport SRS", styles['Heading2']))
        elements.append(Paragraph(rapport_text, styles['Normal']))
        doc.build(elements)
        with open("rapport_complet.pdf", "rb") as f:
            st.download_button("Télécharger rapport complet PDF", f, "rapport_complet.pdf", "application/pdf")
