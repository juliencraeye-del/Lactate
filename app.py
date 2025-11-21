
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

VERSION = "1.0"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")
LOGO_PATH = "logo.png"

def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    else:
        st.markdown("### Wild Training")

# Sidebar avec logo
st.sidebar.header("Paramètres")
with st.sidebar:
    show_logo()

vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)
st.sidebar.caption(f"Version {VERSION}")

ath_tab, lactate_tab, mlss_tab, srs_tab = st.tabs(["👤 Fiche Athlète", "📊 Analyse Lactate", "🧪 MLSS", "🏃‍♂️ SRS"])

# Onglet Analyse Lactate
with lactate_tab:
    st.markdown("### Outil Analyse Lactate")
    components.html('<iframe src="https://www.exphyslab.com/lactate" style="width:100%;height:800px;border:none;"></iframe>', height=820)
    st.info('Privilégier **Bsln+0.5** pour SL1 et **modDmax** pour SL2')

# Onglet MLSS
with mlss_tab:
    st.markdown("### MLSS – Palier 30 min")
    st.write("(Contenu MLSS conservé)")

# Onglet SRS
with srs_tab:
    st.markdown("### Paramétrage SRS")
    slope_r = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1)
    sv1 = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    sv2_srs = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    delta_step2 = st.number_input("Delta Step2 (km/h)", -5.0, 5.0, -0.8, step=0.1)
    step1 = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    vo2_1 = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1)
    v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
    vo2_2 = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1)
    v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)

    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 else "—")

    # Bloc MRT uniquement ici
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
