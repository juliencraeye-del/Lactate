# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.8.2)
# - Logo affiché sur chaque onglet (si présent)
# - Beep réel (si présent) pour alertes MLSS
# - Onglets : Fiche Athlète, Analyse Lactate (iframe), MLSS avec timer + alertes sonores, SRS

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.8.2"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

LOGO_PATH = "logo.png"   # Placez votre logo ici
BEEP_PATH = "beep.wav"   # Placez votre son ici

# Sidebar avec fallback
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=120)
else:
    st.sidebar.markdown("**Wild Training**")

st.sidebar.header("Paramètres")
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)
st.sidebar.caption(f"Version {VERSION}")

# Tabs
ath_tab, lactate_tab, mlss_tab, srs_tab = st.tabs([
    "👤 Fiche Athlète", "📊 Analyse Lactate", "🧪 MLSS", "🏃‍♂️ SRS"
])

# Fonction pour afficher le logo avec fallback
def show_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("### Wild Training")

# Onglet Fiche Athlète
with ath_tab:
    show_logo()
    st.markdown("### Fiche signalétique")
    with st.form("ath_form"):
        nom = st.text_input("Nom", st.session_state.get("nom", ""))
        prenom = st.text_input("Prénom", st.session_state.get("prenom", ""))
        dob = st.date_input("Date de naissance")
        sexe = st.selectbox("Sexe", ["Homme", "Femme"], index=0)
        poids = st.number_input("Poids (kg)", 30.0, 150.0, st.session_state.get("poids", 70.0))
        taille = st.number_input("Taille (cm)", 100.0, 220.0, st.session_state.get("taille", 175.0))
        club = st.text_input("Club", st.session_state.get("club", ""))
        email = st.text_input("Email", st.session_state.get("email", ""))
        tel = st.text_input("Téléphone", st.session_state.get("tel", ""))
        sub = st.form_submit_button("Enregistrer")
        if sub:
            st.session_state.update({
                "nom": nom, "prenom": prenom, "dob": str(dob), "sexe": sexe,
                "poids": poids, "taille": taille, "club": club, "email": email, "tel": tel
            })
            st.success("Fiche enregistrée")

# Onglet Analyse Lactate
with lactate_tab:
    show_logo()
    st.markdown("### Outil Analyse Lactate")
    st.markdown('https://www.exphyslab.com/lactate', unsafe_allow_html=True)

# Onglet MLSS
with mlss_tab:
    show_logo()
    st.markdown("### MLSS – Palier 30 min")
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = time.time()
    elapsed = int(time.time() - st.session_state["start_time"])
    remaining = max(0, 1800 - elapsed)
    mins, secs = divmod(remaining, 60)
    st.markdown(f"⏱ Temps restant : **{mins:02d}:{secs:02d}**")

    # Alerte toutes les 5 min
    if remaining > 0 and remaining % 300 < 2:
        st.warning("Alerte : point lactate !")
        if os.path.exists(BEEP_PATH):
            st.audio(BEEP_PATH)

    # Tableau MLSS
    times = [5, 10, 15, 20, 25, 30]
    if "df_mlss" not in st.session_state:
        st.session_state["df_mlss"] = pd.DataFrame({
            "Temps (min)": times, "Lactate (mmol/L)": np.nan, "FC (bpm)": np.nan
        })

    with st.form("mlss_form"):
        df_mlss_edit = st.data_editor(st.session_state["df_mlss"], num_rows="fixed", hide_index=True)
        sub_mlss = st.form_submit_button("Analyser MLSS")
        if sub_mlss:
            st.session_state["df_mlss"] = df_mlss_edit
            t = pd.to_numeric(df_mlss_edit["Temps (min)"], errors="coerce").to_numpy()
            lac = pd.to_numeric(df_mlss_edit["Lactate (mmol/L)"], errors="coerce").to_numpy()
            valid = np.isfinite(t) & np.isfinite(lac)
            if np.sum(valid) >= 2:
                slope = np.polyfit(t[valid], lac[valid], 1)[0]
                delta = lac[-1] - lac[1] if np.isfinite(lac[-1]) and np.isfinite(lac[1]) else None
                stable = (abs(slope) <= 0.02) and (delta is not None and abs(delta) <= 0.5)
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                color_bg = "#d4edda" if stable else "#f8d7da"
                ax2.set_facecolor(color_bg)
                ax2.plot(t, lac, "o-", color="blue")
                ax2.set_xlabel("Temps (min)"); ax2.set_ylabel("Lactate (mmol/L)")
                ax2.grid(True)
                st.pyplot(fig2)
                if stable:
                    st.success("Lactate stable → compatible MLSS")
                else:
                    st.error("Lactate instable → ajuster vitesse")
                    v_target = st.session_state.get("v_target", vma * 0.85)
                    step = 0.3 if slope > 0 else -0.3
                    suggestion = max(5.0, min(30.0, v_target + step))
                    st.info(f"Proposition vitesse ajustée : {suggestion:.1f} km/h")

# Onglet SRS
with srs_tab:
    show_logo()
    st.markdown("### SRS – à compléter")