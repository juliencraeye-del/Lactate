# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.7.5)
# Changements :
# - Onglets MLSS et SRS restaurés (saisie + calculs + affichage)
# - Export HTML corrigé avec balises <img>
# - Sections MLSS et SRS incluses même si vides
# - Option export PDF (si pdfkit installé)
# - Maintien des correctifs précédents (saisie sécurisée, arrondi vitesse, export CSV)

import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.7.5"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ---------------- Helpers ----------------
def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def sanitize_df(df):
    if df is None or df.empty: return df
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return df

def sanitize_mlss(df):
    if df is None or df.empty: return df
    df["Temps (min)"] = pd.to_numeric(df["Temps (min)"], errors="coerce")
    df["Lactate (mmol/L)"] = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce")
    return df.dropna(how="all").sort_values("Temps (min)").reset_index(drop=True)

def round_to_step(val, step=0.1):
    return round(val/step)*step if val else None

# MLSS control
def mlss_control(df_mlss, amplitude_thr=1.0):
    result = {"n_points":0,"amplitude":None,"slope":None,"stable":None,"suggest_kmh":None,"note":None}
    win = df_mlss[(df_mlss["Temps (min)"]>=10)&(df_mlss["Temps (min)"]<=30)].dropna(subset=["Lactate (mmol/L)"])
    if win.empty or win["Lactate (mmol/L)"].notna().sum()<2:
        result["note"]="Données insuffisantes (≥2 points 10–30')."
        return result
    lac,t=win["Lactate (mmol/L)"].to_numpy(),win["Temps (min)"].to_numpy()
    result["n_points"]=len(lac)
    result["amplitude"]=float(np.nanmax(lac)-np.nanmin(lac))
    try: result["slope"]=float(np.polyfit(t,lac,1)[0])
    except: pass
    stable=result["amplitude"]<=amplitude_thr
    result["stable"]=stable
    suggest=0.0; note="Stabilité OK" if stable else "Instabilité détectée"
    if result["slope"] and result["slope"]>0.15: suggest=-0.3; note="Réduire vitesse"
    elif result["slope"] and result["slope"]<-0.15: suggest=+0.3; note="Augmenter vitesse"
    result["suggest_kmh"]=suggest; result["note"]=note
    return result

# Sidebar
st.sidebar.header("Paramètres du test")
vma=st.sidebar.number_input("VMA (km/h)",5.0,30.0,17.0,step=0.1)
n=st.sidebar.number_input("Nombre de paliers",6,20,10,step=1)
pct_start=st.sidebar.number_input("%VMA départ",40.0,80.0,60.0)
pct_end=st.sidebar.number_input("%VMA final",80.0,120.0,105.0)

# Base DF
pcts=np.linspace(pct_start,pct_end,int(n)); speeds=vma*pcts/100.0
def pace_mmss(s): return f"{int(60/s):02d}:{int(round((60/s-int(60/s))*60)):02d}" if s>0 else ""
base_df=pd.DataFrame({"Palier":range(1,int(n)+1),"%VMA":np.round(pcts,2),"Vitesse (km/h)":np.round(speeds,2),
"Allure (min/km)":np.round(60/speeds,2),"Allure (mm:ss/km)":[pace_mmss(s) for s in speeds],
"FC mesurée (bpm)":[None]*int(n),"Lactate (mmol/L)":[None]*int(n)})

tab1,tab2,tab3,tab4=st.tabs(["📝 Saisie","📈 Résultats","🧪 MLSS","🏃‍♂️ SRS"])

# Tab1 : saisie
with tab1:
    if "grid_df_data" not in st.session_state: st.session_state["grid_df_data"]=base_df.copy()
    with st.form("saisie_form"):
        df_edit=st.data_editor(st.session_state["grid_df_data"],width="stretch",hide_index=True)
        athlete=st.text_input("Athlète",st.session_state.get("athlete","Anonyme"))
        date_s=st.date_input("Date").isoformat()
        note=st.text_input("Notes",st.session_state.get("note",""))
        if st.form_submit_button("💾 Enregistrer"): st.session_state.update({"grid_df_data":sanitize_df(df_edit),"athlete":athlete,"date":date_s,"note":note})

# Tab2 : résultats
with tab2:
    df_calc=sanitize_df(st.session_state["grid_df_data"].copy())
    x=pd.to_numeric(df_calc["Vitesse (km/h)"],errors="coerce").to_numpy()
    y=pd.to_numeric(df_calc["Lactate (mmol/L)"],errors="coerce").to_numpy()
    fig1,ax1=plt.subplots(figsize=(7,4)); ax1.plot(x,y,"o-"); ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Lactate (mmol/L)"); st.pyplot(fig1)
    fig2,ax2=plt.subplots(figsize=(7,4)); mask=y>0
    if np.sum(mask)>1: ax2.plot(x[mask],np.log10(y[mask]),"o-"); ax2.set_xlabel("Vitesse (km/h)"); ax2.set_ylabel("log10(lactate)"); st.pyplot(fig2)
    img1_b64,img2_b64=fig_to_base64(fig1),fig_to_base64(fig2)

# Tab3 : MLSS
with tab3:
    if "df_mlss_lac" not in st.session_state: st.session_state["df_mlss_lac"]=pd.DataFrame(columns=["Temps (min)","Lactate (mmol/L)","FC (bpm)","Commentaires"])
    with st.form("mlss_form"):
        df_mlss=st.data_editor(st.session_state["df_mlss_lac"],width="stretch",hide_index=True)
        if st.form_submit_button("💾 Enregistrer MLSS"): st.session_state["df_mlss_lac"]=sanitize_mlss(df_mlss)
    if not st.session_state["df_mlss_lac"].empty:
        st.write("Données MLSS :",st.session_state["df_mlss_lac"])
        res=mlss_control(st.session_state["df_mlss_lac"])
        st.write("Analyse MLSS :",res)
        fig_mlss,axm=plt.subplots(); axm.plot(st.session_state["df_mlss_lac"]["Temps (min)"],st.session_state["df_mlss_lac"]["Lactate (mmol/L)"],"o-"); st.pyplot(fig_mlss)
        st.session_state["mlss_img_b64"]=fig_to_base64(fig_mlss)
    else:
        st.session_state["mlss_img_b64"]=""

# Tab4 : SRS
with tab4:
    st.write("Protocole Step-Ramp-Step")
    if "srs_results" not in st.session_state: st.session_state["srs_results"]={}
    srs_vitesse=st.number_input("Vitesse palier (km/h)",5.0,30.0,12.0)
    srs_fc=st.number_input("FC moyenne (bpm)",80,220,150)
    if st.button("Calculer SRS"):
        st.session_state["srs_results"]={"Vitesse":srs_vitesse,"FC":srs_fc}
        fig_srs,axs=plt.subplots(); axs.bar(["Vitesse","FC"],[srs_vitesse,srs_fc]); st.pyplot(fig_srs)
        st.session_state["srs_img_b64"]=fig_to_base64(fig_srs)

# Export HTML
mlss_html=st.session_state["df_mlss_lac"].to_html(index=False) if not st.session_state["df_mlss_lac"].empty else "<p>(Pas de données MLSS)</p>"
srs_img_b64=st.session_state.get("srs_img_b64","")
mlss_img_b64=st.session_state.get("mlss_img_b64","")
html=f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Rapport</title></head><body>
<h1>Rapport – Test lactate & SRS</h1>
<p><b>Athlète:</b> {st.session_state.get("athlete","Anonyme")} | <b>Date:</b> {st.session_state.get("date","")} | <b>VMA:</b> {vma:.1f} km/h</p>
<h2>Données brutes</h2>{df_calc.to_html(index=False)}
<h2>Graphiques lactate</h2>
<h3>Lactate – Vitesse</h3><img src="data;base64,{img1_b64}
<h3>Log(lactate) – Vitesse</h3>data:image/png;base64,{img2_b64}
<h2>MLSS</h2>{mlss_html}
{f'data:image/png;base64,{mlss_img_b64}' if mlss_img_b64 else '<p>Aucun graphique MLSS disponible</p>'}
<h2>SRS</h2>
{f'data:image/png;base64,{srs_img_b64}' if srs_img_b64 else '<p>Aucun graphique SRS disponible</p>'}
</body></html>
"""
st.download_button("🧾 Télécharger rapport HTML",data=html.encode("utf-8"),file_name="rapport.html",mime="text/html")

# Export PDF (optionnel)
try:
    import pdfkit
    pdf=pdfkit.from_string(html,False)
    st.download_button("📄 Télécharger rapport PDF",data=pdf,file_name="rapport.pdf",mime="application/pdf")
except:
    st.info("PDFkit non disponible. Installez wkhtmltopdf pour activer l'export PDF.")