# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.7.3)
# Corrections:
# - Graphiques HTML: data:image/png;base64,... pour Lactate et Log-lactate
# - Pas de Markdown cassé
# - Maintien des correctifs v0.7.2 (MLSS, SRS, saisie sécurisée)

import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.7.3"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ---------------- Helpers ----------------
def ensure_columns(df, columns):
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df

def sanitize_df(df):
    if df is None or df.empty: return df
    for col in list(df.columns):
        if str(col).lower().startswith("fc estim"):
            df.drop(columns=[col], inplace=True, errors="ignore")
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    preferred = ["Palier","%VMA","Vitesse (km/h)","Allure (min/km)","Allure (mm:ss/km)","FC mesurée (bpm)","Lactate (mmol/L)"]
    df = ensure_columns(df, preferred)
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]

def sanitize_mlss(df):
    if df is None or df.empty: return df
    df = ensure_columns(df, ["Temps (min)","Lactate (mmol/L)","FC (bpm)","Commentaires"])
    df["Temps (min)"] = pd.to_numeric(df["Temps (min)"], errors="coerce")
    df["Lactate (mmol/L)"] = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce")
    df["FC (bpm)"] = pd.to_numeric(df["FC (bpm)"], errors="coerce")
    return df.dropna(how="all").sort_values("Temps (min)").reset_index(drop=True)

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def round_to_step(val, step=0.1):
    return round(val/step)*step if val else None

# MLSS control
def mlss_control(df_mlss, amplitude_thr=1.0):
    result = {"n_points":0,"amplitude":None,"slope_mmol_per_min":None,"delta_10_30":None,"stable":None,"suggest_kmh":None,"suggest_note":None}
    win = df_mlss[(df_mlss["Temps (min)"]>=10)&(df_mlss["Temps (min)"]<=30)].dropna(subset=["Lactate (mmol/L)"])
    if win.empty or win["Lactate (mmol/L)"].notna().sum()<2:
        result["n_points"]=int(win["Lactate (mmol/L)"].notna().sum())
        result["suggest_note"]="Données insuffisantes (≥2 points 10–30')."
        return result
    lac,t=win["Lactate (mmol/L)"].to_numpy(),win["Temps (min)"].to_numpy()
    result["n_points"]=len(lac)
    result["amplitude"]=float(np.nanmax(lac)-np.nanmin(lac))
    try: result["slope_mmol_per_min"]=float(np.polyfit(t,lac,1)[0])
    except: pass
    d10_30=None
    if 10 in win["Temps (min)"].values and 30 in win["Temps (min)"].values:
        d10_30=float(win[win["Temps (min)"]==30]["Lactate (mmol/L)"].iloc[-1]-win[win["Temps (min)"]==10]["Lactate (mmol/L)"].iloc[0])
    result["delta_10_30"]=d10_30
    stable=result["amplitude"]<=amplitude_thr
    result["stable"]=stable
    suggest=0.0; note="Stabilité satisfaisante." if stable else "Instabilité détectée."
    if d10_30 is not None:
        if d10_30>1.0: suggest=-0.3; note="Lactate ↑ (>+1.0): réduire ~0,3 km/h."
        elif 0.5<d10_30<=1.0: suggest=-0.2; note="Lactate ↑ (+0,5 à +1,0): réduire ~0,2 km/h."
        elif d10_30<-1.0: suggest=+0.3; note="Lactate ↓ (<−1,0): augmenter ~0,3 km/h."
        elif -1.0<d10_30<-0.5: suggest=+0.2; note="Lactate ↓ (−0,5 à −1,0): augmenter ~0,2 km/h."
    elif not stable and result["slope_mmol_per_min"] is not None:
        s=result["slope_mmol_per_min"]
        if s>0.15: suggest=-0.3; note="Pente positive: réduire ~0,3 km/h."
        elif 0.05<=s<=0.15: suggest=-0.2; note="Pente positive: réduire ~0,2 km/h."
        elif s<-0.15: suggest=+0.3; note="Pente négative: augmenter ~0,3 km/h."
        elif -0.15<=s<=-0.05: suggest=+0.2; note="Pente négative: augmenter ~0,2 km/h."
    result["suggest_kmh"]=suggest; result["suggest_note"]=note
    return result

# Sidebar
st.sidebar.header("Paramètres du test")
vma=st.sidebar.number_input("VMA (km/h)",5.0,30.0,17.0,step=0.1)
bsn=st.sidebar.number_input("Lactate de base (mmol/L)",0.5,4.0,1.5,step=0.1)
n=st.sidebar.number_input("Nombre de paliers",6,20,10,step=1)
pct_start=st.sidebar.number_input("%VMA départ",40.0,80.0,60.0)
pct_end=st.sidebar.number_input("%VMA final",80.0,120.0,105.0)

# Base DF
pcts=np.linspace(pct_start,pct_end,int(n)); speeds=vma*pcts/100.0
def pace_mmss(s): return f"{int(60/s):02d}:{int(round((60/s-int(60/s))*60)):02d}" if s>0 else ""
base_df=pd.DataFrame({"Palier":range(1,int(n)+1),"%VMA":np.round(pcts,2),"Vitesse (km/h)":np.round(speeds,2),
"Allure (min/km)":np.round(60/speeds,2),"Allure (mm:ss/km)":[pace_mmss(s) for s in speeds],
"FC mesurée (bpm)":[None]*int(n),"Lactate (mmol/L)":[None]*int(n)})
base_df=sanitize_df(base_df)

tab1,tab2,tab3,tab4=st.tabs(["📝 Saisie","📈 Résultats","🧪 MLSS","🏃‍♂️ SRS"])

# Tab1
with tab1:
    if "grid_df_data" not in st.session_state: st.session_state["grid_df_data"]=base_df.copy()
    with st.form("saisie_form"):
        df_edit=st.data_editor(st.session_state["grid_df_data"],width="stretch",hide_index=True)
        athlete=st.text_input("Athlète",st.session_state.get("athlete","Anonyme"))
        date_s=st.date_input("Date").isoformat()
        note=st.text_input("Notes",st.session_state.get("note",""))
        if st.form_submit_button("💾 Enregistrer"): st.session_state.update({"grid_df_data":sanitize_df(df_edit),"athlete":athlete,"date":date_s,"note":note})

# Tab2
with tab2:
    df_calc=sanitize_df(st.session_state["grid_df_data"].copy())
    x=pd.to_numeric(df_calc["Vitesse (km/h)"],errors="coerce").to_numpy()
    y=pd.to_numeric(df_calc["Lactate (mmol/L)"],errors="coerce").to_numpy()
    fig1,ax1=plt.subplots(figsize=(7,4)); ax1.plot(x,y,"o-"); ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Lactate (mmol/L)"); st.pyplot(fig1)
    fig2,ax2=plt.subplots(figsize=(7,4)); mask=y>0
    if np.sum(mask)>1: ax2.plot(x[mask],np.log10(y[mask]),"o-"); ax2.set_xlabel("Vitesse (km/h)"); ax2.set_ylabel("log10(lactate)"); st.pyplot(fig2)
    img1_b64, img2_b64=fig_to_base64(fig1), fig_to_base64(fig2)
    html=f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>Rapport</title></head><body>
    <h2>Graphiques lactate</h2>
    <h3>Lactate – Vitesse</h3>
    data:image/png;base64,{img1_b64}
    <h3>Log(lactate) – Vitesse</h3>
    data:image/png;base64,{img2_b64}
    </body></html>
    """
    st.download_button("🧾 Télécharger rapport HTML",data=html.encode("utf-8"),file_name="rapport.html",mime="text/html")