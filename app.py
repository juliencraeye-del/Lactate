# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.5.5)
import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.5.5"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ---------------- Helpers ----------------
def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    for col in list(df.columns):
        if str(col).strip().lower().startswith("fc estim"):
            df.drop(columns=[col], inplace=True, errors="ignore")
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    preferred = ["Palier","%VMA","Vitesse (km/h)","Allure (min/km)","Allure (mm:ss/km)","FC mesurée (bpm)","Lactate (mmol/L)"]
    df = ensure_columns(df, preferred)
    return df

def sanitize_mlss(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = ensure_columns(df, ["Temps (min)","Lactate (mmol/L)","FC (bpm)","Commentaires"])
    df["Temps (min)"] = pd.to_numeric(df["Temps (min)"], errors="coerce")
    df["Lactate (mmol/L)"] = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce")
    df["FC (bpm)"] = pd.to_numeric(df["FC (bpm)"], errors="coerce")
    return df.sort_values("Temps (min)").reset_index(drop=True)

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------- Sidebar ----------------
st.sidebar.header("Paramètres du test")
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", 6, 20, 10, step=1)
pct_start = st.sidebar.number_input("%VMA départ", 40.0, 80.0, 60.0, step=1.0)
pct_end = st.sidebar.number_input("%VMA final", 80.0, 120.0, 105.0, step=1.0)
use_poly = st.sidebar.toggle("Lisser la courbe (D‑max polynomial)", value=False)
poly_order = st.sidebar.select_slider("Ordre polynôme", options=[2,3], value=2, disabled=not use_poly)

if st.sidebar.button("🔄 Réinitialiser"):
    for k in ["grid_df_data","df_mlss_lac","mlss_params","mlss_img_b64","srs_results","srs_img_b64","athlete","date","note","historique"]:
        st.session_state.pop(k, None)
    st.rerun()

# ---------------- Base DF ----------------
pcts = np.linspace(pct_start, pct_end, int(n))
speeds = vma * pcts / 100.0
base_df = pd.DataFrame({
    "Palier": np.arange(1,int(n)+1),
    "%VMA": np.round(pcts,2),
    "Vitesse (km/h)": np.round(speeds,2),
    "Allure (min/km)": np.round(60/speeds,2),
    "Allure (mm:ss/km)": [f"{int(60/s):02d}:{int((60/s-int(60/s))*60):02d}" for s in speeds],
    "FC mesurée (bpm)": [None]*int(n),
    "Lactate (mmol/L)": [None]*int(n)
})
base_df = sanitize_df(base_df)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Saisie","📈 Résultats","🧪 MLSS","🏃‍♂️ SRS","🗂️ Historique"])

# ---------------- Tab1: Saisie ----------------
with tab1:
    if "grid_df_data" not in st.session_state:
        st.session_state["grid_df_data"] = base_df.copy()
    else:
        st.session_state["grid_df_data"] = sanitize_df(st.session_state["grid_df_data"])

    df_edit = st.data_editor(st.session_state["grid_df_data"], key="grid_editor", num_rows="fixed", width="stretch", hide_index=True)
    st.session_state["grid_df_data"] = sanitize_df(df_edit)

    athlete = st.text_input("Athlète", value=st.session_state.get("athlete","Anonyme"))
    date_s = st.date_input("Date").isoformat()
    note = st.text_input("Notes", value=st.session_state.get("note",""))
    st.session_state.update({"athlete":athlete,"date":date_s,"note":note})

    out_df = sanitize_df(st.session_state["grid_df_data"])
    out_df = ensure_columns(out_df, ["Lactate (mmol/L)"])
    out_df["log10(lactate)"] = np.where(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")>0,
                                        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")),np.nan)
    buf = io.StringIO(); out_df.to_csv(buf,index=False)
    st.download_button("💾 Télécharger CSV", data=buf.getvalue(), file_name=f"seance_{athlete}_{date_s}.csv", mime="text/csv")

# ---------------- Tab2: Résultats ----------------
with tab2:
    df_calc = sanitize_df(st.session_state["grid_df_data"])
    df_calc = ensure_columns(df_calc, ["Vitesse (km/h)","Lactate (mmol/L)"])
    x = pd.to_numeric(df_calc["Vitesse (km/h)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_calc["Lactate (mmol/L)"], errors="coerce").to_numpy()

    # Graphiques
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(x,y,"o-",color="#1f77b4"); ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Lactate (mmol/L)")
    ax1.grid(True,alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7,4))
    mask = y>0
    if mask.sum()>1:
        ax2.plot(x[mask],np.log10(y[mask]),"o-",color="#1f77b4")
    ax2.set_xlabel("Vitesse (km/h)"); ax2.set_ylabel("log10(lactate)")
    ax2.grid(True,alpha=0.3)
    st.pyplot(fig2)

    img1_b64 = fig_to_base64(fig1); img2_b64 = fig_to_base64(fig2)

    # Export HTML
    html = f"""
    <html><head><meta charset="utf-8"><style>
    body {{ font-family: Arial; margin:24px; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid #ccc; padding:6px; }}
    </style></head><body>
    <h1>Rapport – Test lactate</h1>
    <p><b>Athlète:</b> {athlete} | <b>Date:</b> {date_s}</p>
    <h2>Données</h2>{df_calc.to_html(index=False)}
    <h2>Graphiques</h2>
    data:image/png;base64,{img1_b64}<br/>
    data:image/png;base64,{img2_b64}
    </body></html>
    """
    st.download_button("🧾 Télécharger rapport HTML", data=html.encode("utf-8"), file_name=f"rapport_{athlete}.html", mime="text/html")

# ---------------- Tab3: MLSS ----------------
with tab3:
    sv2_mlss = st.number_input("Vitesse SV2 (km/h)",0.0,30.0,0.0,step=0.1)
    delta_mlss = st.number_input("Delta vs SV2 (km/h)",-5.0,5.0,-0.6,step=0.1)
    v_theo_mlss = sv2_mlss+delta_mlss if sv2_mlss>0 else None
    st.metric("Vitesse théorique MLSS", f"{v_theo_mlss:.1f}" if v_theo_mlss else "—")

    if "df_mlss_lac" not in st.session_state:
        st.session_state.df_mlss_lac = pd.DataFrame({"Temps (min)":[0,5,10,15,20,25,30],"Lactate (mmol/L)":[None]*7,"FC (bpm)":[None]*7,"Commentaires":[""]*7})
    df_mlss = st.data_editor(st.session_state.df_mlss_lac,width="stretch",hide_index=True)
    st.session_state.df_mlss_lac = sanitize_mlss(df_mlss)

    plot_df = sanitize_mlss(st.session_state.df_mlss_lac)
    mlss_img_b64=None
    if plot_df["Lactate (mmol/L)"].notna().sum()>=2:
        fig_mlss,ax_mlss=plt.subplots(figsize=(6,3.5))
        ax_mlss.plot(plot_df["Temps (min)"],plot_df["Lactate (mmol/L)"],"o-",color="#0078d4")
        ax_mlss.set_xlabel("Temps (min)"); ax_mlss.set_ylabel("Lactate (mmol/L)")
        ax_mlss.grid(True,alpha=0.3); st.pyplot(fig_mlss)
        mlss_img_b64=fig_to_base64(fig_mlss)

# ---------------- Tab4: SRS ----------------
with tab4:
    st.write("Paramétrage Step–Ramp–Step (SRS)")
    slope=st.number_input("Pente rampe (km/h/min)",0.0,10.0,0.0,step=0.1)
    sv1=st.number_input("SV1 (km/h)",0.0,30.0,0.0,step=0.1)
    sv2=st.number_input("SV2 (km/h)",0.0,30.0,0.0,step=0.1)
    delta_step2=st.number_input("Delta Step2 (km/h)",-5.0,5.0,-0.8,step=0.1)
    step2=sv2+delta_step2 if sv2>0 else None
    st.metric("Vitesse Step 2",f"{step2:.1f}" if step2 else "—")
    # Calcul MRT
    # (ajoute logique si besoin)