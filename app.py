# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.7.1 corrigée)
import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.7.1"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ---------------- Helpers ----------------
def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    for col in list(df.columns):
        if str(col).strip().lower().startswith("fc estim"):
            df.drop(columns=[col], inplace=True, errors="ignore")
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    preferred = ["Palier","%VMA","Vitesse (km/h)","Allure (min/km)","Allure (mm:ss/km)","FC mesurée (bpm)","Lactate (mmol/L)"]
    df = ensure_columns(df, preferred)
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]

def sanitize_mlss(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df = ensure_columns(df, ["Temps (min)","Lactate (mmol/L)","FC (bpm)","Commentaires"])
    df["Temps (min)"] = pd.to_numeric(df["Temps (min)"], errors="coerce")
    df["Lactate (mmol/L)"] = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce")
    df["FC (bpm)"] = pd.to_numeric(df["FC (bpm)"], errors="coerce")
    df = df.dropna(how="all")
    df = df.sort_values("Temps (min)").reset_index(drop=True)
    return df

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def round_to_step(val, step=0.1):
    if val is None: return None
    return round(val/step)*step

# ---------------- Sidebar ----------------
st.sidebar.header("Paramètres du test")
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", 6, 20, 10, step=1)
pct_start = st.sidebar.number_input("%VMA départ", 40.0, 80.0, 60.0, step=1.0)
pct_end = st.sidebar.number_input("%VMA final", 80.0, 120.0, 105.0, step=1.0)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser la séance"):
    for k in ["grid_df_data","df_mlss_lac","mlss_params","mlss_img_b64",
              "srs_results","srs_img_b64","athlete","date","note","historique"]:
        st.session_state.pop(k, None)
    st.rerun()
st.sidebar.caption(f"Version {VERSION}")

# ---------------- Base DF ----------------
pcts = np.linspace(pct_start, pct_end, int(n))
speeds = vma * pcts / 100.0
def pace_min_per_km(s): return 60/s if s>0 else np.nan
def pace_mmss(s):
    if s<=0: return ""
    p = 60/s; m = int(p); sec = int(round((p-m)*60))
    if sec==60: m+=1; sec=0
    return f"{m:02d}:{sec:02d}"
base_df = pd.DataFrame({
    "Palier": np.arange(1,int(n)+1),
    "%VMA": np.round(pcts,2),
    "Vitesse (km/h)": np.round(speeds,2),
    "Allure (min/km)": np.round([pace_min_per_km(s) for s in speeds],2),
    "Allure (mm:ss/km)": [pace_mmss(s) for s in speeds],
    "FC mesurée (bpm)": [None]*int(n),
    "Lactate (mmol/L)": [None]*int(n)
})
base_df = sanitize_df(base_df)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 Saisie", "📈 Résultats", "🧪 MLSS", "🏃‍♂️ SRS (Step–Ramp–Step)", "📂 Historique"]
)

# ---------------- Tab1: Saisie ----------------
with tab1:
    if "grid_df_data" not in st.session_state:
        st.session_state["grid_df_data"] = base_df.copy()
    else:
        st.session_state["grid_df_data"] = sanitize_df(st.session_state["grid_df_data"])
    st.markdown("### Saisie des lactates (et FC mesurée si dispo)")
    with st.form(key="saisie_form", clear_on_submit=False):
        df_edit = st.data_editor(
            st.session_state["grid_df_data"], key="grid_editor",
            num_rows="fixed", width="stretch", hide_index=True
        )
        athlete = st.text_input("Athlète", value=st.session_state.get("athlete","Anonyme"))
        date_s = st.date_input("Date").isoformat()
        note = st.text_input("Notes", value=st.session_state.get("note",""))
        submitted = st.form_submit_button("💾 Enregistrer la saisie")
        if submitted:
            st.session_state["grid_df_data"] = sanitize_df(df_edit)
            st.session_state.update({"athlete":athlete,"date":date_s,"note":note})
            st.success("Saisie enregistrée.")
    out_df = sanitize_df(st.session_state["grid_df_data"].copy())
    out_df = ensure_columns(out_df, ["Lactate (mmol/L)"])
    out_df["log10(lactate)"] = np.where(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")>0,
                                        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")), np.nan)
    buf = io.StringIO(); out_df.to_csv(buf,index=False)
    st.download_button("📄 Télécharger CSV", data=buf.getvalue(),
                       file_name=f"seance_{st.session_state.get('athlete','Anonyme')}_{st.session_state.get('date','')}.csv",
                       mime="text/csv")

# ---------------- Tab2: Résultats ----------------
with tab2:
    df_calc = sanitize_df(st.session_state["grid_df_data"].copy())
    df_calc = ensure_columns(df_calc, ["Vitesse (km/h)","Lactate (mmol/L)"])
    x = pd.to_numeric(df_calc["Vitesse (km/h)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_calc["Lactate (mmol/L)"], errors="coerce").to_numpy()
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(x, y, "o-", color="#1f77b4")
    ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Lactate (mmol/L)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(7,4))
    mask = y>0
    if np.sum(mask) > 1:
        ax2.plot(x[mask], np.log10(y[mask]), "o-", color="#1f77b4")
        ax2.set_xlabel("Vitesse (km/h)"); ax2.set_ylabel("log10(lactate)")
        ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# ---------------- Tab3: MLSS ----------------
# (Code MLSS complet déjà fourni dans le message précédent)

# ---------------- Tab4: SRS ----------------
with tab4:
    st.markdown("#### Paramétrage SRS")
    c1, c2, c3 = st.columns(3)
    with c1:
        slope = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1)
        sv1 = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
        sv2 = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    with c2:
        delta_step2 = st.number_input("Delta Step2 (km/h, ex. -0,8)", -5.0, 5.0, -0.8, step=0.1)
        step1 = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
        vo2_1 = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1)
    with c3:
        v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
        vo2_2 = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹) (optionnel)", 0.0, 100.0, 0.0, step=0.1)
        v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
    step2 = sv2 + delta_step2 if sv2 > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 is not None else "—")

    mrt1 = mrt2 = mrt_used = None
    if slope and slope > 0:
        alpha_s = slope / 60.0
        if v_equiv1 and step1 and v_equiv1 > 0 and step1 > 0:
            mrt1 = (v_equiv1 - step1) / alpha_s
        if v_equiv2 and step2 and v_equiv2 > 0 and step2 > 0:
            mrt2 = (v_equiv2 - step2) / alpha_s
        candidates = [x for x in [mrt1, mrt2] if x and x > 0]
        mrt_used = float(np.mean(candidates)) if candidates else None
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("MRT Step 1 (s)", f"{mrt1:.0f}" if mrt1 else "—")
    cm2.metric("MRT Step 2 (s)", f"{mrt2:.0f}" if mrt2 else "—")
    cm3.metric("MRT utilisé (s)", f"{mrt_used:.0f}" if mrt_used else "—")

    sv1_corr = sv2_corr = None
    if mrt_used and slope and slope > 0:
        alpha_s = slope / 60.0
        if sv1 and sv1 > 0: sv1_corr = sv1 - alpha_s * mrt_used
        if sv2 and sv2 > 0: sv2_corr = sv2 - alpha_s * mrt_used
    cv1, cv2 = st.columns(2)
    cv1.metric("SV1 corrigée (km/h)", f"{sv1_corr:.2f}" if sv1_corr else "—")
    cv2.metric("SV2 corrigée (km/h)", f"{sv2_corr:.2f}" if sv2_corr else "—")

    srs_img_b64 = None
    labels, raw_vals, corr_vals = [], [], []
    if sv1 and sv1 > 0:
        labels.append("SV1"); raw_vals.append(sv1); corr_vals.append(sv1_corr if sv1_corr is not None else np.nan)
    if sv2 and sv2 > 0:
        labels.append("SV2"); raw_vals.append(sv2); corr_vals.append(sv2_corr if sv2_corr is not None else np.nan)
    if labels and (np.isfinite(corr_vals).any()):
        xx = np.arange(len(labels)); w = 0.35
        fig_srs, ax_srs = plt.subplots(figsize=(6,3.5))
        ax_srs.bar(xx - w/2, raw_vals, width=w, label="Mesuré", color="#767676")
        ax_srs.bar(xx + w/2, corr_vals, width=w, label="Corrigé", color="#107c10")
        ax_srs.set_xticks(xx, labels); ax_srs.set_ylabel("Vitesse (km/h)")
        ax_srs.set_title("Correction des vitesses SV1 / SV2 par MRT")
        ax_srs.legend(); ax_srs.grid(axis="y", alpha=0.3)
        st.pyplot(fig_srs)
        srs_img_b64 = fig_to_base64(fig_srs)

    st.session_state.srs_results = {
        "slope": slope if slope>0 else "—",
        "sv1": sv1 if sv1>0 else None,
        "sv2": sv2 if sv2>0 else None,
        "step1": step1 if step1>0 else None,
        "step2": step2 if step2 else None,
        "vo2_1": vo2_1 if vo2_1>0 else None,
        "vo2_2": vo2_2 if vo2_2>0 else None,
        "v_equiv1": v_equiv1 if v_equiv1>0 else None,
        "v_equiv2": v_equiv2 if v_equiv2>0 else None,
        "mrt1": mrt1, "mrt2": mrt2, "mrt_used": mrt_used,
        "sv1_corr": sv1_corr, "sv2_corr": sv2_corr
    }
    st.session_state.srs_img_b64 = srs_img_b64

# ---------------- Tab5: Historique ----------------
with tab5:
    st.markdown("### Historique local (session)")
    hist = st.session_state.get("historique", [])
    st.write(f"Nombre de séances : **{len(hist)}**")
    uploaded = st.file_uploader("Importer une séance (CSV)", type=["csv"])
    if uploaded:
        try:
            tmp = pd.read_csv(uploaded)
            st.session_state["grid_df_data"] = sanitize_df(tmp)
            st.success("Séance importée (onglet Saisie).")
        except Exception as e:
            st.error(f"Erreur import CSV : {e}")
    if st.button("➕ Ajouter la séance courante à l'historique"):
        rec = {
            "athlete": st.session_state.get("athlete","Anonyme"),
            "date": st.session_state.get("date",""),
            "note": st.session_state.get("note",""),
            "vma": float(vma), "bsn": float(bsn),
            "df": sanitize_df(st.session_state["grid_df_data"].copy())
        }
        hist.append(rec)
        st.session_state["historique"] = hist
        st.success("Séance ajoutée à l’historique.")
    if hist:
        concat_rows = []
        for rec in hist:
            dfh = sanitize_df(rec["df"].copy())
            dfh["athlete"] = rec["athlete"]; dfh["date"] = rec["date"]
            dfh["vma"] = rec["vma"]; dfh["bsn"] = rec["bsn"]
            concat_rows.append(dfh)
        big = pd.concat(concat_rows, ignore_index=True)
        buf_hist = io.StringIO(); big.to_csv(buf_hist, index=False)
        st.download_button("📚 Télécharger l'historique (CSV)", data=buf_hist.getvalue(),
                           file_name="historique_tests_lactate.csv", mime="text/csv")