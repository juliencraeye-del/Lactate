# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.7.2)
# Corrections:
# - Graphiques in-app: fig_to_base64 ne ferme plus la figure (évite affichage cassé)
# - Export HTML: balises data:image/png;base64,... pour TOUTES les figures
# - Saisie sécurisée via st.form: plus de perte à la 1re saisie
# - MLSS: fallback quand Δ10→30 indisponible; ajustements ±0.2/±0.3 km/h selon la pente; arrondi au 0,1 km/h
# - "FC estimée" supprimée partout; robustesse KeyError; width="stretch" pour data_editor

import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.7.2"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ---------------- Helpers ----------------
def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    # Retire toute variante de "FC estimée"
    for col in list(df.columns):
        if str(col).strip().lower().startswith("fc estim"):
            df.drop(columns=[col], inplace=True, errors="ignore")
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    preferred = ["Palier","%VMA","Vitesse (km/h)","Allure (min/km)","Allure (mm:ss/km)","FC mesurée (bpm)","Lactate (mmol/L)"]
    df = ensure_columns(df, preferred)
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]

def sanitize_mlss(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = ensure_columns(df, ["Temps (min)","Lactate (mmol/L)","FC (bpm)","Commentaires"])
    df["Temps (min)"] = pd.to_numeric(df["Temps (min)"], errors="coerce")
    df["Lactate (mmol/L)"] = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce")
    df["FC (bpm)"] = pd.to_numeric(df["FC (bpm)"], errors="coerce")
    df = df.dropna(how="all")
    df = df.sort_values("Temps (min)").reset_index(drop=True)
    return df

def fig_to_base64(fig, dpi=150):
    """Encode la figure en base64 sans la fermer (important pour l’affichage in-app)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    # plt.close(fig)  # ⟵ NE PAS fermer ici; Streamlit gère le cycle d’affichage
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def round_to_step(val, step=0.1):
    if val is None: return None
    return round(val/step)*step

# ----- MLSS control -----
def mlss_control(df_mlss: pd.DataFrame, amplitude_thr: float = 1.0):
    """
    Contrôle MLSS (fenêtre 10–30 min):
    - amplitude (max-min)
    - pente (mmol/L/min) via régression linéaire
    - Δ10→30' si dispo
    - suggestion vitesse selon dérive (pente) avec fallback si Δ manquant
    """
    result = {
        "n_points": 0, "amplitude": None, "slope_mmol_per_min": None, "delta_10_30": None,
        "stable": None, "suggest_kmh": None, "suggest_note": None
    }
    if df_mlss is None or df_mlss.empty: return result

    win = df_mlss[(df_mlss["Temps (min)"] >= 10) & (df_mlss["Temps (min)"] <= 30)].dropna(subset=["Lactate (mmol/L)"])
    if win.empty or win["Lactate (mmol/L)"].notna().sum() < 2:
        result["n_points"] = int(win["Lactate (mmol/L)"].notna().sum())
        result["suggest_note"] = "Données insuffisantes (≥2 points 10–30')."
        return result

    lac = win["Lactate (mmol/L)"].to_numpy(); t = win["Temps (min)"].to_numpy()
    result["n_points"] = int(len(lac))
    amp = float(np.nanmax(lac) - np.nanmin(lac))
    result["amplitude"] = amp

    # Pente linéaire
    try:
        slope = float(np.polyfit(t, lac, 1)[0])
    except Exception:
        slope = None
    result["slope_mmol_per_min"] = slope

    # Δ10→30' si dispo
    lac_10 = win[win["Temps (min)"]==10]["Lactate (mmol/L)"]
    lac_30 = win[win["Temps (min)"]==30]["Lactate (mmol/L)"]
    d10_30 = None
    if lac_10.notna().any() and lac_30.notna().any():
        d10_30 = float(lac_30.iloc[-1] - lac_10.iloc[0])
    result["delta_10_30"] = d10_30

    # Stabilité
    stable = (amp <= amplitude_thr)
    result["stable"] = stable

    # Suggestion vitesse (fallback cohérent) :
    suggest = 0.0
    note = "Variation ≤ 1.0 mmol/L : vitesse convenable." if stable else "Instabilité détectée (amplitude > 1.0 mmol/L)."
    if d10_30 is not None:
        if d10_30 > 1.0: suggest = -0.3; note = "Lactate ↑ (> +1.0) : réduire ~0,3 km/h."
        elif 0.5 < d10_30 <= 1.0: suggest = -0.2; note = "Lactate ↑ (+0,5 à +1,0) : réduire ~0,2 km/h."
        elif -1.0 < d10_30 < -0.5: suggest = +0.2; note = "Lactate ↓ (−0,5 à −1,0) : augmenter ~0,2 km/h."
        elif d10_30 <= -1.0: suggest = +0.3; note = "Lactate ↓ (< −1,0) : augmenter ~0,3 km/h."
        else:
            suggest = 0.0; note = "Variation ≤ 0,5 mmol/L : vitesse convenable."
    else:
        if not stable and slope is not None:
            if slope > +0.15: suggest = -0.3; note = "Pente positive (> +0,15) : réduire ~0,3 km/h."
            elif +0.05 <= slope <= +0.15: suggest = -0.2; note = "Pente positive (+0,05 à +0,15) : réduire ~0,2 km/h."
            elif slope < -0.15: suggest = +0.3; note = "Pente négative (< −0,15) : augmenter ~0,3 km/h."
            elif -0.15 <= slope <= -0.05: suggest = +0.2; note = "Pente négative (−0,05 à −0,15) : augmenter ~0,2 km/h."
            else:
                suggest = 0.0; note = "Pente faible; ajuste au besoin selon protocole."
        elif stable:
            suggest = 0.0; note = "Stabilité satisfaisante."

    result["suggest_kmh"] = suggest
    result["suggest_note"] = note
    return result

# ---------------- Sidebar ----------------
st.sidebar.header("Paramètres du test")
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", 6, 20, 10, step=1)
pct_start = st.sidebar.number_input("%VMA départ", 40.0, 80.0, 60.0, step=1.0)
pct_end   = st.sidebar.number_input("%VMA final", 80.0, 120.0, 105.0, step=1.0)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser la séance"):
    for k in ["grid_df_data","df_mlss_lac","mlss_params","mlss_img_b64",
              "srs_results","srs_img_b64","athlete","date","note","historique"]:
        st.session_state.pop(k, None)
    st.rerun()
st.sidebar.caption(f"Version {VERSION}")

# ---------------- Base DF ----------------
pcts   = np.linspace(pct_start, pct_end, int(n))
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
    ["📝 Saisie", "📈 Résultats", "🧪 MLSS", "🏃‍♂️ SRS (Step–Ramp–Step)", "🗂️ Historique"]
)

# ---------------- Tab1: Saisie (form) ----------------
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
        date_s  = st.date_input("Date").isoformat()
        note    = st.text_input("Notes", value=st.session_state.get("note",""))
        submitted = st.form_submit_button("💾 Enregistrer la saisie")
    if submitted:
        st.session_state["grid_df_data"] = sanitize_df(df_edit)
        st.session_state.update({"athlete":athlete,"date":date_s,"note":note})
        st.success("Saisie enregistrée.")

    # Export CSV
    out_df = sanitize_df(st.session_state["grid_df_data"].copy())
    out_df = ensure_columns(out_df, ["Lactate (mmol/L)"])
    out_df["log10(lactate)"] = np.where(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")>0,
                                        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")), np.nan)
    buf = io.StringIO(); out_df.to_csv(buf,index=False)
    st.download_button("📄 Télécharger CSV", data=buf.getvalue(),
                       file_name=f"seance_{st.session_state.get('athlete','Anonyme')}_{st.session_state.get('date','')}.csv",
                       mime="text/csv")

# ---------------- Tab2: Résultats + Export HTML ----------------
with tab2:
    df_calc = sanitize_df(st.session_state["grid_df_data"].copy())
    df_calc = ensure_columns(df_calc, ["Vitesse (km/h)","Lactate (mmol/L)"])
    x = pd.to_numeric(df_calc["Vitesse (km/h)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_calc["Lactate (mmol/L)"], errors="coerce").to_numpy()

    # Graphe Lactate - Vitesse
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(x, y, "o-", color="#1f77b4")
    ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Lactate (mmol/L)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Graphe Log-lactate - Vitesse
    fig2, ax2 = plt.subplots(figsize=(7,4))
    mask = y>0
    if np.sum(mask) > 1:
        ax2.plot(x[mask], np.log10(y[mask]), "o-", color="#1f77b4")
    ax2.set_xlabel("Vitesse (km/h)"); ax2.set_ylabel("log10(lactate)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Encodage (ne ferme pas les figures)
    img1_b64 = fig_to_base64(fig1)
    img2_b64 = fig_to_base64(fig2)

    # MLSS/SRS (pour export)
    df_mlss = sanitize_mlss(st.session_state.get("df_mlss_lac", pd.DataFrame()))
    mlss_table_html = df_mlss.to_html(index=False) if not df_mlss.empty else ""
    mlss_img_b64    = st.session_state.get("mlss_img_b64", None)
    mlss_params     = st.session_state.get("mlss_params", {})

    srs         = st.session_state.get("srs_results", {})
    srs_img_b64 = st.session_state.get("srs_img_b64", None)

    # HTML autonome: <img> pour toutes les figures
    html = f"""
    <!DOCTYPE html>
    <html lang="fr"><head><meta charset="utf-8">
    <title>Rapport – Tests MLSS & SRS</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
      h1 {{ color: #0067b8; }}
      h2 {{ color: #004578; margin-top: 24px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
      th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: center; }}
      .meta {{ color:#555; }}
      img {{ max-width: 100%; height: auto; border:1px solid #eee; }}
    </style></head><body>
      <h1>Rapport – Test lactate & SRS</h1>
      <p class="meta"><b>Athlète:</b> {st.session_state.get("athlete","Anonyme")} &nbsp;
         <b>Date:</b> {st.session_state.get("date","")} &nbsp;
         <b>VMA:</b> {vma:.2f} km/h &nbsp; <b>Bsn:</b> {bsn:.2f} mmol/L</p>
      <p class="meta"><b>Notes:</b> {st.session_state.get("note","")}</p>

      <h2>Données brutes</h2>
      {df_calc.to_html(index=False)}

      <h2>Graphiques lactate</h2>
      <h3>Lactate – Vitesse</h3>
      data:image/png;base64,{img1_b64}
      <h3>Log(lactate) – Vitesse</h3>
      <img src="data:image/png;base64,{img2_b64}" alt="Log(lact   <ul>
        <li><b>SV2</b> : {mlss_params.get("sv2","—")} km/h</li>
        <li><b>Méthode</b> : {mlss_params.get("method","—")}</li>
        <li><b>Paramètre</b> : {mlss_params.get("param","—")}</li>
        <li><b>Vitesse théorique MLSS</b> : {mlss_params.get("v_theo","—")} km/h</li>
        <li><b>Contrôle MLSS (10–30')</b> :
            points={mlss_params.get("mlss_ctrl_n","0")},
            amplitude={mlss_params.get("mlss_ctrl_amp","—")} mmol/L,
            pente={mlss_params.get("mlss_ctrl_slope","—")} mmol/L/min,
            Δ10→30'={mlss_params.get("mlss_ctrl_d10_30","—")} mmol/L,
            stable={mlss_params.get("mlss_ctrl_stable","—")}</li>
        <li><b>Suggestion vitesse</b> : {mlss_params.get("mlss_ctrl_suggest_note","—")}
            → vitesse conseillée : {mlss_params.get("mlss_ctrl_v_suggest","—")} km/h</li>
      </ul>
      {mlss_table_html}
      {"<img src=\"data:image/png;base64,"+mlss_img_b64+"\" alt=\"Courbe MLSS\"/>" if mlss_img_b64 else "<p><i>(Ajoutez ≥2 valeurs 10–30' pour la courbe MLSS.)</i></p>"}

      <h2>Step–Ramp–Step (SRS)</h2>
      <ul>
        <li><b>Pente rampe</b> : {srs.get("slope","—")} km/h/min</li>
        <li><b>MRT Step 1</b> : {("—" if srs.get("mrt1") is None else round(srs.get("mrt1"),1))} s ;
            <b>MRT Step 2</b> : {("—" if srs.get("mrt2") is None else round(srs.get("mrt2"),1))} s ;
            <b>MRT utilisé</b> : {("—" if srs.get("mrt_used") is None else round(srs.get("mrt_used"),1))} s</li>
        <li><b>SV1 corrigée</b> : {("—" if srs.get("sv1_corr") is None else round(srs.get("sv1_corr"),2))} km/h ;
            <b>SV2 corrigée</b> : {("—" if srs.get("sv2_corr") is None else round(srs.get("sv2_corr"),2))} km/h</li>
      </ul>
      {"<img src=\"data:image/png;base64,"+srs_img_b64+"\" alt=\"Comparatif SRS\"/>" if srs_img_b64 else "<p><i>(Paramétrez SRS pour afficher le comparatif.)</i></p>"}

      <p class="meta" style="margin-top:24px;">Généré par l’app Streamlit (version {VERSION}).</p>
    </body></html>
    """

    st.download_button("🧾 Télécharger le rapport HTML",
                       data=html.encode("utf-8"),
                       file_name=f"rapport_{st.session_state.get('athlete','Anonyme')}.html",
                       mime="text/html")

# ---------------- Tab3: MLSS ----------------
with tab3:
    st.markdown("#### MLSS – Paramètres et saisie")
    sv2_mlss = st.number_input("Vitesse SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)

    mlss_use_percent = st.toggle("Utiliser % de SV2 (sinon Delta fixe)", value=True)
    if mlss_use_percent:
        pct_mlss = st.number_input("Pourcentage de SV2 (%)", 80.0, 100.0, 96.0, step=0.5)
        v_theo_mlss = (sv2_mlss * pct_mlss/100.0) if sv2_mlss > 0 else None
        mlss_method = "Pourcentage SV2"
        mlss_param  = f"{pct_mlss:.1f}%"
    else:
        delta_mlss = st.number_input("Delta vs SV2 (km/h)", -5.0, 5.0, -0.6, step=0.1)
        v_theo_mlss = (sv2_mlss + delta_mlss) if sv2_mlss > 0 else None
        mlss_method = "Delta fixe"
        mlss_param  = f"{delta_mlss:.1f} km/h"

    st.metric("Vitesse théorique MLSS", f"{v_theo_mlss:.1f}" if v_theo_mlss else "—")

    if "df_mlss_lac" not in st.session_state:
        st.session_state.df_mlss_lac = pd.DataFrame({
            "Temps (min)": [0,5,10,15,20,25,30],
            "Lactate (mmol/L)": [None]*7,
            "FC (bpm)": [None]*7,
            "Commentaires": [""]*7
        })
    df_mlss = st.data_editor(st.session_state.df_mlss_lac, width="stretch", hide_index=True)
    st.session_state.df_mlss_lac = sanitize_mlss(df_mlss)

    plot_df = sanitize_mlss(st.session_state.df_mlss_lac)
    mlss_img_b64 = None
    if plot_df["Lactate (mmol/L)"].notna().sum() >= 2:
        fig_mlss, ax_mlss = plt.subplots(figsize=(6,3.5))
        ax_mlss.plot(plot_df["Temps (min)"], plot_df["Lactate (mmol/L)"], "o-", color="#0078d4")
        ax_mlss.set_title("MLSS – évolution du lactate")
        ax_mlss.set_xlabel("Temps (min)"); ax_mlss.set_ylabel("Lactate (mmol/L)")
        ax_mlss.grid(True, alpha=0.3)
        st.pyplot(fig_mlss)
        mlss_img_b64 = fig_to_base64(fig_mlss)

    ctrl = mlss_control(plot_df, amplitude_thr=1.0)
    v_suggest = None
    if v_theo_mlss:
        v_suggest = round_to_step(v_theo_mlss + (ctrl["suggest_kmh"] or 0.0), step=0.1)

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Points (10–30')", f"{ctrl['n_points']}")
    cB.metric("Amplitude (mmol/L)", f"{ctrl['amplitude']:.2f}" if ctrl["amplitude"] is not None else "—")
    cC.metric("Pente (mmol/L/min)", f"{ctrl['slope_mmol_per_min']:.3f}" if ctrl["slope_mmol_per_min"] is not None else "—")
    cD.metric("Δ10→30' (mmol/L)", f"{ctrl['delta_10_30']:.2f}" if ctrl["delta_10_30"] is not None else "—")

    st.info(f"Stabilité : {'✅ Stable' if ctrl['stable'] else '⚠️ Instable' if ctrl['stable'] is not None else '—'}")
    st.write(f"Suggestion : {ctrl['suggest_note']} → **Vitesse conseillée**: {v_suggest if v_suggest is not None else '—'} km/h")

    st.session_state.mlss_params = {
        "sv2": round(sv2_mlss,1) if sv2_mlss>0 else "—",
        "method": mlss_method,
        "param": mlss_param,
        "v_theo": round(v_theo_mlss,1) if v_theo_mlss else "—",
        "mlss_ctrl_n": ctrl["n_points"],
        "mlss_ctrl_amp": round(ctrl["amplitude"],2) if ctrl["amplitude"] is not None else "—",
        "mlss_ctrl_slope": round(ctrl["slope_mmol_per_min"],3) if ctrl["slope_mmol_per_min"] is not None else "—",
        "mlss_ctrl_d10_30": round(ctrl["delta_10_30"],2) if ctrl["delta_10_30"] is not None else "—",
        "mlss_ctrl_stable": ("Oui" if ctrl["stable"] else "Non") if ctrl["stable"] is not None else "—",
        "mlss_ctrl_suggest_note": ctrl["suggest_note"] or "—",
        "mlss_ctrl_v_suggest": v_suggest if v_suggest is not None else "—"
    }
    st.session_state.mlss_img_b64 = mlss_img_b64

# ---------------- Tab4: SRS ----------------
with tab4:
    st.markdown("#### Paramétrage SRS")
    c1, c2, c3 = st.columns(3)
    with c1:
        slope = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1)
        sv1   = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
        sv2   = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    with c2:
        delta_step2 = st.number_input("Delta Step2 (km/h, ex. -0,8)", -5.0, 5.0, -0.8, step=0.1)
        step1       = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
        vo2_1       = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1)
    with c3:
        v_equiv1    = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
        vo2_2       = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹) (optionnel)", 0.0, 100.0, 0.0, step=0.1)
        v_equiv2    = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)

    step2 = sv2 + delta_step2 if sv2 > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 else "—")

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
            "date":    st.session_state.get("date",""),
            "note":    st.session_state.get("note",""),
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