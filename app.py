
# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.7.5 MLSS dédié, fix valeur par défaut)
import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.7.5"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# -------------------- Helpers --------------------
def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    for col in list(df.columns):
        if str(col).strip().lower().startswith("fc estim"):
            df.drop(columns=[col], inplace=True, errors="ignore")
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    preferred = [
        "Palier",
        "%VMA",
        "Vitesse (km/h)",
        "Allure (min/km)",
        "Allure (mm:ss/km)",
        "FC mesurée (bpm)",
        "Lactate (mmol/L)",
    ]
    df = ensure_columns(df, preferred)
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]


def sanitize_mlss(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df, ["Temps (min)", "Lactate (mmol/L)", "FC (bpm)", "Commentaires"])
    df["Temps (min)"] = pd.to_numeric(df["Temps (min)"], errors="coerce")
    df["Lactate (mmol/L)"] = pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce")
    df["FC (bpm)"] = pd.to_numeric(df["FC (bpm)"], errors="coerce")
    df = df.dropna(how="all")
    if "Temps (min)" in df.columns:
        df = df.sort_values("Temps (min)")
    return df.reset_index(drop=True)


def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def round_to_step(val, step=0.1):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return round(val / step) * step

# -------------------- Sidebar --------------------
st.sidebar.header("Paramètres du test (paliers) - hors MLSS")
vma = st.sidebar.number_input("VMA (km/h)", 5.0, 30.0, 17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", 0.5, 4.0, 1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", 6, 20, 10, step=1)
pct_start = st.sidebar.number_input("%VMA départ", 40.0, 80.0, 60.0, step=1.0)
pct_end = st.sidebar.number_input("%VMA final", 80.0, 120.0, 105.0, step=1.0)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser la séance"):
    for k in [
        "grid_df_data",
        "df_mlss_lac",
        "mlss_img_b64",
        "mlss_suggestion_speed",
        "srs_results",
        "srs_img_b64",
        "athlete",
        "date",
        "note",
        "historique",
    ]:
        st.session_state.pop(k, None)
    st.rerun()
st.sidebar.caption(f"Version {VERSION}")

# -------------------- Base DF (paliers classiques) --------------------
pcts = np.linspace(pct_start, pct_end, int(n))
speeds = vma * pcts / 100.0


def pace_min_per_km(s):
    return 60 / s if s > 0 else np.nan


def pace_mmss(s):
    if s <= 0:
        return ""
    p = 60 / s
    m = int(p)
    sec = int(round((p - m) * 60))
    if sec == 60:
        m += 1
        sec = 0
    return f"{m:02d}:{sec:02d}"


base_df = pd.DataFrame(
    {
        "Palier": np.arange(1, int(n) + 1),
        "%VMA": np.round(pcts, 2),
        "Vitesse (km/h)": np.round(speeds, 2),
        "Allure (min/km)": np.round([pace_min_per_km(s) for s in speeds], 2),
        "Allure (mm:ss/km)": [pace_mmss(s) for s in speeds],
        "FC mesurée (bpm)": [None] * int(n),
        "Lactate (mmol/L)": [None] * int(n),
    }
)
base_df = sanitize_df(base_df)

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 Saisie", "📈 Résultats", "🧪 MLSS (30’ à vitesse constante)", "🏃‍♂️ SRS (Step–Ramp–Step)", "📂 Historique"]
)

# -------------------- Tab1: Saisie --------------------
with tab1:
    if "grid_df_data" not in st.session_state:
        st.session_state["grid_df_data"] = base_df.copy()
    else:
        st.session_state["grid_df_data"] = sanitize_df(st.session_state["grid_df_data"])

    st.markdown("### Saisie des lactates (et FC mesurée si dispo)")
    with st.form(key="saisie_form", clear_on_submit=False):
        df_edit = st.data_editor(
            st.session_state["grid_df_data"],
            key="grid_editor",
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
        )
        athlete = st.text_input("Athlète", value=st.session_state.get("athlete", "Anonyme"))
        date_s = st.date_input("Date").isoformat()
        note = st.text_input("Notes", value=st.session_state.get("note", ""))
        submitted = st.form_submit_button("💾 Enregistrer la saisie")
        if submitted:
            st.session_state["grid_df_data"] = sanitize_df(df_edit)
            st.session_state.update({"athlete": athlete, "date": date_s, "note": note})
            st.success("Saisie enregistrée.")

    out_df = sanitize_df(st.session_state["grid_df_data"].copy())
    out_df = ensure_columns(out_df, ["Lactate (mmol/L)"])
    out_df["log10(lactate)"] = np.where(
        pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce") > 0,
        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")),
        np.nan,
    )
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    st.download_button(
        "📄 Télécharger CSV",
        data=buf.getvalue(),
        file_name=f"seance_{st.session_state.get('athlete', 'Anonyme')}_{st.session_state.get('date', '')}.csv",
        mime="text/csv",
    )

# -------------------- Tab2: Résultats --------------------
with tab2:
    df_calc = sanitize_df(st.session_state["grid_df_data"].copy())
    df_calc = ensure_columns(df_calc, ["Vitesse (km/h)", "Lactate (mmol/L)"])
    x = pd.to_numeric(df_calc["Vitesse (km/h)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_calc["Lactate (mmol/L)"], errors="coerce").to_numpy()

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, y, "o-", color="#1f77b4")
    ax1.set_xlabel("Vitesse (km/h)")
    ax1.set_ylabel("Lactate (mmol/L)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    mask = y > 0
    if np.sum(mask) > 1:
        ax2.plot(x[mask], np.log10(y[mask]), "o-", color="#1f77b4")
    ax2.set_xlabel("Vitesse (km/h)")
    ax2.set_ylabel("log10(lactate)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# -------------------- Tab3: MLSS (30 min à vitesse constante) --------------------
with tab3:
    st.markdown("### MLSS – Palier **30 min** à **vitesse constante** (saisie toutes les 5 min)")

    # 1) SV2 et proposition 96% SV2
    c0, c1 = st.columns([1.2, 1])
    with c0:
        sv2 = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    with c1:
        suggested_from_sv2 = round_to_step(sv2 * 0.96, 0.1) if sv2 and sv2 > 0 else None

    # 2) Vitesse cible MLSS (pré-remplie par 96% SV2 ou 85% VMA)
    default_speed = suggested_from_sv2 if suggested_from_sv2 is not None else round_to_step(vma * 0.85, 0.1)
    # sécurise la valeur par défaut dans les bornes du widget
    MIN_V, MAX_V = 5.0, 30.0
    if default_speed is None or not np.isfinite(default_speed):
        default_speed = MIN_V
    default_speed = float(np.clip(default_speed, MIN_V, MAX_V))

    v_target = st.number_input(
        "Vitesse cible du test MLSS (km/h)",
        MIN_V, MAX_V, default_speed, step=0.1
    )
    if suggested_from_sv2:
        st.caption(f"Suggestion basée sur SV2: ≈ **{suggested_from_sv2:.1f} km/h** (96% de SV2)")

    # 3) Tableau fixe 6 points
    times = [5, 10, 15, 20, 25, 30]
    base = pd.DataFrame({
        "Temps (min)": times,
        "Lactate (mmol/L)": [np.nan]*6,
        "FC (bpm)": [np.nan]*6,
        "Commentaires": [""]*6,
    })

    cur = st.session_state.get("df_mlss_lac", None)
    if cur is None or not isinstance(cur, pd.DataFrame):
        cur = base.copy()
    else:
        cur = sanitize_mlss(cur)
        if "Temps (min)" not in cur.columns:
            cur = base.copy()
        else:
            cur = base.merge(cur, on=["Temps (min)"], how="left", suffixes=("", "_y"))
            for col in ["Lactate (mmol/L)", "FC (bpm)", "Commentaires"]:
                y = f"{col}_y"
                if y in cur.columns:
                    cur[col] = cur[y].combine_first(cur[col])
                    cur.drop(columns=[y], inplace=True)
    st.session_state["df_mlss_lac"] = cur

    st.markdown("#### Saisie unique des mesures (aucun bouton de sauvegarde)")
    df_display = st.data_editor(
        st.session_state["df_mlss_lac"],
        key="mlss_editor",
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Temps (min)": st.column_config.NumberColumn(disabled=True),
            "Lactate (mmol/L)": st.column_config.NumberColumn(step=0.1),
            "FC (bpm)": st.column_config.NumberColumn(step=1),
            "Commentaires": st.column_config.TextColumn(),
        },
    )
    st.session_state["df_mlss_lac"] = sanitize_mlss(df_display)

    # 4) Analyse interne
    DELTA_10_30_THR = 0.5  # mmol/L
    SLOPE_THR = 0.02       # mmol·L⁻¹·min⁻¹

    dfm = st.session_state["df_mlss_lac"].copy()
    t = pd.to_numeric(dfm["Temps (min)"], errors="coerce").to_numpy()
    lac = pd.to_numeric(dfm["Lactate (mmol/L)"], errors="coerce").to_numpy()
    hr = pd.to_numeric(dfm["FC (bpm)"], errors="coerce").to_numpy()

    valid = np.isfinite(t) & np.isfinite(lac)
    t_valid, lac_valid = t[valid], lac[valid]

    # Δ10→30
    delta_10_30 = None
    try:
        idx10 = int(np.where(t == 10)[0][0])
        idx30 = int(np.where(t == 30)[0][0])
        v10 = lac[idx10]
        v30 = lac[idx30]
        if np.isfinite(v10) and np.isfinite(v30):
            delta_10_30 = float(v30 - v10)
    except Exception:
        delta_10_30 = None

    # Pente globale
    slope = None
    if len(t_valid) >= 2:
        slope = float(np.polyfit(t_valid, lac_valid, 1)[0])

    stable = False
    if (delta_10_30 is not None) and (slope is not None):
        stable = (abs(delta_10_30) <= DELTA_10_30_THR) and (abs(slope) <= SLOPE_THR)

    if len(t_valid) < 2:
        st.info("Renseigne au moins deux mesures de lactate pour évaluer la stabilité.")
    else:
        cm_a, cm_b, cm_c = st.columns(3)
        cm_a.metric("Δ lactate 10→30 min", f"{delta_10_30:.2f} mmol/L" if delta_10_30 is not None else "—")
        cm_b.metric("Pente dL/dt", f"{slope:.3f} mmol·L⁻¹·min⁻¹" if slope is not None else "—")
        cm_c.metric("Bsn", f"{bsn:.1f} mmol/L")
        if stable:
            st.success("Lactate **stable** → compatible MLSS pour la vitesse testée.")
        else:
            st.error("Lactate **non stable** → au-dessus/au‑dessous du MLSS.")

    # 5) Graphique
    if len(t_valid) >= 2:
        fig_mlss, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, lac, "o-", color="#005a9e", label="Lactate")
        ax.axhline(bsn, color="#767676", linestyle="--", linewidth=1, label=f"Bsn ≈ {bsn:.1f} mmol/L")
        ax.set_xlabel("Temps (min)"); ax.set_ylabel("Lactate (mmol/L)")
        ax.grid(True, alpha=0.3)
        if np.isfinite(hr).sum() >= 2:
            ax2 = ax.twinx(); ax2.plot(t, hr, "s-", color="#d83b01", label="FC (bpm)"); ax2.set_ylabel("FC (bpm)")
            lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper left")
        else:
            ax.legend(loc="upper left")
        st.pyplot(fig_mlss)
        st.session_state["mlss_img_b64"] = fig_to_base64(fig_mlss)

    # 6) Suggestion de vitesse
    def step_from_slope(s):
        a = abs(s)
        if a <= 0.02: return 0.2
        if a <= 0.04: return 0.3
        if a <= 0.06: return 0.5
        return 0.8

    suggestion = None; rationale = ""
    if slope is not None:
        step = step_from_slope(slope)
        vs_from_sv2 = suggested_from_sv2  # peut être None
        if stable:
            if vs_from_sv2:
                suggestion = round_to_step(vs_from_sv2, 0.1)
                rationale = "Stable : confirmer ~96% de SV2 pour validation finale."
            else:
                suggestion = round_to_step(v_target, 0.1)
                rationale = "Stable : conserver la vitesse testée (ou affiner ±0,1–0,2 km/h)."
        else:
            if slope > SLOPE_THR:
                candidate = round_to_step(max(MIN_V, v_target - step), 0.1)
                if vs_from_sv2 and vs_from_sv2 < v_target:
                    suggestion = vs_from_sv2
                    rationale = "Lactate en hausse : revenir vers ~96% SV2."
                else:
                    suggestion = candidate
                    rationale = f"Lactate en hausse : baisse ~{step:.1f} km/h."
            elif slope < -SLOPE_THR:
                candidate = round_to_step(min(MAX_V, v_target + step), 0.1)
                if vs_from_sv2 and vs_from_sv2 > v_target:
                    suggestion = vs_from_sv2
                    rationale = "Lactate en baisse : se rapprocher de ~96% SV2."
                else:
                    suggestion = candidate
                    rationale = f"Lactate en baisse : augmente ~{step:.1f} km/h."
            else:
                suggestion = round_to_step(v_target, 0.1)
                rationale = "Pente quasi nulle : conserver/affiner ±0,1–0,2 km/h."

    if suggestion is not None:
        st.session_state["mlss_suggestion_speed"] = suggestion
        st.info(f"Vitesse suggérée pour le prochain test : **{suggestion:.1f} km/h**. {rationale}")

    # Export PNG
    mlss_png_b64 = st.session_state.get("mlss_img_b64")
    if mlss_png_b64:
        st.download_button(
            "📷 Télécharger le graphique MLSS (PNG)",
            data=base64.b64decode(mlss_png_b64.encode("utf-8")),
            file_name="mlss_plot.png",
            mime="image/png",
        )

# -------------------- Tab4 & Tab5 (inchangés) --------------------
with tab4:
    st.markdown("#### Paramétrage SRS")
    c1, c2, c3 = st.columns(3)
    with c1:
        slope_r = st.number_input("Pente rampe (km/h/min)", 0.0, 10.0, 0.0, step=0.1)
        sv1 = st.number_input("SV1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
        sv2_srs = st.number_input("SV2 (km/h)", 0.0, 30.0, 0.0, step=0.1)
    with c2:
        delta_step2 = st.number_input("Delta Step2 (km/h, ex. -0,8)", -5.0, 5.0, -0.8, step=0.1)
        step1 = st.number_input("Vitesse Step 1 (km/h)", 0.0, 30.0, 0.0, step=0.1)
        vo2_1 = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", 0.0, 100.0, 0.0, step=0.1)
    with c3:
        v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)
        vo2_2 = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹) (optionnel)", 0.0, 100.0, 0.0, step=0.1)
        v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", 0.0, 30.0, 0.0, step=0.1)

    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
    st.metric("Vitesse Step 2 (km/h)", f"{step2:.1f}" if step2 is not None else "—")

    mrt1 = mrt2 = mrt_used = None
    if slope_r and slope_r > 0:
        alpha_s = slope_r / 60.0
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
    if mrt_used and slope_r and slope_r > 0:
        alpha_s = slope_r / 60.0
        if sv1 and sv1 > 0:
            sv1_corr = sv1 - alpha_s * mrt_used
        if sv2_srs and sv2_srs > 0:
            sv2_corr = sv2_srs - alpha_s * mrt_used

    cv1, cv2 = st.columns(2)
    cv1.metric("SV1 corrigée (km/h)", f"{sv1_corr:.2f}" if sv1_corr else "—")
    cv2.metric("SV2 corrigée (km/h)", f"{sv2_corr:.2f}" if sv2_corr else "—")

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
            "athlete": st.session_state.get("athlete", "Anonyme"),
            "date": st.session_state.get("date", ""),
            "note": st.session_state.get("note", ""),
            "vma": float(vma),
            "bsn": float(bsn),
            "df": sanitize_df(st.session_state["grid_df_data"].copy()),
            "mlss_vitesse": st.session_state.get("mlss_suggestion_speed", np.nan),
        }
        hist.append(rec)
        st.session_state["historique"] = hist
        st.success("Séance ajoutée à l’historique.")

    if hist:
        concat_rows = []
        for rec in hist:
            dfh = sanitize_df(rec["df"].copy())
            dfh["athlete"] = rec["athlete"]
            dfh["date"] = rec["date"]
            dfh["vma"] = rec["vma"]
            dfh["bsn"] = rec["bsn"]
            dfh["mlss_vitesse"] = rec.get("mlss_vitesse", np.nan)
            concat_rows.append(dfh)
        big = pd.concat(concat_rows, ignore_index=True)
        buf_hist = io.StringIO(); big.to_csv(buf_hist, index=False)
        st.download_button(
            "📚 Télécharger l'historique (CSV)", data=buf_hist.getvalue(),
            file_name="historique_tests_lactate.csv", mime="text/csv"
        )
