# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (v0.5.2)
# Correctifs majeurs :
# - Suppression robuste de "FC estimée" (affichage, session, import/export)
# - Conservation de la 1re saisie (init de session unique, pas d’écrasement)
# - Export HTML : data:image/png;base64,... pour TOUTES les figures
# - MLSS : coercition numérique + tri par "Temps (min)" avant tracé/export
# - Petites corrections de clés, f-strings et nettoyage centralisé

import io, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.5.2"
st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ----------------------------- Helpers -----------------------------
def pace_min_per_km(speed_kmh: float):
    if speed_kmh and speed_kmh > 0:
        return 60.0 / speed_kmh
    return np.nan

def pace_mmss(speed_kmh: float):
    p = pace_min_per_km(speed_kmh)
    if np.isnan(p): return ""
    m = int(p); s = int(round((p - m) * 60))
    if s == 60: m += 1; s = 0
    return f"{m:02d}:{s:02d}"

def linear_regression(x, y):
    if len(x) < 2:
        return np.nan, np.nan, np.full_like(x, np.nan, dtype=float), np.inf
    m, b = np.polyfit(x, y, 1)
    yhat = m * np.array(x) + b
    sse = float(((np.array(y) - yhat) ** 2).sum())
    return m, b, yhat, sse

def interp_speed_at_lactate(x, y, target):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2: return np.nan
    for k in range(1, len(x)):
        if y[k-1] < target <= y[k]:
            if y[k] == y[k-1]: return float(x[k])
            return float(x[k-1] + (target - y[k-1]) * (x[k] - x[k-1]) / (y[k] - y[k-1]))
    return np.nan

def point_to_line_distances(xs, ys, x1, y1, x2, y2):
    xs = np.array(xs, dtype=float); ys = np.array(ys, dtype=float)
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    A = y2 - y1; B = x1 - x2; C = x2 * y1 - y2 * x1
    denom = np.sqrt(A*A + B*B)
    if denom == 0 or len(xs) == 0:
        return np.array([]), np.array([])
    d = np.abs(A*xs + B*ys + C) / denom
    return d, mask

def dmax_lt2(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y); x, y = x[mask], y[mask]
    if len(x) < 3: return np.nan, None
    d, _ = point_to_line_distances(x, y, x[0], y[0], x[-1], y[-1])
    if d.size == 0: return np.nan, None
    idx = int(np.nanargmax(d))
    return float(x[idx]), {"distances": d, "idx": idx, "x": x, "y": y}

def moddmax_lt2(x, y, x_start, y_start):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y); x, y = x[mask], y[mask]
    if len(x) < 3 or np.isnan(x_start) or np.isnan(y_start):
        return np.nan, None
    d, _ = point_to_line_distances(x, y, x_start, y_start, x[-1], y[-1])
    if d.size == 0: return np.nan, None
    idx = int(np.nanargmax(d)); 
    return float(x[idx]), {"distances": d, "idx": idx, "x": x, "y": y, "x_start": x_start, "y_start": y_start}

def lt1_log_lactate_breakpoint(x, y_lac):
    mask = ~np.isnan(x) & ~np.isnan(y_lac) & (y_lac > 0)
    x = np.array(x)[mask]; y = np.array(y_lac)[mask]
    if len(x) < 4: return np.nan, None
    ylog = np.log10(y)
    best = {"idx": None, "sse": np.inf, "m1": np.nan, "b1": np.nan, "m2": np.nan, "b2": np.nan}
    for b in range(1, len(x) - 2 + 1):
        x1, y1 = x[: b + 1], ylog[: b + 1]
        x2, y2 = x[b:], ylog[b:]
        if len(x1) < 2 or len(x2) < 2: continue
        m1, c1, _, sse1 = linear_regression(x1, y1)
        m2, c2, _, sse2 = linear_regression(x2, y2)
        sse = sse1 + sse2
        if sse < best["sse"]:
            best = {"idx": b, "sse": sse, "m1": m1, "b1": c1, "m2": m2, "b2": c2}
    if best["idx"] is None: return np.nan, None
    bp_speed = x[best["idx"]]
    best["x"] = x; best["ylog"] = ylog
    return float(bp_speed), best

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# --- SANITIZE functions ---
def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la saisie principale : supprime 'FC estimée', colonnes/lignes vides, réindexe."""
    if df is None or df.empty: 
        return df
    # Drop variantes de 'FC estimée'
    for col in list(df.columns):
        lc = str(col).strip().lower()
        if lc in ("fc estimée (bpm)", "fc estimee (bpm)", "fc estimée", "fc estimee"):
            df = df.drop(columns=[col], errors="ignore")
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    # Forcer l’ordre des colonnes attendues si présentes
    preferred = ["Palier", "%VMA", "Vitesse (km/h)", "Allure (min/km)", "Allure (mm:ss/km)", "FC mesurée (bpm)", "Lactate (mmol/L)"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]

def sanitize_mlss(df_mlss: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le tableau MLSS : convertit en numérique, tri par Temps (min)."""
    if df_mlss is None or df_mlss.empty:
        return df_mlss
    out = df_mlss.copy()
    if "Temps (min)" in out.columns:
        out["Temps (min)"] = pd.to_numeric(out["Temps (min)"], errors="coerce")
    if "Lactate (mmol/L)" in out.columns:
        out["Lactate (mmol/L)"] = pd.to_numeric(out["Lactate (mmol/L)"], errors="coerce")
    if "FC (bpm)" in out.columns:
        out["FC (bpm)"] = pd.to_numeric(out["FC (bpm)"], errors="coerce")
    out = out.dropna(how="all")
    if "Temps (min)" in out.columns:
        out = out.sort_values("Temps (min)").reset_index(drop=True)
    return out

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Paramètres du test")
vma = st.sidebar.number_input("VMA (km/h)", min_value=5.0, max_value=30.0, value=17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", min_value=6, max_value=20, value=10, step=1)
pct_start = st.sidebar.number_input("%VMA palier de départ", min_value=40.0, max_value=80.0, value=60.0, step=1.0)
pct_end = st.sidebar.number_input("%VMA palier final", min_value=80.0, max_value=120.0, value=105.0, step=1.0)
duree_min = st.sidebar.number_input("Durée d'un palier (min) – informatif", min_value=2.0, max_value=6.0, value=3.0, step=0.5)

st.sidebar.markdown("---")
use_poly = st.sidebar.toggle("Lisser la courbe (D‑max polynomial)", value=False)
poly_order = st.sidebar.select_slider("Ordre du polynôme", options=[2, 3], value=2, disabled=not use_poly)

# Reset
if st.sidebar.button("🔄 Réinitialiser la séance"):
    for key in ["grid_df_data", "historique", "plots", "grid_editor", "athlete", "date", "note",
                "df_mlss_lac", "mlss_params", "mlss_img_b64", "srs_results", "srs_img_b64"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
st.sidebar.caption(f"Version {VERSION}")

# ----------------------------- Base DF -----------------------------
pcts = np.linspace(pct_start, pct_end, int(n))
speeds = vma * pcts / 100.0
allure_min = np.array([pace_min_per_km(s) for s in speeds])
allure_txt = np.array([pace_mmss(s) for s in speeds])

# A la source : PAS de "FC estimée (bpm)"
base_df = pd.DataFrame({
    "Palier": np.arange(1, int(n)+1, dtype=int),
    "%VMA": np.round(pcts, 2),
    "Vitesse (km/h)": np.round(speeds, 2),
    "Allure (min/km)": np.round(allure_min, 2),
    "Allure (mm:ss/km)": allure_txt,
    "FC mesurée (bpm)": [None]*int(n),
    "Lactate (mmol/L)": [None]*int(n)
})

# ----------------------------- Onglets -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 Saisie", "📈 Résultats", "🧪 MLSS", "🏃‍♂️ SRS (Step–Ramp–Step)", "🗂️ Historique & Comparaison"]
)

# ----------------------------- Onglet 1: Saisie -----------------------------
with tab1:
    st.markdown("### Saisie des lactates (et FC mesurée si dispo)")

    # Initialisation unique (évite la perte après 1re saisie)
    if "grid_df_data" not in st.session_state:
        st.session_state["grid_df_data"] = base_df.copy()
    else:
        # Sanitize si ancienne session contenait "FC estimée"
        st.session_state["grid_df_data"] = sanitize_df(st.session_state["grid_df_data"])

    initial_df = st.session_state["grid_df_data"]

    df_edit = st.data_editor(
        initial_df,
        key="grid_editor",
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,  # masque l’index 0..9
        column_config={
            "Allure (mm:ss/km)": st.column_config.Column(disabled=True),
            "Allure (min/km)": st.column_config.Column(disabled=True),
            "%VMA": st.column_config.Column(disabled=True),
            "Vitesse (km/h)": st.column_config.Column(disabled=True),
            "Palier": st.column_config.Column(disabled=True),
        }
    )
    # Stockage du DF édité en session (immédiat et nettoyé)
    st.session_state["grid_df_data"] = sanitize_df(df_edit)

    # Métadonnées séance
    st.markdown("#### Métadonnées de la séance")
    colm1, colm2, colm3 = st.columns([1, 1, 2])
    with colm1:
        athlete = st.text_input("Athlète", value=st.session_state.get("athlete", "Anonyme"))
    with colm2:
        date_s = st.date_input("Date").isoformat()
    with colm3:
        note = st.text_input("Notes (surface, météo, protocole)", value=st.session_state.get("note", ""))

    st.session_state["athlete"] = athlete
    st.session_state["date"] = date_s
    st.session_state["note"] = note

    # Historique (mémoire de session)
    if st.button("➕ Ajouter cette séance à l'historique"):
        hist = st.session_state.get("historique", [])
        record = {
            "athlete": athlete, "date": date_s, "note": note,
            "vma": float(vma), "bsn": float(bsn), "n": int(n),
            "pct_start": float(pct_start), "pct_end": float(pct_end),
            "df": st.session_state["grid_df_data"].copy()
        }
        hist.append(record)
        st.session_state["historique"] = hist
        st.success("Séance ajoutée à l’historique (mémoire de session).")

    # Export CSV séance courante
    out_df = st.session_state["grid_df_data"].copy()
    out_df["log10(lactate)"] = np.where(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce") > 0,
                                        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")),
                                        np.nan)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button(
        "💾 Télécharger la séance (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"seance_{athlete}_{date_s}.csv",
        mime="text/csv"
    )

# ----------------------------- Calculs (cache) -----------------------------
@st.cache_data(show_spinner=False)
def compute_all(df_edit, bsn, use_poly, poly_order):
    x = pd.to_numeric(df_edit["Vitesse (km/h)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_edit["Lactate (mmol/L)"], errors="coerce").to_numpy()
    sl1_level = bsn + 0.5
    sl1_speed = interp_speed_at_lactate(x, y, sl1_level)
    lt1_speed, lt1_model = lt1_log_lactate_breakpoint(x, y)
    lt2_dmax_speed, dmax_meta = dmax_lt2(x, y)
    lt2_dmax_poly = np.nan; poly_meta = None
    if use_poly:
        xv = x[~np.isnan(x) & ~np.isnan(y)]
        yv = y[~np.isnan(x) & ~np.isnan(y)]
        if len(xv) >= poly_order + 1:
            xgrid = np.linspace(np.nanmin(xv), np.nanmax(xv), 200)
            coef = np.polyfit(xv, yv, poly_order)
            ygrid = np.polyval(coef, xgrid)
            lt2_dmax_poly, poly_meta = dmax_lt2(xgrid, ygrid)
    x_sl1_for_line = sl1_speed if not np.isnan(sl1_speed) else np.nan
    lt2_moddmax_speed, moddmax_meta = moddmax_lt2(x, y, x_sl1_for_line, sl1_level if not np.isnan(x_sl1_for_line) else np.nan)
    ylog = np.where(y > 0, np.log10(y), np.nan)
    return {
        "x": x, "y": y, "ylog": ylog,
        "sl1_level": sl1_level, "sl1_speed": sl1_speed,
        "lt1_speed": lt1_speed, "lt1_model": lt1_model,
        "lt2_dmax_speed": lt2_dmax_speed, "dmax_meta": dmax_meta,
        "lt2_dmax_poly": lt2_dmax_poly, "poly_meta": poly_meta,
        "lt2_moddmax_speed": lt2_moddmax_speed, "moddmax_meta": moddmax_meta
    }

calc = compute_all(st.session_state["grid_df_data"], bsn, use_poly, poly_order)

# ----------------------------- Onglet 2: Résultats -----------------------------
with tab2:
    st.markdown("### Seuils (vitesses estimées)")
    method_lt2 = st.radio("Méthode LT2", ["D-max (points)", "D-max polynomial", "ModDmax"], horizontal=True)
    if method_lt2 == "D-max (points)":
        lt2_selected = calc["lt2_dmax_speed"]
    elif method_lt2 == "D-max polynomial":
        lt2_selected = calc["lt2_dmax_poly"]
    else:
        lt2_selected = calc["lt2_moddmax_speed"]

    mlss_speed = lt2_selected * 0.9 if lt2_selected == lt2_selected else np.nan

    colA, colB = st.columns([1, 1])
    with colA:
        st.write({
            "LT1 (log-lactate)": None if np.isnan(calc["lt1_speed"]) else round(calc["lt1_speed"], 2),
            "SL1 (Bsn+0,5)": None if np.isnan(calc["sl1_speed"]) else round(calc["sl1_speed"], 2),
            "LT2 (D-max points)": None if np.isnan(calc["lt2_dmax_speed"]) else round(calc["lt2_dmax_speed"], 2),
            "LT2 (D-max poly)": None if np.isnan(calc["lt2_dmax_poly"]) else round(calc["lt2_dmax_poly"], 2),
            "LT2 (ModDmax)": None if np.isnan(calc["lt2_moddmax_speed"]) else round(calc["lt2_moddmax_speed"], 2),
            "MLSS (90% LT2 choisi)": None if np.isnan(mlss_speed) else round(mlss_speed, 2)
        })
        if np.sum(~np.isnan(calc["y"]) & (calc["y"] > 0)) < 4:
            st.warning("💡 Saisis ≥ 4 lactates > 0 mmol/L pour fiabiliser LT1 (log-lactate).")
        if np.isfinite(np.nanmax(calc["y"])) and (np.nanmax(calc["y"]) < (bsn + 0.5)):
            st.info("ℹ️ La courbe n’atteint pas Bsn+0,5 → SL1/ModDmax non déterminables.")
    with colB:
        st.markdown("#### Courbe lactate – vitesse")
        x, y = calc["x"], calc["y"]
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(x, y, "o-", label="Mesures", color="#1f77b4")
        for i, (xi, yi) in enumerate(zip(x, y)):
            if not np.isnan(xi) and not np.isnan(yi):
                ax1.annotate(str(i+1), (xi, yi), textcoords="offset points", xytext=(0, 6),
                             ha='center', fontsize=8, color="#1f77b4")
        def vline(val, label, color):
            if val and not np.isnan(val):
                ax1.axvline(val, color=color, linestyle="--", alpha=0.8, label=label)
        vline(calc["lt1_speed"], "LT1 (log)", "#2ca02c")
        vline(calc["sl1_speed"], "SL1 (Bsn+0,5)", "#17becf")
        vline(calc["lt2_dmax_speed"], "LT2 (D-max)", "#d62728")
        if use_poly:
            vline(calc["lt2_dmax_poly"], "LT2 (D-max poly)", "#9467bd")
        vline(calc["lt2_moddmax_speed"], "LT2 (ModDmax)", "#8c564b")
        ymax = (np.nanmax(y)*1.15) if np.isfinite(np.nanmax(y)) and np.nanmax(y) > 0 else 5
        ax1.set_xlabel("Vitesse (km/h)")
        ax1.set_ylabel("Lactate (mmol/L)")
        ax1.set_ylim(0, ymax)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        st.pyplot(fig1)

        st.markdown("#### Log-lactate – vitesse (rupture LT1)")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        mask_pos = ~np.isnan(calc["y"]) & (calc["y"] > 0)
        xv = calc["x"][mask_pos]; yv = calc["y"][mask_pos]
        if len(xv) >= 2:
            ax2.plot(xv, np.log10(yv), "o-", label="log10(lactate)", color="#1f77b4")
            if calc["lt1_model"] is not None and calc["lt1_model"]["idx"] is not None:
                b = calc["lt1_model"]["idx"]
                x1 = xv[:b+1]; y1_fit = calc["lt1_model"]["m1"]*x1 + calc["lt1_model"]["b1"]
                x2 = xv[b:];   y2_fit = calc["lt1_model"]["m2"]*x2 + calc["lt1_model"]["b2"]
                ax2.plot(x1, y1_fit, "-", color="#2ca02c", label="Fit segment 1")
                ax2.plot(x2, y2_fit, "-", color="#d62728", label="Fit segment 2")
                ax2.axvline(calc["lt1_speed"], color="#2ca02c", linestyle="--", alpha=0.8, label="Rupture (LT1)")
            ax2.set_xlabel("Vitesse (km/h)"); ax2.set_ylabel("log10(lactate)")
            ax2.grid(True, alpha=0.3); ax2.legend()
        else:
            ax2.text(0.02, 0.6, "Saisir au moins 2 mesures > 0 mmol/L", transform=ax2.transAxes)
        st.pyplot(fig2)

    # Export HTML (avec balises data:image/png;base64, ...)
    img1_b64 = fig_to_base64(fig1)
    img2_b64 = fig_to_base64(fig2)

    synth = {
        "LT1_log": calc["lt1_speed"],
        "SL1_Bsn+0.5": calc["sl1_speed"],
        "LT2_Dmax": calc["lt2_dmax_speed"],
        "LT2_Dmax_poly": calc["lt2_dmax_poly"] if use_poly else np.nan,
        "LT2_ModDmax": calc["lt2_moddmax_speed"],
        "MLSS_90pct": mlss_speed
    }

    df_display = st.session_state["grid_df_data"].copy()
    # Retrait robuste de "FC estimée" côté export
    for col in list(df_display.columns):
        if str(col).strip().lower() in ("fc estimée (bpm)", "fc estimee (bpm)", "fc estimée", "fc estimee"):
            df_display.drop(columns=[col], inplace=True, errors="ignore")
    df_display["log10(lactate)"] = np.where(pd.to_numeric(df_display["Lactate (mmol/L)"], errors="coerce") > 0,
                                            np.log10(pd.to_numeric(df_display["Lactate (mmol/L)"], errors="coerce")),
                                            np.nan)
    table_html = df_display.to_html(index=False, escape=False)

    # Sections MLSS & SRS pour le rapport
    mlss_params = st.session_state.get("mlss_params", None)
    mlss_img_b64 = st.session_state.get("mlss_img_b64", None)
    df_mlss_lac = sanitize_mlss(st.session_state.get("df_mlss_lac", pd.DataFrame()))
    mlss_table_html = df_mlss_lac.to_html(index=False) if df_mlss_lac is not None and not df_mlss_lac.empty else ""

    mlss_html = ""
    if mlss_params is not None:
        sv2p = mlss_params.get("sv2", None)
        delta = mlss_params.get("delta", None)
        vtheo = mlss_params.get("v_theo", None)
        img_tag = (f'data:image/png;base64,{mlss_img_b64}'
                   if mlss_img_b64 else '<p><i>(Ajoute ≥2 valeurs de lactate pour afficher la courbe.)</i></p>')
        mlss_html = f"""
        <h2 class="section">MLSS</h2>
        <ul>
          <li><b>Vitesse au SV2</b> : {sv2p if sv2p else "—"} km/h</li>
          <li><b>Delta vs SV2</b> : {delta if delta is not None else "—"} km/h</li>
          <li><b>Vitesse théorique MLSS</b> : {vtheo if vtheo else "—"} km/h</li>
        </ul>
        {mlss_table_html}
        {img_tag}
        """

    srs = st.session_state.get("srs_results", None)
    srs_img_b64 = st.session_state.get("srs_img_b64", None)
    srs_html = ""
    if srs is not None:
        slope_v = srs.get("slope", None)
        mrt1 = srs.get("mrt1", None)
        mrt2 = srs.get("mrt2", None)
        mrtu = srs.get("mrt_used", None)
        sv1c = srs.get("sv1_corr", None)
        sv2c = srs.get("sv2_corr", None)
        img_tag = (f'data:image/png;base64,{srs_img_b64}'
                   if srs_img_b64 else '<p><i>(Renseigne la pente, SV1/SV2 et vitesses équivalentes pour obtenir la correction.)</i></p>')
        srs_html = f"""
        <h2 class="section">Step–Ramp–Step (SRS)</h2>
        <ul>
          <li><b>Pente de la rampe</b> : {slope_v if slope_v else "—"} km/h/min</li>
          <li><b>MRT Step 1</b> : {round(mrt1,1) if mrt1 else "—"} s ;
              <b>MRT Step 2</b> : {round(mrt2,1) if mrt2 else "—"} s ;
              <b>MRT utilisé</b> : {round(mrtu,1) if mrtu else "—"} s</li>
          <li><b>SV1 corrigée</b> : {round(sv1c,2) if sv1c else "—"} km/h ;
              <b>SV2 corrigée</b> : {round(sv2c,2) if sv2c else "—"} km/h</li>
        </ul>
        {img_tag}
        """

    athlete = st.session_state.get("athlete", "Anonyme")
    date_s = st.session_state.get("date", "")
    note = st.session_state.get("note", "")

    html = f"""
    <html><head><meta charset="utf-8">
    <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1,h2,h3 {{ margin: 0.2em 0; }}
    .grid img {{ max-width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: center; }}
    .meta {{ color:#555; }}
    .section {{ margin-top: 18px; }}
    </style></head><body>
    <h1>Rapport – Test lactate</h1>
    <p class="meta"><b>Athlète:</b> {athlete} &nbsp; <b>Date:</b> {date_s} &nbsp;
    <b>VMA:</b> {vma:.2f} km/h &nbsp; <b>Bsn:</b> {bsn:.2f} mmol/L</p>
    <p class="meta"><b>Notes:</b> {note}</p>

    <h2>Seuils</h2>
    <ul>
      <li>LT1 (log-lactate): {'' if np.isnan(synth['LT1_log']) else f'{synth["LT1_log"]:.2f} km/h'}</li>
      <li>SL1 (Bsn+0,5): {'' if np.isnan(synth['SL1_Bsn+0.5']) else f'{synth["SL1_Bsn+0.5"]:.2f} km/h'}</li>
      <li>LT2 (D-max points): {'' if np.isnan(synth['LT2_Dmax']) else f'{synth["LT2_Dmax"]:.2f} km/h'}</li>
      <li>LT2 (D-max poly): {'' if np.isnan(synth['LT2_Dmax_poly']) else f'{synth["LT2_Dmax_poly"]:.2f} km/h'}</li>
      <li>LT2 (ModDmax): {'' if np.isnan(synth['LT2_ModDmax']) else f'{synth["LT2_ModDmax"]:.2f} km/h'}</li>
      <li>MLSS (90% du LT2 choisi): {'' if np.isnan(synth['MLSS_90pct']) else f'{synth["MLSS_90pct"]:.2f} km/h'}</li>
    </ul>

    <h2>Données</h2>
    {table_html}

    <h2>Graphiques</h2>
    <div class="grid section">
      <h3>Courbe lactate – vitesse</h3>
      data:image/png;base64,{img1_b64}
      <h3>Log-lactate – vitesse</h3>
      <img src="data:image/pngmg2_b64}
    </div>

    {mlss_html}
    {srs_html}

    <p style="margin-top:24px;font-size:12px;color:#666;">Généré par l’app Streamlit (version {VERSION}).</p>
    </body></html>
    """

    st.markdown("#### Export “rapport séance” (HTML)")
    st.download_button(
        "🧾 Télécharger le rapport (HTML)",
        data=html.encode("utf-8"),
        file_name=f"rapport_{athlete}.html",
        mime="text/html"
    )

# ----------------------------- Onglet 3: MLSS -----------------------------
with tab3:
    st.markdown("### MLSS – Paramètres et saisie")
    c1, c2, c3 = st.columns(3)
    with c1:
        sv2_mlss = st.number_input("Vitesse au SV2 (km/h)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    with c2:
        delta_mlss = st.number_input("Delta vs SV2 pour MLSS (km/h)", value=-0.6, step=0.1, format="%.1f")
    with c3:
        v_theo_mlss = (sv2_mlss + delta_mlss) if sv2_mlss > 0 else None
        st.metric("Vitesse théorique MLSS (km/h)", f"{v_theo_mlss:.1f}" if v_theo_mlss else "—")

    default_times = [0, 5, 10, 15, 20, 25, 30]
    if "df_mlss_lac" not in st.session_state:
        st.session_state.df_mlss_lac = pd.DataFrame({
            "Temps (min)": default_times,
            "Lactate (mmol/L)": [None]*len(default_times),
            "FC (bpm)": [None]*len(default_times),
            "Commentaires": ["" for _ in default_times],
        })

    # Editor MLSS (masque index)
    df_mlss_lac = st.data_editor(
        st.session_state.df_mlss_lac,
        hide_index=True,
        use_container_width=True
    )
    # Nettoyage + tri par temps
    st.session_state.df_mlss_lac = sanitize_mlss(df_mlss_lac)

    # Graphe MLSS : tri & numeric
    mlss_img_b64 = None
    plot_df = sanitize_mlss(st.session_state.df_mlss_lac)
    if not plot_df.empty and plot_df["Lactate (mmol/L)"].notna().sum() >= 2:
        fig_mlss, ax_mlss = plt.subplots(figsize=(6, 3.5))
        ax_mlss.plot(plot_df["Temps (min)"], plot_df["Lactate (mmol/L)"], marker="o", color="#0078d4")
        ax_mlss.set_title("MLSS – évolution du lactate")
        ax_mlss.set_xlabel("Temps (min)"); ax_mlss.set_ylabel("Lactate (mmol/L)")
        ax_mlss.grid(True, alpha=0.3)
        st.pyplot(fig_mlss)
        mlss_img_b64 = fig_to_base64(fig_mlss, dpi=150)
    else:
        st.info("Ajoute au moins deux valeurs de lactate pour afficher la courbe.")

    st.session_state.mlss_params = {"sv2": sv2_mlss if sv2_mlss > 0 else None,
                                    "delta": delta_mlss,
                                    "v_theo": v_theo_mlss}
    st.session_state.mlss_img_b64 = mlss_img_b64

# ----------------------------- Onglet 4: SRS -----------------------------
with tab4:
    st.markdown("### SRS – Paramétrage et corrections MRT")
    c1, c2, c3 = st.columns(3)
    with c1:
        slope = st.number_input("Pente de la rampe (km/h par min)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
        sv1_srs = st.number_input("SV1 mesuré (km/h)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
        sv2_srs = st.number_input("SV2 mesuré (km/h)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    with c2:
        delta_step2 = st.number_input("Correction SV2 → Step 2 (delta km/h, ex. -0,8)", value=-0.8, step=0.1, format="%.1f")
        step1 = st.number_input("Vitesse Step 1 (km/h)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
        vo2_1 = st.number_input("VO₂ Step 1 (ml·kg⁻¹·min⁻¹)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    with c3:
        v_equiv1 = st.number_input("Vitesse équivalente VO₂ Step 1 (rampe) (km/h)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
        vo2_2 = st.number_input("VO₂ Step 2 (ml·kg⁻¹·min⁻¹) (optionnel)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
        v_equiv2 = st.number_input("Vitesse équivalente VO₂ Step 2 (rampe) (km/h)", min_value=0.0, step=0.1, value=0.0, format="%.1f")

    step2 = sv2_srs + delta_step2 if sv2_srs > 0 else None
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
        if sv1_srs and sv1_srs > 0:
            sv1_corr = sv1_srs - alpha_s * mrt_used
        if sv2_srs and sv2_srs > 0:
            sv2_corr = sv2_srs - alpha_s * mrt_used

    cv1, cv2 = st.columns(2)
    cv1.metric("SV1 corrigée (km/h)", f"{sv1_corr:.2f}" if sv1_corr else "—")
    cv2.metric("SV2 corrigée (km/h)", f"{sv2_corr:.2f}" if sv2_corr else "—")

    srs_img_b64 = None
    labels, raw_vals, corr_vals = [], [], []
    if sv1_srs and sv1_srs > 0:
        labels.append("SV1"); raw_vals.append(sv1_srs); corr_vals.append(sv1_corr if sv1_corr is not None else np.nan)
    if sv2_srs and sv2_srs > 0:
        labels.append("SV2"); raw_vals.append(sv2_srs); corr_vals.append(sv2_corr if sv2_corr is not None else np.nan)
    if labels and (np.isfinite(corr_vals).any()):
        x_ = np.arange(len(labels)); w = 0.35
        fig_srs, ax_srs = plt.subplots(figsize=(6, 3.5))
        ax_srs.bar(x_ - w/2, raw_vals, width=w, label="Mesuré", color="#767676")
        ax_srs.bar(x_ + w/2, corr_vals, width=w, label="Corrigé", color="#107c10")
        ax_srs.set_xticks(x_, labels); ax_srs.set_ylabel("Vitesse (km/h)")
        ax_srs.set_title("Correction des vitesses SV1 / SV2 par MRT")
        ax_srs.legend(); ax_srs.grid(axis="y", alpha=0.3)
        st.pyplot(fig_srs)
        srs_img_b64 = fig_to_base64(fig_srs, dpi=150)
    else:
        st.info("Renseigne la pente, SV1/SV2 et vitesses équivalentes (rampe) pour afficher la comparaison.")

    st.session_state.srs_results = {
        "slope": slope if slope > 0 else None,
        "sv1": sv1_srs if sv1_srs > 0 else None,
        "sv2": sv2_srs if sv2_srs > 0 else None,
        "step1": step1 if step1 > 0 else None,
        "step2": step2 if step2 else None,
        "vo2_1": vo2_1 if vo2_1 > 0 else None,
        "vo2_2": vo2_2 if vo2_2 > 0 else None,
        "v_equiv1": v_equiv1 if v_equiv1 > 0 else None,
        "v_equiv2": v_equiv2 if v_equiv2 > 0 else None,
        "mrt1": mrt1, "mrt2": mrt2, "mrt_used": mrt_used,
        "sv1_corr": sv1_corr, "sv2_corr": sv2_corr,
    }
    st.session_state.srs_img_b64 = srs_img_b64

# ----------------------------- Onglet 5: Historique & Comparaison -----------------------------
with tab5:
    st.markdown("### Historique local (session)")
    hist = st.session_state.get("historique", [])
    st.write(f"Nombre de séances en mémoire : **{len(hist)}**")

    uploaded = st.file_uploader("Importer une séance (CSV)", type=["csv"])
    if uploaded:
        try:
            tmp = pd.read_csv(uploaded)
            req = {"Palier", "%VMA", "Vitesse (km/h)", "Lactate (mmol/L)"}
            if not req.issubset(tmp.columns):
                st.error("Colonnes manquantes dans le CSV importé.")
            else:
                st.session_state["grid_df_data"] = sanitize_df(tmp)
                st.success("Séance importée dans la grille de saisie (onglet Saisie).")
        except Exception as e:
            st.error(f"Erreur import CSV : {e}")

    if hist:
        concat_rows = []
        for rec in hist:
            dfh = sanitize_df(rec["df"].copy())
            dfh["athlete"] = rec["athlete"]; dfh["date"] = rec["date"]
            dfh["vma"] = rec["vma"]; dfh["bsn"] = rec["bsn"]
            concat_rows.append(dfh)
        big = pd.concat(concat_rows, ignore_index=True)
        buf_hist = io.StringIO(); big.to_csv(buf_hist, index=False)
        st.download_button(
            "📚 Télécharger l'historique (CSV concaténé)",
            data=buf_hist.getvalue(),
            file_name="historique_tests_lactate.csv",
            mime="text/csv"
        )
    st.markdown("---")
    st.markdown("### Comparaison de courbes (sélectionne 2 séances)")
    if len(hist) >= 2:
        labels = [f"{i+1} – {rec['athlete']} – {rec['date']}" for i, rec in enumerate(hist)]
        idx1 = st.selectbox("Séance A", options=list(range(len(hist))), format_func=lambda i: labels[i], key="cmpA")
        idx2 = st.selectbox("Séance B", options=list(range(len(hist))), format_func=lambda i: labels[i], key="cmpB")
        if idx1 == idx2:
            st.info("Choisis deux séances **différentes**.")
        else:
            recA, recB = hist[idx1], hist[idx2]
            def xy_from_df(df):
                d = sanitize_df(df)
                return pd.to_numeric(d["Vitesse (km/h)"], errors="coerce").to_numpy(), \
                       pd.to_numeric(d["Lactate (mmol/L)"], errors="coerce").to_numpy()
            xA, yA = xy_from_df(recA["df"]); xB, yB = xy_from_df(recB["df"])
            figC, axC = plt.subplots(figsize=(7, 4))
            axC.plot(xA, yA, "o-", label=f"A: {recA['athlete']} {recA['date']}", color="#1f77b4")
            axC.plot(xB, yB, "s--", label=f"B: {recB['athlete']} {recB['date']}", color="#ff7f0e")
            axC.set_xlabel("Vitesse (km/h)"); axC.set_ylabel("Lactate (mmol/L)")
            axC.grid(True, alpha=0.3); axC.legend()
            st.pyplot(figC)
    else:
        st.info("Ajoute au moins **2 séances** à l’historique pour comparer.")

# ----------------------------- Footer -----------------------------
st.caption("Astuce : pour un PDF, ouvre le rapport HTML puis Fichier → Imprimer → “Enregistrer au format PDF”.")