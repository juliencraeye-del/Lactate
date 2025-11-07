# -*- coding: utf-8 -*-
# Seuils Lactate – VMA (avec annotations, export rapport, historique & comparaison, D-max polynomial, UX & caching)
import io, base64, math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

VERSION = "0.4.0"

st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# ------------------------- Helpers génériques -------------------------
def pace_min_per_km(speed_kmh: float):
    if speed_kmh and speed_kmh > 0:
        return 60.0 / speed_kmh
    return np.nan

def pace_mmss(speed_kmh: float):
    p = pace_min_per_km(speed_kmh)
    if np.isnan(p):
        return ""
    m = int(p)
    s = int(round((p - m) * 60))
    if s == 60:
        m += 1; s = 0
    return f"{m:02d}:{s:02d}"

def fc_estimee_from_vma_pct(pct, fc_rest, fc_max):
    if any([fc_rest is None, fc_max is None]): return np.nan
    if np.isnan(fc_rest) or np.isnan(fc_max) or fc_rest <= 0 or fc_max <= 0 or fc_max <= fc_rest:
        return np.nan
    return float(fc_rest + (fc_max - fc_rest) * pct / 100.0)

def linear_regression(x, y):
    # retourne slope, intercept, y_pred, SSE
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
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - y2 * x1
    denom = np.sqrt(A*A + B*B)
    if denom == 0 or len(xs) == 0:
        return np.array([]), np.array([])
    d = np.abs(A*xs + B*ys + C) / denom
    return d, mask

def dmax_lt2(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3: return np.nan, None
    d, _ = point_to_line_distances(x, y, x[0], y[0], x[-1], y[-1])
    if d.size == 0: return np.nan, None
    idx = int(np.nanargmax(d))
    return float(x[idx]), {"distances": d, "idx": idx, "x": x, "y": y}

def moddmax_lt2(x, y, x_start, y_start):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.isnan(x_start) or np.isnan(y_start):
        return np.nan, None
    d, _ = point_to_line_distances(x, y, x_start, y_start, x[-1], y[-1])
    if d.size == 0: return np.nan, None
    idx = int(np.nanargmax(d))
    return float(x[idx]), {"distances": d, "idx": idx, "x": x, "y": y, "x_start": x_start, "y_start": y_start}

def lt1_log_lactate_breakpoint(x, y_lac):
    mask = ~np.isnan(x) & ~np.isnan(y_lac) & (y_lac > 0)
    x = np.array(x)[mask]; y = np.array(y_lac)[mask]
    if len(x) < 4:  # min 4 pts
        return np.nan, None
    ylog = np.log10(y)

    best = {"idx": None, "sse": np.inf, "m1": np.nan, "b1": np.nan, "m2": np.nan, "b2": np.nan}
    for b in range(1, len(x) - 2 + 1):  # rupture entre points
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

def zones_from_thresholds(lt1, lt2):
    if np.isnan(lt1) or np.isnan(lt2): return {}
    return {"Zone 1 (≤LT1)": (0, lt1), "Zone 2 (LT1–LT2)": (lt1, lt2), "Zone 3 (≥LT2)": (lt2, None)}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ------------------------- Sidebar (paramètres) -------------------------
st.sidebar.header("Paramètres du test")
vma = st.sidebar.number_input("VMA (km/h)", min_value=5.0, max_value=30.0, value=17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", min_value=6, max_value=20, value=10, step=1)
pct_start = st.sidebar.number_input("%VMA palier de départ", min_value=40.0, max_value=80.0, value=60.0, step=1.0)
pct_end   = st.sidebar.number_input("%VMA palier final",   min_value=80.0, max_value=120.0, value=105.0, step=1.0)
duree_min = st.sidebar.number_input("Durée d'un palier (min) – informatif", min_value=2.0, max_value=6.0, value=3.0, step=0.5)
fc_rest = st.sidebar.number_input("FC repos (bpm) – optionnel", min_value=0, max_value=120, value=0, step=1)
fc_max  = st.sidebar.number_input("FC max (bpm) – optionnel",   min_value=0, max_value=240, value=0, step=1)

# D-max polynomial
st.sidebar.markdown("---")
use_poly = st.sidebar.toggle("Lisser la courbe (D‑max polynomial)", value=False)
poly_order = st.sidebar.select_slider("Ordre du polynôme", options=[2,3], value=2, disabled=not use_poly)

# Reset
if st.sidebar.button("🔄 Réinitialiser la séance"):
    for key in ["grid_df", "historique", "plots"]:
        if key in st.session_state: del st.session_state[key]
    st.rerun()

st.sidebar.caption(f"Version {VERSION}")

# ------------------------- Construction des paliers -------------------------
pcts = np.linspace(pct_start, pct_end, int(n))
speeds = vma * pcts / 100.0
allure_min = np.array([pace_min_per_km(s) for s in speeds])
allure_txt = np.array([pace_mmss(s) for s in speeds])
fc_est = np.array([fc_estimee_from_vma_pct(p, fc_rest if fc_rest>0 else np.nan, fc_max if fc_max>0 else np.nan) for p in pcts])

base_df = pd.DataFrame({
    "Palier": np.arange(1, int(n)+1, dtype=int),
    "%VMA": np.round(pcts, 2),
    "Vitesse (km/h)": np.round(speeds, 2),
    "Allure (min/km)": np.round(allure_min, 2),
    "Allure (mm:ss/km)": allure_txt,
    "FC estimée (bpm)": np.where(np.isnan(fc_est), None, np.round(fc_est).astype("float")),
    "FC mesurée (bpm)": [None]*int(n),
    "Lactate (mmol/L)": [None]*int(n)
})

# ------------------------- Tabs UI -------------------------
tab1, tab2, tab3 = st.tabs(["📝 Saisie", "📊 Résultats", "🗂️ Historique & Comparaison"])

with tab1:
    st.markdown("### Saisie des lactates (et FC mesurée si dispo)")
    df_edit = st.data_editor(
        st.session_state.get("grid_df", base_df),
        key="grid_df",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Allure (mm:ss/km)": st.column_config.Column(disabled=True),
            "Allure (min/km)": st.column_config.Column(disabled=True),
            "%VMA": st.column_config.Column(disabled=True),
            "Vitesse (km/h)": st.column_config.Column(disabled=True),
            "FC estimée (bpm)": st.column_config.Column(disabled=True),
            "Palier": st.column_config.Column(disabled=True),
        }
    )
    # Métadonnées séance
    st.markdown("#### Métadonnées de la séance")
    colm1, colm2, colm3 = st.columns([1,1,2])
    with colm1:
        athlete = st.text_input("Athlète", value=st.session_state.get("athlete", "Anonyme"))
    with colm2:
        date_s = st.date_input("Date").isoformat()
    with colm3:
        note = st.text_input("Notes (surface, météo, protocole)", value=st.session_state.get("note", ""))

    # Enregistrer dans l'historique local (session)
    if st.button("➕ Ajouter cette séance à l'historique"):
        hist = st.session_state.get("historique", [])
        # stocke datas + meta
        record = {
            "athlete": athlete, "date": date_s, "note": note,
            "vma": float(vma), "bsn": float(bsn), "n": int(n),
            "pct_start": float(pct_start), "pct_end": float(pct_end),
            "df": df_edit.copy()
        }
        hist.append(record)
        st.session_state["historique"] = hist
        st.success("Séance ajoutée à l’historique (mémoire de session).")
    # Export CSV de la séance courante
    out_df = df_edit.copy()
    out_df["log10(lactate)"] = np.where(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")>0,
                                        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")),
                                        np.nan)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("💾 Télécharger la séance (CSV)", data=csv_buf.getvalue(),
                       file_name=f"seance_{athlete}_{date_s}.csv", mime="text/csv")

with tab3:
    st.markdown("### Historique local (session)")
    hist = st.session_state.get("historique", [])
    st.write(f"Nombre de séances en mémoire : **{len(hist)}**")
    # Import CSV dans la grille (pour recharger une séance)
    uploaded = st.file_uploader("Importer une séance (CSV)", type=["csv"])
    if uploaded:
        try:
            tmp = pd.read_csv(uploaded)
            # contrôle colonnes minimales
            required_cols = {"Palier","%VMA","Vitesse (km/h)","Lactate (mmol/L)"}
            if not required_cols.issubset(tmp.columns):
                st.error("Colonnes manquantes dans le CSV importé.")
            else:
                st.session_state["grid_df"] = tmp
                st.success("Séance importée dans la grille de saisie (onglet Saisie).")
        except Exception as e:
            st.error(f"Erreur import CSV : {e}")

    # Export global de l'historique (zip CSVs – ici: concat simple)
    if hist:
        concat_rows = []
        for rec in hist:
            dfh = rec["df"].copy()
            dfh["athlete"] = rec["athlete"]; dfh["date"] = rec["date"]
            dfh["vma"] = rec["vma"]; dfh["bsn"] = rec["bsn"]
            concat_rows.append(dfh)
        big = pd.concat(concat_rows, ignore_index=True)
        buf_hist = io.StringIO(); big.to_csv(buf_hist, index=False)
        st.download_button("📚 Télécharger l'historique (CSV concaténé)", data=buf_hist.getvalue(),
                           file_name="historique_tests_lactate.csv", mime="text/csv")

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
                return pd.to_numeric(df["Vitesse (km/h)"], errors="coerce").to_numpy(), \
                       pd.to_numeric(df["Lactate (mmol/L)"], errors="coerce").to_numpy()

            xA, yA = xy_from_df(recA["df"])
            xB, yB = xy_from_df(recB["df"])

            figC, axC = plt.subplots(figsize=(7,4))
            axC.plot(xA, yA, "o-", label=f"A: {recA['athlete']} {recA['date']}", color="#1f77b4")
            axC.plot(xB, yB, "s--", label=f"B: {recB['athlete']} {recB['date']}", color="#ff7f0e")
            axC.set_xlabel("Vitesse (km/h)"); axC.set_ylabel("Lactate (mmol/L)"); axC.grid(True, alpha=0.3); axC.legend()
            st.pyplot(figC)
    else:
        st.info("Ajoute au moins **2 séances** à l’historique pour comparer.")

# ------------------------- Calculs (cache) -------------------------
@st.cache_data(show_spinner=False)
def compute_all(df_edit, bsn, use_poly, poly_order):
    x = pd.to_numeric(df_edit["Vitesse (km/h)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_edit["Lactate (mmol/L)"], errors="coerce").to_numpy()
    # seuil SL1
    sl1_level = bsn + 0.5
    sl1_speed = interp_speed_at_lactate(x, y, sl1_level)
    # LT1
    lt1_speed, lt1_model = lt1_log_lactate_breakpoint(x, y)
    # Dmax brut
    lt2_dmax_speed, dmax_meta = dmax_lt2(x, y)
    # Dmax polynomial (option)
    lt2_dmax_poly = np.nan; poly_meta = None
    if use_poly:
        xv = x[~np.isnan(x) & ~np.isnan(y)]
        yv = y[~np.isnan(x) & ~np.isnan(y)]
        if len(xv) >= poly_order + 1:
            # grille lissée
            xgrid = np.linspace(np.nanmin(xv), np.nanmax(xv), 200)
            coef = np.polyfit(xv, yv, poly_order)
            ygrid = np.polyval(coef, xgrid)
            lt2_dmax_poly, poly_meta = dmax_lt2(xgrid, ygrid)
    # ModDmax (ligne depuis SL1 -> dernier point)
    x_sl1_for_line = sl1_speed if not np.isnan(sl1_speed) else np.nan
    lt2_moddmax_speed, moddmax_meta = moddmax_lt2(x, y, x_sl1_for_line, sl1_level if not np.isnan(x_sl1_for_line) else np.nan)
    # y log pour graph
    ylog = np.where(y>0, np.log10(y), np.nan)
    return {
        "x": x, "y": y, "ylog": ylog,
        "sl1_level": sl1_level, "sl1_speed": sl1_speed,
        "lt1_speed": lt1_speed, "lt1_model": lt1_model,
        "lt2_dmax_speed": lt2_dmax_speed, "dmax_meta": dmax_meta,
        "lt2_dmax_poly": lt2_dmax_poly, "poly_meta": poly_meta,
        "lt2_moddmax_speed": lt2_moddmax_speed, "moddmax_meta": moddmax_meta
    }

calc = compute_all(st.session_state["grid_df"], bsn, use_poly, poly_order)

# ------------------------- Résultats & Graphiques -------------------------
with tab2:
    st.markdown("### Seuils (vitesses estimées)")
    method_lt2 = st.radio("Méthode LT2", ["D-max (points)", "D-max polynomial", "ModDmax"], horizontal=True)
    lt2_selected = (
        calc["lt2_dmax_speed"] if method_lt2=="D-max (points)"
        else (calc["lt2_dmax_poly"] if method_lt2=="D-max polynomial" else calc["lt2_moddmax_speed"])
    )
    mlss_speed = lt2_selected * 0.9 if lt2_selected==lt2_selected else np.nan

    colA, colB = st.columns([1,1])
    with colA:
        st.write({
            "LT1 (log-lactate)": None if np.isnan(calc["lt1_speed"]) else round(calc["lt1_speed"], 2),
            "SL1 (Bsn+0,5)": None if np.isnan(calc["sl1_speed"]) else round(calc["sl1_speed"], 2),
            "LT2 (D-max points)": None if np.isnan(calc["lt2_dmax_speed"]) else round(calc["lt2_dmax_speed"], 2),
            "LT2 (D-max poly)": None if np.isnan(calc["lt2_dmax_poly"]) else round(calc["lt2_dmax_poly"], 2),
            "LT2 (ModDmax)": None if np.isnan(calc["lt2_moddmax_speed"]) else round(calc["lt2_moddmax_speed"], 2),
            "MLSS (90% LT2 choisi)": None if np.isnan(mlss_speed) else round(mlss_speed, 2)
        })
        if np.sum(~np.isnan(calc["y"]) & (calc["y"]>0)) < 4:
            st.warning("💡 Saisis ≥ 4 lactates > 0 mmol/L pour fiabiliser LT1 (log-lactate).")
        if np.nanmax(calc["y"]) < (bsn + 0.5):
            st.info("ℹ️ La courbe n’atteint pas Bsn+0,5 → SL1/ModDmax non déterminables.")

    # --- Courbe lactate (avec annotations paliers + traits verticaux) ---
    with colB:
        st.markdown("#### Courbe lactate – vitesse")
        x, y = calc["x"], calc["y"]
        fig1, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(x, y, "o-", label="Mesures", color="#1f77b4")
        # annotations palier
        for i, (xi, yi) in enumerate(zip(x, y)):
            if not np.isnan(xi) and not np.isnan(yi):
                ax1.annotate(str(i+1), (xi, yi), textcoords="offset points", xytext=(0,6),
                             ha='center', fontsize=8, color="#1f77b4")
        # traits verticaux
        def vline(val, label, color):
            if val and not np.isnan(val):
                ax1.axvline(val, color=color, linestyle="--", alpha=0.8, label=label)
        vline(calc["lt1_speed"], "LT1 (log)", "#2ca02c")
        vline(calc["sl1_speed"], "SL1 (Bsn+0,5)", "#17becf")
        vline(calc["lt2_dmax_speed"], "LT2 (D-max)", "#d62728")
        if use_poly:
            vline(calc["lt2_dmax_poly"], "LT2 (D-max poly)", "#9467bd")
        vline(calc["lt2_moddmax_speed"], "LT2 (ModDmax)", "#8c564b")

        ymax = (np.nanmax(y)*1.15) if np.isfinite(np.nanmax(y)) and np.nanmax(y)>0 else 5
        ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Lactate (mmol/L)")
        ax1.set_ylim(0, ymax)
        ax1.grid(True, alpha=0.3); ax1.legend()
        st.pyplot(fig1)

    # --- Log-lactate + segments LT1 ---
    st.markdown("#### Log-lactate – vitesse (rupture LT1)")
    fig2, ax2 = plt.subplots(figsize=(7,4))
    xv = x[~np.isnan(calc["y"]) & (calc["y"]>0)]
    yv = calc["y"][~np.isnan(calc["y"]) & (calc["y"]>0)]
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

    # --- Rapport séance (HTML) ---
    st.markdown("#### Export “rapport séance” (HTML)")
    # Encode figures
    img1_b64 = fig_to_base64(fig1)
    img2_b64 = fig_to_base64(fig2)

    # Synthèse seuils
    synth = {
        "LT1_log": calc["lt1_speed"],
        "SL1_Bsn+0.5": calc["sl1_speed"],
        "LT2_Dmax": calc["lt2_dmax_speed"],
        "LT2_Dmax_poly": calc["lt2_dmax_poly"] if use_poly else np.nan,
        "LT2_ModDmax": calc["lt2_moddmax_speed"],
        "MLSS_90pct": mlss_speed
    }

    # Table HTML
    df_display = st.session_state["grid_df"].copy()
    df_display["log10(lactate)"] = np.where(pd.to_numeric(df_display["Lactate (mmol/L)"], errors="coerce")>0,
                                            np.log10(pd.to_numeric(df_display["Lactate (mmol/L)"], errors="coerce")), np.nan)
    table_html = df_display.to_html(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, (int,float,np.floating)) and not math.isnan(x) else x)

    html = f"""
    <html><head><meta charset="utf-8">
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      h1,h2,h3 {{ margin: 0.2em 0; }}
      .grid img {{ max-width: 100%; height: auto; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
      th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: center; }}
    </style></head><body>
    <h1>Rapport – Test lactate</h1>
    <p><b>Athlète:</b> {st.session_state.get('athlete','Anonyme')} &nbsp; | &nbsp;
       <b>Date:</b> {st.session_state.get('date', '')} &nbsp; | &nbsp;
       <b>VMA:</b> {vma:.2f} km/h &nbsp; | &nbsp; <b>Bsn:</b> {bsn:.2f} mmol/L</p>
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
    <div class="grid">
      <h3>Courbe lactate – vitesse</h3>
      data:image/png;base64,{img1_b64}
      <h3>Log-lactate – vitesse</h3>
      data:image/png;base64,{img2_b64}
    </div>
    <p style="margin-top:24px;font-size:12px;color:#666;">Généré par l’app Streamlit (version {VERSION}).</p>
    </body></html>
    """
    st.download_button("🧾 Télécharger le rapport (HTML)", data=html.encode("utf-8"),
                       file_name=f"rapport_{st.session_state.get('athlete','Anonyme')}.html",
                       mime="text/html")

# ------------------------- Pied de page -------------------------
st.caption("Conseil : pour un PDF, ouvre le rapport HTML et fais Fichier → Imprimer → Enregistrer au format PDF.")