import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Seuils Lactate – VMA", layout="wide")

# --------------- Helpers ---------------

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
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"

def linear_regression(x, y):
    # retourne slope, intercept, y_pred et SSE
    if len(x) < 2:
        return np.nan, np.nan, np.full_like(x, np.nan, dtype=float), np.inf
    m, b = np.polyfit(x, y, 1)
    yhat = m * np.array(x) + b
    sse = float(((np.array(y) - yhat) ** 2).sum())
    return m, b, yhat, sse

def lt1_log_lactate_breakpoint(x, y_lac):
    # x: vitesses, y_lac: lactates; on travaille sur log10
    mask = ~np.isnan(x) & ~np.isnan(y_lac) & (y_lac > 0)
    x = np.array(x)[mask]
    y = np.array(y_lac)[mask]
    if len(x) < 4:  # minimum 4 points pour 2 segments fiables
        return np.nan, None

    ylog = np.log10(y)

    best = {"idx": None, "sse": np.inf, "m1": np.nan, "b1": np.nan, "m2": np.nan, "b2": np.nan}
    # breakpoint b = index fin du segment 1 (au moins 2 pts chacun)
    for b in range(1, len(x) - 2 + 1):
        x1, y1 = x[: b + 1], ylog[: b + 1]
        x2, y2 = x[b:], ylog[b:]
        if len(x1) < 2 or len(x2) < 2:
            continue
        m1, c1, yhat1, sse1 = linear_regression(x1, y1)
        m2, c2, yhat2, sse2 = linear_regression(x2, y2)
        sse = sse1 + sse2
        if sse < best["sse"]:
            best = {"idx": b, "sse": sse, "m1": m1, "b1": c1, "m2": m2, "b2": c2}

    if best["idx"] is None:
        return np.nan, None

    bp_speed = x[best["idx"]]  # vitesse au point de rupture
    return float(bp_speed), best

def interp_speed_at_lactate(x, y, target):
    # cherche première traversée y >= target puis interpole
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    for k in range(1, len(x)):
        if y[k-1] < target <= y[k]:
            # interpolation linéaire
            if y[k] == y[k-1]:
                return float(x[k])  # cas rare: palier plat sur le seuil
            return float(x[k-1] + (target - y[k-1]) * (x[k] - x[k-1]) / (y[k] - y[k-1]))
    return np.nan

def point_to_line_distances(xs, ys, x1, y1, x2, y2):
    # distances perpendiculaires des points (xs,ys) à la droite (x1,y1)-(x2,y2)
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
    # D‑max traditionnel: droite entre 1er et dernier point
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, None
    d, _ = point_to_line_distances(x, y, x[0], y[0], x[-1], y[-1])
    if d.size == 0: return np.nan, None
    idx = int(np.nanargmax(d))
    return float(x[idx]), {"distances": d, "idx": idx}

def moddmax_lt2(x, y, x_start, y_start):
    # ModDmax: droite de (x_start,y_start) vers le dernier point de la courbe
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.isnan(x_start) or np.isnan(y_start):
        return np.nan, None
    d, _ = point_to_line_distances(x, y, x_start, y_start, x[-1], y[-1])
    if d.size == 0: return np.nan, None
    idx = int(np.nanargmax(d))
    return float(x[idx]), {"distances": d, "idx": idx}

def fc_estimee_from_vma_pct(pct, fc_rest, fc_max):
    if np.isnan(fc_rest) or np.isnan(fc_max) or fc_rest <= 0 or fc_max <= 0 or fc_max <= fc_rest:
        return np.nan
    return float(fc_rest + (fc_max - fc_rest) * pct / 100.0)

# --------------- Sidebar (paramètres) ---------------
st.sidebar.header("Paramètres du test")
vma = st.sidebar.number_input("VMA (km/h)", min_value=5.0, max_value=30.0, value=17.0, step=0.1)
bsn = st.sidebar.number_input("Lactate de base Bsn (mmol/L)", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
n = st.sidebar.number_input("Nombre de paliers", min_value=6, max_value=20, value=10, step=1)
pct_start = st.sidebar.number_input("%VMA palier de départ", min_value=40.0, max_value=80.0, value=60.0, step=1.0)
pct_end   = st.sidebar.number_input("%VMA palier final",   min_value=80.0, max_value=120.0, value=105.0, step=1.0)
duree_min = st.sidebar.number_input("Durée d'un palier (min)", min_value=2.0, max_value=6.0, value=3.0, step=0.5)
fc_rest = st.sidebar.number_input("FC repos (bpm) – optionnel", min_value=0, max_value=120, value=0, step=1)
fc_max  = st.sidebar.number_input("FC max (bpm) – optionnel",   min_value=0, max_value=240, value=0, step=1)

# --------------- Génération du tableau des paliers ---------------
pcts = np.linspace(pct_start, pct_end, int(n))
speeds = vma * pcts / 100.0
allure_min = np.array([pace_min_per_km(s) for s in speeds])
allure_txt = np.array([pace_mmss(s) for s in speeds])
fc_est = np.array([fc_estimee_from_vma_pct(p, fc_rest if fc_rest>0 else np.nan, fc_max if fc_max>0 else np.nan) for p in pcts])

df = pd.DataFrame({
    "Palier": np.arange(1, n+1, dtype=int),
    "%VMA": np.round(pcts, 2),
    "Vitesse (km/h)": np.round(speeds, 2),
    "Allure (min/km)": np.round(allure_min, 2),
    "Allure (mm:ss/km)": allure_txt,
    "FC estimée (bpm)": np.where(np.isnan(fc_est), None, np.round(fc_est).astype("float")),
    "FC mesurée (bpm)": [None]*n,
    "Lactate (mmol/L)": [None]*n
})

st.markdown("### Saisie des lactates (et FC mesurée si dispo)")
df_edit = st.data_editor(
    df,
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

# --------------- Calculs ---------------
x = df_edit["Vitesse (km/h)"].astype(float).to_numpy()
y = pd.to_numeric(df_edit["Lactate (mmol/L)"], errors="coerce").to_numpy()

# SL1 (Bsn+0,5)
sl1_level = bsn + 0.5
sl1_speed = interp_speed_at_lactate(x, y, sl1_level)

# LT1 (log-lactate)
lt1_speed, lt1_model = lt1_log_lactate_breakpoint(x, y)

# LT2 Dmax
lt2_dmax_speed, dmax_meta = dmax_lt2(x, y)

# LT2 ModDmax (ligne depuis (x at Bsn+0,5; Bsn+0,5) vers dernier point)
x_sl1_for_line = sl1_speed if not np.isnan(sl1_speed) else np.nan
lt2_moddmax_speed, moddmax_meta = moddmax_lt2(x, y, x_sl1_for_line, sl1_level if not np.isnan(x_sl1_for_line) else np.nan)

# MLSS (90% du LT2 choisi)
lt2_method_choice = st.selectbox("Méthode LT2 pour MLSS", ["Dmax", "ModDmax"], index=0)
lt2_selected = lt2_dmax_speed if lt2_method_choice == "Dmax" else lt2_moddmax_speed
mlss_speed = lt2_selected * 0.9 if lt2_selected and not np.isnan(lt2_selected) else np.nan

# --------------- Affichage résultats ---------------
colA, colB = st.columns([1,1])

with colA:
    st.markdown("### Seuils (vitesses estimées)")
    st.write({
        "LT1 (log-lactate)": None if np.isnan(lt1_speed) else round(lt1_speed, 2),
        "SL1 (Bsn+0,5)": None if np.isnan(sl1_speed) else round(sl1_speed, 2),
        "LT2 (D-max)": None if np.isnan(lt2_dmax_speed) else round(lt2_dmax_speed, 2),
        "LT2 (ModDmax)": None if np.isnan(lt2_moddmax_speed) else round(lt2_moddmax_speed, 2),
        "MLSS (90% du LT2 choisi)": None if np.isnan(mlss_speed) else round(mlss_speed, 2)
    })
    st.caption("Note : SL1/ModDmax nécessitent que la courbe atteigne au moins Bsn+0,5 mmol/L.")

with colB:
    # Téléchargements CSV
    out_df = df_edit.copy()
    out_df["log10(lactate)"] = np.where(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")>0,
                                        np.log10(pd.to_numeric(out_df["Lactate (mmol/L)"], errors="coerce")), np.nan)
    meta = {
        "LT1_log_vitesse": lt1_speed,
        "SL1_Bsn+0.5_vitesse": sl1_speed,
        "LT2_Dmax_vitesse": lt2_dmax_speed,
        "LT2_ModDmax_vitesse": lt2_moddmax_speed,
        "MLSS_vitesse": mlss_speed,
        "Bsn": bsn, "VMA": vma
    }
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("Télécharger les données (CSV)", data=csv_buf.getvalue(), file_name="test_lactate.csv", mime="text/csv")

    meta_buf = io.StringIO()
    pd.DataFrame([meta]).to_csv(meta_buf, index=False)
    st.download_button("Télécharger la synthèse (CSV)", data=meta_buf.getvalue(), file_name="resultats_seuils.csv", mime="text/csv")

# --------------- Graphiques ---------------
st.markdown("### Courbe lactate – vitesse")
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(x, y, "o-", label="Mesures")
ymax = np.nanmax(y) if np.isfinite(np.nanmax(y)) else 0
ymax = ymax*1.15 if ymax>0 else 5

# Traits verticaux
def vline(val, label, color):
    if val and not np.isnan(val):
        ax1.axvline(val, color=color, linestyle="--", alpha=0.8, label=label)

vline(lt1_speed, "LT1 (log)", "#1f77b4")
vline(sl1_speed, "SL1 (Bsn+0,5)", "#2ca02c")
vline(lt2_dmax_speed, "LT2 (D-max)", "#d62728")
vline(lt2_moddmax_speed, "LT2 (ModDmax)", "#9467bd")

ax1.set_xlabel("Vitesse (km/h)")
ax1.set_ylabel("Lactate (mmol/L)")
ax1.set_ylim(0, ymax if ymax>0 else 5)
ax1.grid(True, alpha=0.3)
ax1.legend()
st.pyplot(fig1)

st.markdown("### Log-lactate – vitesse (avec rupture LT1)")
fig2, ax2 = plt.subplots(figsize=(7,4))
mask_valid = ~np.isnan(y) & (y>0)
xv, yv = x[mask_valid], y[mask_valid]
if len(xv) >= 2:
    ax2.plot(xv, np.log10(yv), "o-", label="log10(lactate)")
    if lt1_model is not None and lt1_model["idx"] is not None:
        b = lt1_model["idx"]
        x1 = xv[:b+1]
        y1_fit = lt1_model["m1"]*x1 + lt1_model["b1"]
        ax2.plot(x1, y1_fit, "-", color="#1f77b4", label="Fit segment 1")
        x2 = xv[b:]
        y2_fit = lt1_model["m2"]*x2 + lt1_model["b2"]
        ax2.plot(x2, y2_fit, "-", color="#d62728", label="Fit segment 2")
        ax2.axvline(lt1_speed, color="#1f77b4", linestyle="--", alpha=0.8, label="Rupture (LT1)")
    ax2.set_xlabel("Vitesse (km/h)")
    ax2.set_ylabel("log10(lactate)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
else:
    ax2.text(0.02, 0.6, "Saisir au moins 2 mesures > 0 mmol/L", transform=ax2.transAxes)
st.pyplot(fig2)

# --------------- Conseils MLSS ---------------
st.markdown("### Protocole MLSS (rappel)")
st.write(f"**Vitesse cible** (90% de {lt2_method_choice}) : **{mlss_speed:.2f} km/h**" if mlss_speed==mlss_speed else
         "Définis d’abord LT2 (D-max ou ModDmax) pour estimer la vitesse MLSS.")
st.caption("30' à vitesse constante ; prélèvements lactate à 10’–20’–30’. Critère MLSS : hausse ≤ 1,0 mmol/L entre 10’ et 30’.")