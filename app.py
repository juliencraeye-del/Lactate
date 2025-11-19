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

# -------------------- Helpers --------------------
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

# ... (toutes les fonctions inchangées jusqu’au bloc HTML export)

# Bloc HTML corrigé :
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
<p class="meta"><b>Athlète:</b> {athlete} <b>Date:</b> {date_s}
<b>VMA:</b> {vma:.2f} km/h <b>Bsn:</b> {bsn:.2f} mmol/L</p>
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
data:image/png;base64,{img2_b64}
</div>
{mlss_html}
{srs_html}
<p style="margin-top:24px;font-size:12px;color:#666;">Généré par l’app Streamlit (version {VERSION}).</p>
</body></html>
"""

# Correction MLSS :
img_tag = (f'data:image/png;base64,{mlss_img_b64}'
           if mlss_img_b64 else '<p><i>(Ajoute ≥2 valeurs de lactate pour afficher la courbe.)</i></p>')

# Correction SRS :
img_tag = (f'<img src="data:image/png;baseg_b64}'
           if srs_img_b64 else '<p><i>(Renseigne la pente, SV1/SV2 et vitesses équivalentes pour obtenir la correction.)</i></p>')