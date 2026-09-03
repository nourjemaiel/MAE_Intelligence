import sys
# Meme fix que mae_backend/main.py : console Windows en cp1252 par defaut,
# qui ne sait pas encoder les emojis utilises dans les print() ci-dessous.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =================================================================
# CONFIGURATION
# =================================================================
# POURQUOI CE FICHIER : "combien d'accidents au total sur l'annee a venir ?"
# -- distinct d'une prevision du MONTANT regle (abandonnee, voir historique :
# le montant a un coefficient de variation ~1.27, trop bruite pour qu'aucun
# modele ne batte une prevision plate). Le NOMBRE d'accidents/mois, lui, a
# un CV ~0.06 (961-1121 sur 12 mois) avec une tendance a la hausse
# detectable -- un signal exploitable.
#
# MODELE RETENU : Regression de Poisson (GLM) -- PAS les memes modeles que
# pour le CA (Regression Lineaire/ETS/Prophet, adaptes a une variable
# monetaire CONTINUE). Le nombre d'accidents est une variable de COMPTAGE
# (entier, non-negatif) : voir model_comparison_accidents.ipynb, qui compare
# les modeles propres a ce type de donnees -- Naif, Poisson GLM, Binomiale
# Negative GLM (Optuna sur le parametre de surdispersion alpha), ARIMA
# (Optuna sur p,d,q), ETS. Sur le backtest a horizon complet (entrainement
# 9 mois -> prevision des 3 derniers d'un coup, la mesure pertinente pour
# une prevision annuelle), Poisson et Binomiale Negative sont ex-aequo
# (RMSE 34.2), ETS tres proche (34.6), ARIMA et Naif nettement derriere.
# Poisson retenu : meme prevision ponctuelle que la Binomiale Negative
# (dont le alpha=1.96 confirme une vraie surdispersion, mais qui n'ameliore
# que l'incertitude, pas la tendance predite), avec un seul parametre a
# estimer et l'interpretation actuarielle standard pour une frequence de
# sinistres (lien logarithmique => previsions toujours positives, contrairement
# a une regression lineaire classique qui pourrait extrapoler en negatif).
#
# RAFFINEMENT (apres diagnostic des residus) : le modele tendance-seule
# retenu ci-dessus produit une prevision quasi rectiligne qui ignore
# l'alternance mois-haut/mois-bas visible sur les 12 mois reels. Voir
# build_features() ci-dessous pour le terme ajoute et sa justification.
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_Forecasting_Final")
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =================================================================
# 1. CHARGEMENT
# =================================================================
def load_monthly_count():
    path = "processed_data/Sinistres_Cleaned.csv"
    if not os.path.exists(path):
        print("❌ Fichier introuvable.")
        return None
    df = pd.read_csv(path, low_memory=False)
    df['DATE_ACCIDENT'] = pd.to_datetime(df['DATE_ACCIDENT'], errors='coerce')
    df['Mois'] = df['DATE_ACCIDENT'].dt.to_period('M')
    monthly = df.groupby('Mois').size().reset_index(name='NB_ACCIDENTS')
    monthly['ds'] = monthly['Mois'].dt.to_timestamp()
    monthly = monthly.sort_values('ds').reset_index(drop=True)
    print(f"✅ {len(monthly)} mois de données chargés.")
    print(monthly[['ds', 'NB_ACCIDENTS']].to_string(index=False))
    return monthly


# =================================================================
# 2. ENTRAINEMENT (Regression de Poisson -- voir justification ci-dessus)
# =================================================================
# Le modele tendance-seule (const + t) sous-ajuste : les residus montrent une
# autocorrelation au lag 1 fortement negative (~-0.68 sur les 12 mois reels),
# un mois haut est presque toujours suivi d'un mois bas -- une vraie
# alternance mois-pair/mois-impair, pas juste du bruit (terme significatif
# a p=0.028 sur les 12 mois, backtest MAE 32.7 -> 14.1 une fois ajoute).
# cos(pi*t) vaut +1/-1 en alternance : c'est le moyen le plus simple d'ajouter
# ce seul degre de liberte sans le surparametrage qu'un terme saisonnier
# annuel complet imposerait sur une serie de seulement 12 points.
def build_features(t):
    alternance = np.cos(np.pi * t)
    return sm.add_constant(np.column_stack([t, alternance]))


def train_model(monthly):
    t = np.arange(len(monthly))
    X = build_features(t)
    y = monthly['NB_ACCIDENTS'].values.astype(float)

    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    y_pred = model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    with mlflow.start_run(run_name="Accidents_PoissonGLM"):
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_param("n_mois_entrainement", len(monthly))
        mlflow.log_param("pente_log", model.params[1])
        mlflow.log_param("coef_alternance_log", model.params[2])

    print(f"\n📊 Performance du modèle (Régression de Poisson, tendance + alternance) :")
    print(f"   MAE  : {mae:.1f} accidents")
    print(f"   RMSE : {rmse:.1f} accidents")
    print(f"   R²   : {r2:.3f}")
    print(f"   Pente (échelle log) : {model.params[1]:+.4f}")
    print(f"   Coef. alternance (échelle log) : {model.params[2]:+.4f}")
    return model, y_pred, rmse


# =================================================================
# 3. PRÉDICTION 12 MOIS
# =================================================================
def predict_future(monthly, model, resid_rmse, periods=12):
    t_future = np.arange(len(monthly), len(monthly) + periods)
    X_future = build_features(t_future)
    yhat = model.predict(X_future)
    # Largeur dynamique (pas un +-RMSE fixe repete 12 fois) : un mois predit
    # loin dans l'horizon est moins fiable qu'un mois predit juste apres la
    # derniere donnee reelle. width ~ resid_rmse * sqrt(horizon), convention
    # standard pour une marche aleatoire (variance qui croit lineairement
    # avec le nombre de pas -> ecart-type en sqrt(pas)) : l'intervalle est
    # le plus etroit au mois 1 et s'elargit progressivement jusqu'au mois 12.
    horizon = np.arange(1, periods + 1)
    width = resid_rmse * np.sqrt(horizon)
    future_dates = pd.date_range(monthly['ds'].max(), periods=periods + 1, freq='MS')[1:]
    future_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': np.round(yhat).astype(int),
        'yhat_lower': np.clip(np.round(yhat - width), 0, None).astype(int),
        'yhat_upper': np.round(yhat + width).astype(int),
    })
    future_df['Mois'] = future_df['ds'].dt.strftime('%Y-%m')
    return future_df


# =================================================================
# 4. VISUALISATION
# =================================================================
def plot_forecast(monthly, y_fitted, future_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    last_date = monthly['ds'].max()
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    ax.fill_between(monthly['ds'], monthly['NB_ACCIDENTS'], alpha=0.12, color='#A8D5BA')
    ax.plot(monthly['ds'], monthly['NB_ACCIDENTS'], marker='o', color='#2e7d4f', linewidth=2.5,
            markersize=7, label='Accidents réels', zorder=5)
    ax.plot(monthly['ds'], y_fitted, color='#888', linewidth=1.5, linestyle='--', alpha=0.7,
            label='Ajustement modèle (Poisson GLM)')
    ax.fill_between(future_df['ds'], future_df['yhat_lower'], future_df['yhat_upper'],
                    alpha=0.20, color='#F4B6B6', label="Intervalle de confiance")
    ax.plot(future_df['ds'], future_df['yhat'], marker='o', color='#b03a3a', linewidth=2.5,
            linestyle='--', markersize=7, label='Accidents prévus', zorder=5)

    ax.axvline(x=last_date, color='#888', linestyle=':', linewidth=1.5)
    ax.set_title("Prévision du Nombre d'Accidents — 12 prochains mois (modèle : Régression de Poisson)",
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel("Nombre d'accidents / mois", fontsize=12)

    # Zoom vertical sur la plage reelle des valeurs : un axe partant de 0
    # (necessaire pour le fill_between ci-dessus si on le laissait non borne)
    # ecrase visuellement une variation reelle de ~150-250 accidents/mois
    # contre une echelle 0-1250. Le fill_between reste visuellement correct :
    # matplotlib le coupe proprement a la limite basse de l'axe.
    y_min = min(monthly['NB_ACCIDENTS'].min(), future_df['yhat_lower'].min())
    y_max = max(monthly['NB_ACCIDENTS'].max(), future_df['yhat_upper'].max())
    margin = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/13_forecast_accidents.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 13 — Prévision du nombre d'accidents générée.")


# =================================================================
# 5. EXPORT
# =================================================================
def export_results(future_df):
    future_df[['Mois', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'yhat': 'Accidents_Prevus', 'yhat_lower': 'Borne_Basse', 'yhat_upper': 'Borne_Haute'
    }).to_csv('outputs/previsions_accidents_12mois.csv', index=False)
    print("\n📁 Exporté : outputs/previsions_accidents_12mois.csv")
    print(future_df[['Mois', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))


# =================================================================
# EXÉCUTION
# =================================================================
if __name__ == "__main__":
    print("🚀 Prévision du nombre d'accidents 12 mois — MAE Assurances")
    print("   (modèle choisi via model_comparison_accidents.ipynb)\n")

    monthly = load_monthly_count()
    if monthly is None:
        exit()

    model, y_fitted, rmse = train_model(monthly)
    future_df = predict_future(monthly, model, rmse, periods=12)

    plot_forecast(monthly, y_fitted, future_df)
    export_results(future_df)

    total_12 = future_df['yhat'].sum()
    print(f"\n✅ TERMINÉ — {total_12:,} accidents prévus sur les 12 prochains mois "
          f"(vs {int(monthly['NB_ACCIDENTS'].sum()):,} sur les 12 derniers). "
          f"Vérifie plots/13_forecast_accidents.png")
