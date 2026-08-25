import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import mlflow
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =================================================================
# CONFIGURATION
# =================================================================
plt.style.use('seaborn-v0_8-whitegrid')
PASTEL_BLUE   = '#AEC6CF'
PASTEL_PINK   = '#FFB7C5'
DARK_BLUE     = '#6A9BAF'
DARK_PINK     = '#c45c7a'
PASTEL_PURPLE = '#C3B1E1'

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_Forecasting_Final")
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =================================================================
# 1. CHARGEMENT
# =================================================================
def load_monthly_ca():
    path = "processed_data/Production_Cleaned.csv"
    if not os.path.exists(path):
        print("❌ Fichier introuvable.")
        return None

    df = pd.read_csv(path, low_memory=False)
    df['DEBUT_PERI'] = pd.to_datetime(df['DEBUT_PERI'], errors='coerce')
    if 'PRIME_NETTE' not in df.columns:
        print("❌ Colonne PRIME_NETTE manquante.")
        return None

    # MISE A JOUR (remarque 2, superviseur) : l'ancien filtre "annee == 2025"
    # ne correspond plus a rien depuis le decalage de dates DYNAMIQUE (voir
    # 02_cleaning.py, compute_dynamic_date_shift) -- les donnees ne tombent
    # plus dans une seule annee calendaire fixe. Remplace par la meme
    # detection DONNEES-DRIVEE que model_comparison.ipynb : le fichier brut
    # contient ~1 an de donnees denses (des milliers de contrats/semaine)
    # precede d'une longue traine eparse (0.05% du volume, artefact probable
    # d'un export "instantane des polices actives"). On isole la region
    # dense et on exclut le mois-frontiere partiel cree par la coupure
    # semaine/mois.
    weekly_counts = df.groupby(df['DEBUT_PERI'].dt.to_period('W')).size()
    dense_weeks = weekly_counts[weekly_counts > 100].index
    df = df[df['DEBUT_PERI'].dt.to_period('W').isin(dense_weeks)].copy()

    df['Mois'] = df['DEBUT_PERI'].dt.to_period('M')
    monthly = df.groupby('Mois')['PRIME_NETTE'].sum().reset_index()
    monthly.columns = ['Mois', 'CA']
    monthly['ds'] = monthly['Mois'].dt.to_timestamp()
    monthly = monthly.sort_values('ds').reset_index(drop=True)

    _before = len(monthly)
    monthly = monthly[monthly['CA'] > 1e6].reset_index(drop=True)
    if len(monthly) < _before:
        print(f"⚠️  {_before - len(monthly)} mois partiel(s) exclu(s) (effet de bord semaine/mois).")

    if len(monthly) < 3:
        print(f"⚠️  Seulement {len(monthly)} mois disponibles — prévision peu fiable.")

    print(f"✅ {len(monthly)} mois de données représentatives chargés.")
    print(monthly[['ds', 'CA']].to_string(index=False))
    return monthly


# =================================================================
# 2. ENTRAÎNEMENT (Prophet)
# =================================================================
# CHANGEMENT (remarque 2, superviseur : "R²=0.533 n'est pas bon, essayez
# d'autres modeles") : la Regression Lineaire Saisonniere a ete comparee
# empiriquement a 7 alternatives dans model_comparison.ipynb (Naif, SARIMA,
# ETS, ARIMA sans saisonnalite, XGBoost, LSTM, Prophet), dans une seule
# comparaison unifiee (intra-echantillon + LOOCV + backtest glissant hors-
# echantillon, ce dernier etant la mesure qui fait foi vu le peu de
# donnees). Prophet generalise nettement mieux que tous les autres modeles
# testes sur les trois mesures -- voir model_comparison.ipynb pour le
# detail. Adopte ici en production a la place de la regression lineaire +
# croissance manuelle.
# Hyperparametres tunes par Optuna + LOOCV (meme notebook, cellule "MODELE
# 7 : PROPHET") plutot que choisis a la main -- a regenerer ici si le
# notebook est ré-exécuté sur de nouvelles données.
PROPHET_CONFIG = dict(
    yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
    seasonality_mode='additive', changepoint_prior_scale=0.0244, seasonality_prior_scale=0.599,
)
PROPHET_FOURIER_ORDER = 1

def train_model(monthly):
    df_prophet = monthly[['ds', 'CA']].rename(columns={'CA': 'y'})
    model = Prophet(**PROPHET_CONFIG)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=PROPHET_FOURIER_ORDER)
    model.fit(df_prophet)

    y = monthly['CA'].values
    y_pred = model.predict(df_prophet)['yhat'].values
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    with mlflow.start_run(run_name="Prophet_Seasonal"):
        mlflow.log_metric("MAE",  mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)
        mlflow.log_param("n_mois_entrainement", len(monthly))

    print(f"\n📊 Performance du modèle (Prophet) :")
    print(f"   MAE  : {mae/1e6:.2f}M TND")
    print(f"   RMSE : {rmse/1e6:.2f}M TND")
    print(f"   R²   : {r2:.3f}")
    return model, y_pred


# =================================================================
# 3. PRÉDICTION 12 MOIS
# =================================================================
# Intervalles de confiance et tendance viennent directement de Prophet
# (interval_width par defaut = 80%) -- plus de correction d'ancrage ni de
# taux de croissance fixe superposes a la main : Prophet apprend deja sa
# propre tendance et incertitude a partir des donnees.
def predict_future(model, monthly, periods=12):
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    future_df = forecast.iloc[-periods:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_df['yhat_lower'] = future_df['yhat_lower'].clip(lower=0)
    future_df['Mois'] = future_df['ds'].dt.strftime('%Y-%m')
    future_df = future_df.reset_index(drop=True)
    return future_df


# =================================================================
# 5. VISUALISATION
# =================================================================
def plot_forecast(monthly, y_fitted, future_df):
    last_date = monthly['ds'].max()

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    hist_label = f"CA Réel ({monthly['ds'].min().strftime('%b %Y')} – {monthly['ds'].max().strftime('%b %Y')})"
    forecast_label = f"CA Prévu ({future_df['ds'].min().strftime('%b %Y')} – {future_df['ds'].max().strftime('%b %Y')})"

    # Données réelles
    ax.fill_between(monthly['ds'], monthly['CA'], alpha=0.12, color=PASTEL_BLUE)
    ax.plot(monthly['ds'], monthly['CA'],
            marker='o', color=DARK_BLUE, linewidth=2.5,
            markersize=7, label=hist_label, zorder=5)

    # Ajustement modèle
    ax.plot(monthly['ds'], y_fitted,
            color=PASTEL_PURPLE, linewidth=1.5,
            linestyle='--', alpha=0.7, label='Ajustement modèle')

    # Intervalle de confiance (natif Prophet, 80% par défaut)
    ax.fill_between(future_df['ds'],
                    future_df['yhat_lower'], future_df['yhat_upper'],
                    alpha=0.20, color=PASTEL_PINK,
                    label='Intervalle de confiance (Prophet, 80%)')

    # Courbe prédite
    ax.plot(future_df['ds'], future_df['yhat'],
            marker='o', color=DARK_PINK, linewidth=2.5,
            linestyle='--', markersize=7,
            label=forecast_label, zorder=5)

    # Annotations (1 sur 2 pour éviter surcharge)
    for i, (_, row) in enumerate(future_df.iterrows()):
        if i % 2 == 0:
            ax.annotate(
                f"{row['yhat']/1e6:.0f}M",
                xy=(row['ds'], row['yhat']),
                xytext=(0, 14), textcoords='offset points',
                ha='center', fontsize=9, color=DARK_PINK, fontweight='bold'
            )

    # Ligne de séparation historique/prévision
    ax.axvline(x=last_date, color='#888', linestyle=':', linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(last_date, ylim[1] * 0.98,
            '  Début prévision ▶',
            color='#666', fontsize=10, va='top', style='italic')

    # Boîte résumé croissance (moyenne prévue vs moyenne historique, pas un
    # taux fixe imposé -- directement lu sur la sortie du modèle)
    avg_hist     = monthly['CA'].mean()
    avg_forecast = future_df['yhat'].mean()
    growth_pct = (avg_forecast / avg_hist - 1) * 100
    ax.text(0.75, 0.05,
            f'Croissance prévue : {growth_pct:+.1f}%',
            transform=ax.transAxes, fontsize=11, color=DARK_PINK,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=PASTEL_PINK, alpha=0.8))

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M TND'))
    ax.set_ylim(bottom=monthly['CA'].min() * 0.85)
    ax.set_title(
        f"Prévision du Chiffre d'Affaires — {future_df['ds'].min().strftime('%b %Y')} "
        f"à {future_df['ds'].max().strftime('%b %Y')}",
        fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('CA Mensuel (TND)', fontsize=12)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/07_forecast_ca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 7 — Prévision CA 12 mois généré.")


# =================================================================
# 6. EXPORT (2 fichiers : 12 mois + 6 mois pour compatibilité app)
# =================================================================
def export_results(future_df):
    cols_rename = {
        'Mois': 'Mois',
        'yhat': 'CA_Prevu_TND',
        'yhat_lower': 'Borne_Basse_TND',
        'yhat_upper': 'Borne_Haute_TND'
    }
    out = future_df[list(cols_rename.keys())].copy()
    out = out.rename(columns=cols_rename)
    for col in ['CA_Prevu_TND', 'Borne_Basse_TND', 'Borne_Haute_TND']:
        out[col] = out[col].round(0).astype(int)

    # 12 mois complet
    out.to_csv('outputs/previsions_ca_12mois.csv', index=False)
    print("\n📁 Exporté : outputs/previsions_ca_12mois.csv")

    # 6 mois (compatibilité avec d'éventuels scripts qui lisent ce fichier)
    out.head(6).to_csv('outputs/previsions_ca_6mois.csv', index=False)
    print("📁 Exporté : outputs/previsions_ca_6mois.csv")
    print(out.to_string(index=False))


# =================================================================
# EXÉCUTION
# =================================================================
if __name__ == "__main__":
    print("🚀 Prévision CA 12 mois — MAE Assurances\n")

    monthly = load_monthly_ca()
    if monthly is None:
        exit()

    model, y_fitted = train_model(monthly)
    future_df = predict_future(model, monthly, periods=12)

    plot_forecast(monthly, y_fitted, future_df)
    export_results(future_df)

    print("\n✅ TERMINÉ — Vérifie plots/07_forecast_ca.png")