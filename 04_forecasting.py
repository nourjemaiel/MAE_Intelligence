import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import mlflow
from sklearn.linear_model import LinearRegression
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

# Croissance annuelle observée : +4.7% → mensuelle : (1.047)^(1/12) - 1
MONTHLY_GROWTH_RATE = (1.047 ** (1 / 12)) - 1

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

    # CORRIGÉ : filtre strict 2025 + vérification colonne
    df = df[df['DEBUT_PERI'].dt.year == 2025].copy()
    if 'PRIME_NETTE' not in df.columns:
        print("❌ Colonne PRIME_NETTE manquante.")
        return None

    df['Mois'] = df['DEBUT_PERI'].dt.to_period('M')
    monthly = df.groupby('Mois')['PRIME_NETTE'].sum().reset_index()
    monthly.columns = ['Mois', 'CA']
    monthly['ds'] = monthly['Mois'].dt.to_timestamp()
    monthly = monthly.sort_values('ds').reset_index(drop=True)

    if len(monthly) < 3:
        print(f"⚠️  Seulement {len(monthly)} mois disponibles — prévision peu fiable.")

    print(f"✅ {len(monthly)} mois de données chargés (2025).")
    print(monthly[['ds', 'CA']].to_string(index=False))
    print(f"\n📈 Taux de croissance mensuel : {MONTHLY_GROWTH_RATE*100:.3f}%")
    return monthly


# =================================================================
# 2. FEATURES : tendance + saisonnalite sin/cos (1 harmonique)
# =================================================================
# CHANGEMENT (voir model_comparison.ipynb, section tuning Optuna) : ce
# modele avait a l'origine 2 harmoniques (4 features saisonnieres + 1
# tendance = 5 au total). Sur seulement 12 points d'entrainement (l'annee
# 2025), un tuning Optuna evalue en validation croisee leave-one-out
# (LOOCV -- pas en erreur intra-echantillon, qui favorise mecaniquement la
# complexite sans garantir la generalisation) a montre qu'1 seule
# harmonique generalise MIEUX : RMSE LOOCV ~1.00M TND contre ~1.74M TND
# pour 2 harmoniques, soit ~43% d'erreur en moins hors-echantillon. La 2e
# harmonique captait probablement du bruit propre a 2025 plutot qu'un
# motif saisonnier reel. Feature set reduit a 3 (tendance + 1 harmonique).
def make_features(ds_series, t_offset=0):
    t = np.arange(t_offset, t_offset + len(ds_series))
    month = ds_series.dt.month.values
    X = np.column_stack([
        t,
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ])
    return X


# =================================================================
# 3. ENTRAÎNEMENT
# =================================================================
def train_model(monthly):
    X = make_features(monthly['ds'])
    y = monthly['CA'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)
    std  = np.std(y - y_pred)

    with mlflow.start_run(run_name="LinearRegression_Seasonal_Growth"):
        mlflow.log_metric("MAE",  mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)
        mlflow.log_metric("monthly_growth_rate", MONTHLY_GROWTH_RATE)
        mlflow.log_param("n_mois_entrainement", len(monthly))

    print(f"\n📊 Performance du modèle :")
    print(f"   MAE  : {mae/1e6:.2f}M TND")
    print(f"   RMSE : {rmse/1e6:.2f}M TND")
    print(f"   R²   : {r2:.3f}")
    return model, y_pred, std


# =================================================================
# 4. PRÉDICTION 12 MOIS AVEC TENDANCE DE CROISSANCE RÉELLE
# =================================================================
def predict_future(model, monthly, std, periods=12):
    last_date = monthly['ds'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods, freq='MS'
    )
    future_ds = pd.Series(future_dates)
    X_future  = make_features(future_ds, t_offset=len(monthly))

    y_base = model.predict(X_future)

    # Correction d'ancrage sur le dernier CA réel
    last_ca   = monthly['CA'].iloc[-1]
    base_last = model.predict(
        make_features(pd.Series([last_date]), t_offset=len(monthly) - 1)
    )[0]
    correction = last_ca / base_last if base_last > 0 else 1.0

    growth_factors = np.array([(1 + MONTHLY_GROWTH_RATE) ** (i + 1) for i in range(periods)])
    y_future = y_base * correction * growth_factors

    # Intervalle de confiance croissant
    ci_factors = np.array([1 + 0.1 * i for i in range(periods)])
    lower = y_future - 1.5 * std * ci_factors
    upper = y_future + 1.5 * std * ci_factors

    future_df = pd.DataFrame({
        'ds'         : future_dates,
        'Mois'       : future_dates.strftime('%Y-%m'),
        'yhat'       : y_future,
        'yhat_lower' : np.maximum(lower, 0),
        'yhat_upper' : upper,
    })
    return future_df


# =================================================================
# 5. VISUALISATION
# =================================================================
def plot_forecast(monthly, y_fitted, future_df):
    last_date = monthly['ds'].max()

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # Données réelles
    ax.fill_between(monthly['ds'], monthly['CA'], alpha=0.12, color=PASTEL_BLUE)
    ax.plot(monthly['ds'], monthly['CA'],
            marker='o', color=DARK_BLUE, linewidth=2.5,
            markersize=7, label='CA Réel 2025', zorder=5)

    # Ajustement modèle
    ax.plot(monthly['ds'], y_fitted,
            color=PASTEL_PURPLE, linewidth=1.5,
            linestyle='--', alpha=0.7, label='Ajustement modèle')

    # Intervalle de confiance
    ax.fill_between(future_df['ds'],
                    future_df['yhat_lower'], future_df['yhat_upper'],
                    alpha=0.20, color=PASTEL_PINK,
                    label='Intervalle de confiance (±1.5σ)')

    # Courbe prédite
    ax.plot(future_df['ds'], future_df['yhat'],
            marker='o', color=DARK_PINK, linewidth=2.5,
            linestyle='--', markersize=7,
            label='CA Prédit 2026 (+4.7% croissance)', zorder=5)

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

    # Boîte résumé croissance
    avg_2025 = monthly['CA'].mean()
    avg_2026 = future_df['yhat'].mean()
    growth_pct = (avg_2026 / avg_2025 - 1) * 100
    ax.text(0.75, 0.05,
            f'Croissance prévue 2026 : +{growth_pct:.1f}%',
            transform=ax.transAxes, fontsize=11, color=DARK_PINK,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=PASTEL_PINK, alpha=0.8))

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M TND'))
    ax.set_ylim(bottom=monthly['CA'].min() * 0.85)
    ax.set_title("Prévision du Chiffre d'Affaires — Jan à Déc 2026",
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
    print("🚀 Prévision CA 12 mois — MAE Assurances 2026\n")

    monthly = load_monthly_ca()
    if monthly is None:
        exit()

    model, y_fitted, std = train_model(monthly)
    future_df = predict_future(model, monthly, std, periods=12)

    plot_forecast(monthly, y_fitted, future_df)
    export_results(future_df)

    print("\n✅ TERMINÉ — Vérifie plots/07_forecast_ca.png")