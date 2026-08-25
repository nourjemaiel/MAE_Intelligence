import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =================================================================
# CONFIGURATION
# =================================================================
# POURQUOI CE FICHIER (remarque superviseur) : la premiere version de la
# "prediction sinistres" etait une prevision de la SOMME MENSUELLE des
# sinistres (06_sinistres_forecasting.py) -- seulement 12 points de donnees,
# sans tendance ni saisonnalite detectable (verifie par backtest ET par
# validation glissante sur 6 points, voir historique de session : le
# meilleur modele reste plat quel que soit le modele teste, y compris
# Simple Exponential Smoothing). Une prevision plate sur seulement 12
# points ressemble visuellement a "juste la moyenne", ce qui n'est pas
# presentable comme une vraie prediction.
#
# CE FICHIER remplace cette approche par une prediction au niveau CLIENT
# (pas mensuel agrege) : "ce client va-t-il avoir un sinistre ?", en
# s'appuyant sur outputs/segments_clients.csv (72 675 clients, 12.7% avec
# au moins un sinistre) -- 6 000x plus d'observations que la serie
# mensuelle, donc un probleme de prediction reellement exploitable avec un
# vrai signal a comparer entre modeles.
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_Forecasting_Final")
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

FEATURES = ['NB_CONTRATS', 'CA_TOTAL', 'PRIME_MOYENNE', 'BONUS_MALUS_MOY', 'AGE', 'CAPITAUX_MOY']
TARGET   = 'HAS_SINISTRE'


# =================================================================
# 1. CHARGEMENT
# =================================================================
def load_data():
    path = "outputs/segments_clients.csv"
    if not os.path.exists(path):
        print("❌ Fichier introuvable -- executer 05_clustering.py d'abord.")
        return None
    df = pd.read_csv(path)
    df[TARGET] = (df['NB_SINISTRES'] > 0).astype(int)
    print(f"✅ {len(df):,} clients chargés — {df[TARGET].sum():,} avec sinistre "
          f"({df[TARGET].mean()*100:.1f}%), {len(df)-df[TARGET].sum():,} sans.")
    return df


# =================================================================
# 2. TUNING (Optuna) -- Random Forest uniquement
# =================================================================
# POURQUOI Optuna ici et pas partout : la Regression Logistique n'a quasi
# aucun hyperparametre a fort impact (C reste secondaire une fois
# class_weight='balanced' fixe) ; le Random Forest, lui, en a plusieurs qui
# interagissent (n_estimators, max_depth, min_samples_leaf/split,
# max_features) -- avant ce correctif, choisis a la main (200, 8) sans
# recherche systematique. Meme methodologie que le tuning Prophet de
# 04_forecasting.py (voir model_comparison.ipynb) : recherche optimisee
# plutot que des valeurs choisies au jugement.
def tune_random_forest(X, y, n_trials=30):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 50),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 50),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"   🔧 Optuna ({n_trials} essais) : meilleur ROC-AUC={study.best_value:.3f}, "
          f"params={study.best_params}")
    with mlflow.start_run(run_name="Sinistres_RF_Optuna_Tuning"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_roc_auc", study.best_value)
    return study.best_params


# =================================================================
# 3. COMPARAISON DE MODELES (validation croisée stratifiée, 5 folds)
# =================================================================
# Classes desequilibrees (12.7% positif) -- l'accuracy seule serait
# trompeuse (un modele qui predit toujours "pas de sinistre" aurait deja
# 87.3% d'accuracy sans rien apprendre). ROC-AUC et le rappel (recall) sont
# les mesures qui comptent ici : un assureur qui rate les vrais clients a
# risque (faux negatifs) est plus problematique qu'un faux positif.
def compare_models(df, rf_params):
    X = df[FEATURES].values
    y = df[TARGET].values

    models = {
        'Regression Logistique': Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ]),
        'Random Forest': RandomForestClassifier(
            **rf_params, class_weight='balanced', random_state=42, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['roc_auc', 'precision', 'recall', 'f1']

    results = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results.append({
            'Modele':    name,
            'ROC_AUC':   scores['test_roc_auc'].mean(),
            'Precision': scores['test_precision'].mean(),
            'Rappel':    scores['test_recall'].mean(),
            'F1':        scores['test_f1'].mean(),
        })
        print(f"   {name:25s}  ROC-AUC={scores['test_roc_auc'].mean():.3f}  "
              f"Precision={scores['test_precision'].mean():.3f}  "
              f"Rappel={scores['test_recall'].mean():.3f}  F1={scores['test_f1'].mean():.3f}")

    results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False).reset_index(drop=True)

    with mlflow.start_run(run_name="Sinistres_Risk_Classification_Comparison"):
        for _, row in results_df.iterrows():
            for metric in ['ROC_AUC', 'Precision', 'Rappel', 'F1']:
                mlflow.log_metric(f"{metric}_{row['Modele']}", row[metric])

    winner = results_df.iloc[0]['Modele']
    print(f"\n🏆 Meilleur modèle (ROC-AUC le plus haut) : {winner}")
    return winner, results_df


# =================================================================
# 4. ENTRAINEMENT FINAL + IMPORTANCE DES VARIABLES
# =================================================================
def train_final_and_explain(df, winner, rf_params):
    X = df[FEATURES].values
    y = df[TARGET].values

    if winner == 'Random Forest':
        model = RandomForestClassifier(
            **rf_params, class_weight='balanced', random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        importances = model.feature_importances_
    else:
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ])
        pipe.fit(X, y)
        model = pipe
        # |coefficient| sur variables standardisees == importance relative comparable
        importances = np.abs(pipe.named_steps['clf'].coef_[0])

    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
    imp_df['Importance'] = imp_df['Importance'] / imp_df['Importance'].sum()  # normalise a 1
    imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    df['SCORE_RISQUE'] = model.predict_proba(X)[:, 1]
    return imp_df, df


# =================================================================
# 5. VISUALISATION
# =================================================================
def plot_importance(imp_df, winner):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#0b3d1f', '#156b32', '#1a7a3a', '#2ea84f', '#4cba6a', '#7fd99a']
    ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1], color=colors[:len(imp_df)][::-1])
    ax.set_title(f"Importance des variables — prédiction du risque de sinistre ({winner})",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance relative")
    plt.tight_layout()
    plt.savefig('plots/12_feature_importance_risque.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 12 — Importance des variables générée.")


# =================================================================
# 6. EXPORT
# =================================================================
def export_results(comparison_df, imp_df, df, winner):
    comparison_df.to_csv('outputs/comparaison_modeles_risque_sinistre.csv', index=False)
    print("📁 Exporté : outputs/comparaison_modeles_risque_sinistre.csv")

    imp_df.to_csv('outputs/importance_variables_risque.csv', index=False)
    print("📁 Exporté : outputs/importance_variables_risque.csv")

    # Liste nominative (top 200) -- usage interne uniquement (ex. ciblage
    # commercial par un gestionnaire de compte), PAS pour le tableau de bord
    # Direction Generale (remarque superviseur : la DG a besoin de stats/
    # graphes agreges, pas d'une liste client par client).
    top_risque = df[['N_CLIENT', 'Segment_Label', 'CA_TOTAL', 'BONUS_MALUS_MOY', 'SCORE_RISQUE']] \
        .sort_values('SCORE_RISQUE', ascending=False).head(200)
    top_risque.to_csv('outputs/scores_risque_clients.csv', index=False)
    print(f"📁 Exporté : outputs/scores_risque_clients.csv (top 200, usage interne, modèle {winner})")

    # Agrege par segment -- usage interne/rapport uniquement (pas affiche sur
    # le tableau de bord : le decoupage par segment reflete surtout
    # l'exposition, voir la note dans train_final_and_explain, donc un
    # affichage brut serait trompeur -- remplace sur le dashboard par la
    # prevision du nombre d'accidents, 08_accidents_forecasting.py). Seuil
    # "risque eleve" = quartile superieur des scores predits sur l'ensemble
    # du portefeuille (top 25%, seuil relatif derive des donnees, pas un
    # chiffre absolu choisi a la main).
    seuil_eleve = df['SCORE_RISQUE'].quantile(0.75)
    df['RISQUE_ELEVE_PREDIT'] = df['SCORE_RISQUE'] >= seuil_eleve
    par_segment = df.groupby('Segment_Label').agg(
        nb_clients=('N_CLIENT', 'count'),
        nb_risque_eleve=('RISQUE_ELEVE_PREDIT', 'sum'),
    ).reset_index()
    par_segment['part_pct'] = (par_segment['nb_risque_eleve'] / par_segment['nb_clients'] * 100).round(1)
    par_segment = par_segment.sort_values('part_pct', ascending=False)
    par_segment.to_csv('outputs/risque_predit_par_segment.csv', index=False)
    print(f"📁 Exporté : outputs/risque_predit_par_segment.csv (seuil top 25%, {int(df['RISQUE_ELEVE_PREDIT'].sum())} clients)")


# =================================================================
# EXÉCUTION
# =================================================================
if __name__ == "__main__":
    print("🚀 Prédiction du risque de sinistre par client — MAE Assurances\n")

    df = load_data()
    if df is None:
        exit()

    print("\n🔧 Tuning du Random Forest (Optuna, 30 essais) :")
    rf_params = tune_random_forest(df[FEATURES].values, df[TARGET].values)

    print("\n📊 Comparaison de modèles (validation croisée stratifiée, 5 folds) :")
    winner, comparison_df = compare_models(df, rf_params)

    imp_df, df = train_final_and_explain(df, winner, rf_params)
    print(f"\n📈 Importance des variables ({winner}) :")
    print(imp_df.to_string(index=False))

    plot_importance(imp_df, winner)
    export_results(comparison_df, imp_df, df, winner)

    print("\n✅ TERMINÉ — Vérifie plots/12_feature_importance_risque.png")
