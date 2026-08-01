import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =================================================================
# CONFIGURATION
# =================================================================
plt.style.use('seaborn-v0_8-whitegrid')
PASTEL_BLUE   = '#AEC6CF'
PASTEL_PINK   = '#FFB7C5'
PASTEL_PURPLE = '#C3B1E1'
PASTEL_GREEN  = '#B5EAD7'
PASTEL_ORANGE = '#FFDAB9'
PALETTE       = [PASTEL_BLUE, PASTEL_PINK, PASTEL_PURPLE, PASTEL_GREEN, PASTEL_ORANGE]

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_Clustering_Clients")
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Noms des segments ordonnés par CA décroissant (index 0 = plus fort CA)
# Emojis retirés pour éviter les avertissements de polices sous Windows/Matplotlib
SEGMENT_NOMS = [
    'Client Premium',
    'Client Standard',
    'Client Occasionnel',
    'Client a Risque',
    'Client Inactif',
]

# =================================================================
# 1. CHARGEMENT ET FEATURE ENGINEERING
# =================================================================
def load_and_prepare():
    path = "processed_data/Production_Cleaned.csv"
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable : {path}")
        return None, None

    df = pd.read_csv(path, low_memory=False)
    df['DEBUT_PERI'] = pd.to_datetime(df['DEBUT_PERI'], errors='coerce')
    df = df[df['DEBUT_PERI'].dt.year == 2025].copy()

    # Âge (date de naissance NON time-shiftée — préservée dans le cleaning)
    if 'DT_NAISS' in df.columns:
        df['DT_NAISS'] = pd.to_datetime(df['DT_NAISS'], errors='coerce', dayfirst=True)
        df['AGE'] = 2025 - df['DT_NAISS'].dt.year
    else:
        df['AGE'] = np.nan

    # Durée contrat
    if 'FIN_PERIOD' in df.columns:
        df['FIN_PERIOD'] = pd.to_datetime(df['FIN_PERIOD'], errors='coerce')
        df['DUREE_CONTRAT'] = (df['FIN_PERIOD'] - df['DEBUT_PERI']).dt.days
    else:
        df['DUREE_CONTRAT'] = np.nan

    # SEXE numérique
    if 'SEXE' in df.columns:
        df['SEXE_NUM'] = df['SEXE'].map({'H': 1, 'F': 0, 'M': 1})
    else:
        df['SEXE_NUM'] = np.nan

    # BONUS_MALUS numérique (peut contenir des strings)
    if 'BONUS_MALUS' in df.columns:
        df['BONUS_MALUS'] = pd.to_numeric(df['BONUS_MALUS'], errors='coerce')

    # Agréger par client
    agg_dict = {
        'NB_CONTRATS':     ('N_POLICE', 'count'),
        'CA_TOTAL':        ('PRIME_NETTE', 'sum'),
        'PRIME_MOYENNE':   ('PRIME_NETTE', 'mean'),
        'BONUS_MALUS_MOY': ('BONUS_MALUS', 'mean'),
        'AGE':             ('AGE', 'mean'),
        'DUREE_MOY':       ('DUREE_CONTRAT', 'mean'),
        'SEXE_NUM':        ('SEXE_NUM', 'mean'),
    }
    # CAPITAUX optionnel
    if 'CAPITAUX' in df.columns:
        agg_dict['CAPITAUX_MOY'] = ('CAPITAUX', 'mean')

    # Region optionnel (pour l'export, pas le clustering)
    region_col = 'Region' if 'Region' in df.columns else None

    group_cols = ['N_CLIENT'] if 'N_CLIENT' in df.columns else []
    if not group_cols:
        print("❌ Colonne N_CLIENT introuvable.")
        return None, None

    client_df = df.groupby('N_CLIENT').agg(**agg_dict).reset_index()

    if region_col:
        region_map = df.groupby('N_CLIENT')[region_col].first()
        client_df = client_df.join(region_map, on='N_CLIENT')

    # Features utilisées pour le clustering (sans CAPITAUX_MOY si absent)
    features = ['NB_CONTRATS', 'CA_TOTAL', 'PRIME_MOYENNE',
                'BONUS_MALUS_MOY', 'AGE', 'DUREE_MOY', 'SEXE_NUM']
    if 'CAPITAUX_MOY' in client_df.columns:
        features.insert(3, 'CAPITAUX_MOY')

    # Nettoyage : supprimer lignes avec NaN dans les features
    before = len(client_df)
    client_df = client_df.dropna(subset=features)
    client_df = client_df[
        (client_df['AGE'] >= 18) & (client_df['AGE'] <= 80) &
        (client_df['CA_TOTAL'] > 0)
    ].reset_index(drop=True)

    print(f"✅ {len(client_df):,} clients prêts ({before - len(client_df):,} écartés).")
    return client_df, features


# =================================================================
# 2. MÉTHODE DU COUDE + SILHOUETTE (AVEC SAMPLING INTELLIGENT)
# =================================================================
def plot_elbow(X_scaled):
    inertias    = []
    silhouettes = []
    K_range     = range(2, 9)

    # Échantillonner pour calculer la silhouette rapidement (max 10 000 lignes)
    np.random.seed(42)
    sample_size = min(10000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled[indices]

    print(f"⚡ Analyse de la silhouette sur un échantillon représentatif de {sample_size:,} clients...")

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        # On entraîne sur tout le dataset pour l'inertie globale
        labels_full = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        
        # On calcule le score de silhouette uniquement sur l'échantillon
        labels_sample = km.predict(X_sample)
        score = silhouette_score(X_sample, labels_sample)
        silhouettes.append(score)
        print(f"   k={k} traité (Silhouette: {score:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Choix du Nombre de Clusters', fontsize=14, fontweight='bold')

    axes[0].plot(list(K_range), inertias, marker='o', color=PASTEL_BLUE,
                 linewidth=2.5, markersize=8)
    axes[0].set_title('Méthode du Coude (Inertie)')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(K_range), silhouettes, marker='o', color=PASTEL_PINK,
                 linewidth=2.5, markersize=8)
    axes[1].set_title('Score de Silhouette')
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Score Silhouette')
    axes[1].grid(True, alpha=0.3)

    best_k = list(K_range)[int(np.argmax(silhouettes))]
    axes[1].axvline(x=best_k, color='gray', linestyle='--', alpha=0.7)
    axes[1].text(best_k + 0.1, max(silhouettes) * 0.99,
                 f' k={best_k} optimal', color='gray', fontsize=10)

    plt.tight_layout()
    
    # Suppression de sécurité pour Windows
    out_path = 'plots/08_elbow_silhouette.png'
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass

    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Plot 8 — Coude & Silhouette généré (k optimal = {best_k}).")
    return best_k


# =================================================================
# 3. ENTRAÎNEMENT K-MEANS
# =================================================================
def train_kmeans(X_scaled, k):
    with mlflow.start_run(run_name=f"KMeans_k{k}"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        
        # Silhouette rapide sur échantillon de 10k pour MLFlow
        np.random.seed(42)
        sample_size = min(10000, len(X_scaled))
        indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
        sil = silhouette_score(X_scaled[indices], labels[indices])
        
        mlflow.log_param("k", k)
        mlflow.log_metric("silhouette_score", sil)
        print(f"\n📊 KMeans k={k} — Silhouette Score (Échantillon) : {sil:.3f}")
    return model, labels


# =================================================================
# 4. NOMMER LES CLUSTERS (mapping cluster_id → nom métier)
# =================================================================
def build_cluster_name_map(client_df, k):
    ca_by_cluster = (client_df.groupby('CLUSTER')['CA_TOTAL']
                     .mean()
                     .sort_values(ascending=False))
    name_map   = {}
    label_map  = {}
    for rank, cluster_id in enumerate(ca_by_cluster.index):
        name_map[int(cluster_id)]  = SEGMENT_NOMS[rank]
        label_map[int(cluster_id)] = f"Seg_{rank+1}_{SEGMENT_NOMS[rank].split()[-1]}"
    return name_map, label_map


# =================================================================
# 5. VISUALISATION PCA 2D
# =================================================================
def plot_pca(X_scaled, labels, k, name_map):
    pca   = PCA(n_components=2, random_state=42)
    # Pour accélérer l'affichage des 72k points, on sous-échantillonne à 50k points max pour le scatter plot
    plot_sample_size = min(50000, len(X_scaled))
    np.random.seed(42)
    indices = np.random.choice(len(X_scaled), size=plot_sample_size, replace=False)
    
    X_scaled_sampled = X_scaled[indices]
    labels_sampled = labels[indices]
    
    X_pca = pca.fit_transform(X_scaled_sampled)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster_id in range(k):
        mask  = labels_sampled == cluster_id
        label = name_map.get(cluster_id, f'Segment {cluster_id+1}')
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=PALETTE[cluster_id % len(PALETTE)],
                   label=label, alpha=0.45, s=14, edgecolors='none')

    ax.set_title('Segmentation Clients — Projection PCA 2D (Échantillon de 50k)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f'Composante 1 ({explained[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'Composante 2 ({explained[1]:.1f}% variance)', fontsize=11)
    ax.legend(fontsize=10, markerscale=2, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    out_path = 'plots/09_clustering_pca.png'
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass

    plt.savefig(out_path, dpi=150)
    plt.close()
    print("✅ Plot 9 — PCA 2D généré.")
    return X_pca


# =================================================================
# 6. PROFIL DE CHAQUE SEGMENT
# =================================================================
def plot_segment_profiles(client_df, k, name_map):
    client_df = client_df.copy()
    client_df['SEG_LABEL'] = client_df['CLUSTER'].map(
        lambda c: name_map.get(int(c), f'Seg {c+1}')
    )
    order = [name_map[i] for i in sorted(
        name_map.keys(),
        key=lambda c: client_df[client_df['CLUSTER'] == c]['CA_TOTAL'].mean(),
        reverse=True
    )]

    metrics = [
        ('CA_TOTAL',        'CA Total Moyen (TND)',       1e3,  'K'),
        ('PRIME_MOYENNE',   'Prime Nette Moyenne (TND)',   1,    ''),
        ('NB_CONTRATS',     'Nb Contrats Moyen',           1,    ''),
        ('AGE',             'Âge Moyen',                   1,    'ans'),
        ('BONUS_MALUS_MOY', 'Bonus-Malus Moyen',           1,    ''),
        ('DUREE_MOY',       'Durée Contrat Moy. (jours)',  1,    'j'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Profil des Segments Clients MAE', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, (col, label, divisor, unit) in enumerate(metrics):
        if col not in client_df.columns:
            axes[idx].set_visible(False)
            continue

        means = (client_df.groupby('SEG_LABEL')[col].mean() / divisor)
        means = means.reindex(order)

        bars = axes[idx].bar(
            means.index, means.values,
            color=PALETTE[:k], edgecolor='white'
        )
        for bar, val in zip(bars, means.values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                f'{val:,.0f}{unit}',
                ha='center', fontsize=8, color='#444'
            )
        axes[idx].set_title(label, fontsize=11, fontweight='bold')
        
        # CORRIGÉ : Définition explicite des ticks avant les labels pour éviter l'avertissement UserWarning
        axes[idx].set_xticks(range(len(means)))
        axes[idx].set_xticklabels(means.index, rotation=20, ha='right', fontsize=8)
        
        axes[idx].grid(True, alpha=0.2, axis='y')
        axes[idx].set_ylim(bottom=0, top=means.max() * 1.25 if means.max() > 0 else 1)
        axes[idx].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
        )

    plt.tight_layout()
    
    out_path = 'plots/10_segment_profiles.png'
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass

    plt.savefig(out_path, dpi=150)
    plt.close()
    print("✅ Plot 10 — Profils des segments généré.")


# =================================================================
# 7. RÉSUMÉ CONSOLE
# =================================================================
def print_summary(client_df, k, name_map):
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES SEGMENTS")
    print("=" * 60)
    summary = client_df.groupby('CLUSTER').agg(
        NB_CLIENTS  =('N_CLIENT', 'count'),
        CA_MOYEN    =('CA_TOTAL', 'mean'),
        AGE_MOY     =('AGE', 'mean'),
        BONUS_MOY   =('BONUS_MALUS_MOY', 'mean'),
        NB_CONTRATS =('NB_CONTRATS', 'mean'),
    ).reset_index()

    summary['NOM'] = summary['CLUSTER'].map(name_map)
    summary = summary.sort_values('CA_MOYEN', ascending=False)

    for _, row in summary.iterrows():
        print(f"\n{row['NOM']}")
        print(f"   Clients     : {int(row['NB_CLIENTS']):,}")
        print(f"   CA Moyen    : {row['CA_MOYEN']/1e3:.1f}K TND")
        print(f"   Âge Moyen   : {row['AGE_MOY']:.0f} ans")
        print(f"   Bonus-Malus : {row['BONUS_MOY']:.1f}")
        print(f"   Nb Contrats : {row['NB_CONTRATS']:.1f}")

    return summary


# =================================================================
# 8. EXPORT
# =================================================================
def export_results(client_df, name_map):
    out = client_df.copy()

    out['CLUSTER_ID']    = out['CLUSTER'] + 1          # 1-indexed pour lisibilité
    out['Segment_Label'] = out['CLUSTER'].map(name_map) # label métier

    export_cols = ['N_CLIENT', 'CLUSTER_ID', 'Segment_Label',
                   'CA_TOTAL', 'PRIME_MOYENNE', 'NB_CONTRATS',
                   'AGE', 'BONUS_MALUS_MOY']
    if 'CAPITAUX_MOY' in out.columns:
        export_cols.append('CAPITAUX_MOY')

    out[export_cols].to_csv('outputs/segments_clients.csv', index=False)
    print("\n📁 Exporté : outputs/segments_clients.csv")

    prod_path    = "processed_data/Production_Cleaned.csv"
    cluster_path = "processed_data/Production_Clusters.csv"
    if os.path.exists(prod_path):
        df_prod = pd.read_csv(prod_path, low_memory=False)
        df_prod = df_prod.merge(
            out[['N_CLIENT', 'CLUSTER_ID', 'Segment_Label']],
            on='N_CLIENT', how='left'
        )
        df_prod.to_csv(cluster_path, index=False)
        print(f"📁 Exporté : {cluster_path} (enrichi avec segments)")


# =================================================================
# EXÉCUTION
# =================================================================
if __name__ == "__main__":
    print("🚀 Segmentation Clients — MAE Assurances\n")

    client_df, features = load_and_prepare()
    if client_df is None:
        exit()

    # Normalisation
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(client_df[features])

    # Trouver k optimal (échantillonné pour aller ultra vite)
    best_k = plot_elbow(X_scaled)

    # Forcer k entre 3 et 5 pour garantir l'actionnabilité métier
    k = best_k if 3 <= best_k <= 5 else 4
    print(f"\n🎯 Nombre de clusters utilisé : k={k}")

    # Entraînement
    model, labels = train_kmeans(X_scaled, k)
    client_df['CLUSTER'] = labels

    # Mapping cluster → nom métier (basé sur CA)
    name_map, label_map = build_cluster_name_map(client_df, k)

    # Visualisations
    plot_pca(X_scaled, labels, k, name_map)
    plot_segment_profiles(client_df, k, name_map)

    # Résumé et export
    print_summary(client_df, k, name_map)
    export_results(client_df, name_map)

    print("\n✅ TERMINÉ — plots/08, 09, 10 + outputs/segments_clients.csv")