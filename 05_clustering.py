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
PASTEL_YELLOW = '#FDFD96'
PASTEL_TEAL   = '#A0E7E5'
PASTEL_MAUVE  = '#D8BFD8'
# 8 couleurs distinctes -- k n'est plus fige a 5 (remarque superviseur), il
# faut assez de couleurs pour eviter que 2 segments partagent la meme.
PALETTE = [PASTEL_BLUE, PASTEL_PINK, PASTEL_PURPLE, PASTEL_GREEN,
           PASTEL_ORANGE, PASTEL_YELLOW, PASTEL_TEAL, PASTEL_MAUVE]

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_Clustering_Clients")
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

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

    # Meme detection DONNEES-DRIVEE que 04_forecasting.py/model_comparison.ipynb :
    # l'ancien filtre "annee == 2025" ne gardait que 41% des lignes (le decalage
    # de dates dynamique fait deborder la region dense sur 2 annees civiles).
    weekly_counts = df.groupby(df['DEBUT_PERI'].dt.to_period('W')).size()
    dense_weeks = weekly_counts[weekly_counts > 100].index
    df = df[df['DEBUT_PERI'].dt.to_period('W').isin(dense_weeks)].copy()

    # Âge au moment du contrat (par ligne, pas une annee de reference fixe --
    # les contrats s'etalent sur 2 annees civiles dans la region dense).
    if 'DT_NAISS' in df.columns:
        df['DT_NAISS'] = pd.to_datetime(df['DT_NAISS'], errors='coerce', dayfirst=True)
        df['AGE'] = df['DEBUT_PERI'].dt.year - df['DT_NAISS'].dt.year
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

    # Sinistres reels (remarque superviseur : utiliser le fichier Sinistres
    # au-dela de l'EDA/dashboard) -- calcules ici pour NOMMER le segment
    # "Client a Risque" sur un vrai sinistre observe plutot que sur
    # BONUS_MALUS_MOY (un proxy Production), mais PAS utilises comme feature
    # de clustering : teste empiriquement, RATIO_SP/NB_SINISTRES en entree
    # du clustering font tout simplement s'effondrer les 7 personas nuances
    # en 2 clusters (a risque / pas a risque) -- le signal sinistres est si
    # concentre (87% des clients a 0 sinistre, quelques valeurs extremes)
    # qu'il ecrase toutes les autres dimensions. Reste donc une donnee de
    # PROFIL/NOMMAGE post-clustering, pas un axe de segmentation.
    sin_path = "processed_data/Sinistres_Cleaned.csv"
    if os.path.exists(sin_path):
        sin = pd.read_csv(sin_path, low_memory=False)
        sin_agg = sin.groupby('N_CLIENT').agg(
            NB_SINISTRES=('N_SINISTRE', 'count'),
            TOTAL_REGLEMENTS=('REGLEMENTS', 'sum'),
        ).reset_index()
        client_df = client_df.merge(sin_agg, on='N_CLIENT', how='left')
        client_df['NB_SINISTRES']     = client_df['NB_SINISTRES'].fillna(0)
        client_df['TOTAL_REGLEMENTS'] = client_df['TOTAL_REGLEMENTS'].fillna(0)
        client_df['RATIO_SP'] = client_df['TOTAL_REGLEMENTS'] / client_df['CA_TOTAL']
    else:
        print(f"⚠️  {sin_path} introuvable -- clustering sans features de sinistres.")
        client_df['NB_SINISTRES'] = 0.0
        client_df['RATIO_SP'] = 0.0

    # Features utilisées pour le clustering (sans CAPITAUX_MOY si absent).
    # NB_SINISTRES/RATIO_SP ne sont PAS ici (voir commentaire ci-dessus) --
    # disponibles dans client_df pour le nommage/export, hors clustering.
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
    K_range     = range(2, 11)

    # Échantillonner pour calculer la silhouette rapidement (max 10 000 lignes)
    np.random.seed(42)
    sample_size = min(10000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled[indices]

    print(f"⚡ Analyse de la silhouette sur un échantillon représentatif de {sample_size:,} clients...")

    for k in K_range:
        # n_init=10 (pas 5) -- coherent avec le fit final (train_kmeans) : le
        # choix de k doit s'appuyer sur des clusters aussi bien optimises que
        # ceux qui seront reellement utilises, sinon le k retenu peut refleter
        # un optimum local sous-entraine plutot que la vraie structure.
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        # On entraîne sur tout le dataset pour l'inertie globale
        labels_full = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)

        # Score de silhouette sur l'echantillon (rapide, representatif)
        labels_sample = km.predict(X_sample)
        score = silhouette_score(X_sample, labels_sample)
        silhouettes.append(score)
        print(f"   k={k} traité (Silhouette: {score:.3f})")

    # Davies-Bouldin abandonne (remarque superviseur) : sur ces donnees, son
    # minimum ne coincidait jamais exactement avec le k retenu par la
    # silhouette, et forcer un discours de "quasi-accord" (plateau ±2%)
    # ajoutait de la confusion sans rien trancher. Plutot que de garder deux
    # métriques qui se contredisent et de devoir justifier l'ecart a chaque
    # fois, un seul critere clair est retenu : le score de silhouette, la
    # metrique de validation interne la plus standard pour K-Means. Coherent
    # avec l'esprit "tout doit etre data-driven" : mieux vaut un seul signal
    # net qu'un debat non tranche entre deux signaux.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Choix du Nombre de Clusters', fontsize=14, fontweight='bold')

    # Coude (inertie) affiche SANS coude marque : la courbe ne s'aplatit
    # nettement a aucun k (verifie que ce n'est pas un bug -- teste :
    # transformation log des variables tres asymetriques, suppression de
    # CA_TOTAL redondant avec NB_CONTRATS x PRIME_MOYENNE -- aucun des deux
    # ne fait apparaitre un coude net). Marquer un "k detecte" ici serait
    # une affirmation que la courbe elle-meme ne soutient pas visuellement
    # -- ce panneau reste donc volontairement sans annotation de k.
    axes[0].plot(list(K_range), inertias, marker='o', color=PASTEL_BLUE,
                 linewidth=2.5, markersize=8)
    axes[0].set_title('Méthode du Coude (Inertie) — pas de coude net observé')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].grid(True, alpha=0.3)

    best_k = list(K_range)[int(np.argmax(silhouettes))]

    axes[1].plot(list(K_range), silhouettes, marker='o', color=PASTEL_PINK,
                 linewidth=2.5, markersize=8)
    axes[1].set_title('Score de Silhouette (plus haut = mieux) — critère retenu')
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Score Silhouette')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=best_k, color='gray', linestyle='--', alpha=0.7)
    axes[1].text(best_k + 0.1, max(silhouettes) * 0.99,
                 f' k={best_k} retenu', color='gray', fontsize=10)

    print(f"   Silhouette (max) : k={best_k} -- retenu comme seul critere de choix de k.")

    plt.tight_layout()

    # Suppression de sécurité pour Windows
    out_path = 'plots/08_elbow_silhouette.png'
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass

    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Plot 8 — Coude & Silhouette généré (k retenu = {best_k}).")
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
    """
    Nom assigne au cluster qui porte le plus, en z-score, le trait le plus
    distinctif -- pas par rang de CA (remarque superviseur : k n'est plus
    fige a 5, et un persona par rang de CA ne correspond plus a la realite
    des clusters une fois k>5). Chaque archetype est reclame par le
    cluster le plus extreme sur SA metrique, par ordre de priorite ; un
    cluster deja reclame ne peut pas etre repris par un archetype suivant.
    Le rang de CA sert seulement a ordonner l'affichage.
    """
    # Risque nomme sur le RATIO SINISTRES/PRIMES AGREGE par cluster (somme
    # des sinistres du cluster / somme de son CA), pas sur une moyenne de
    # ratios individuels ni sur la part de clients ayant eu >=1 sinistre.
    # Teste et ecarte les deux alternatives :
    #  - part de clients avec >=1 sinistre : biaisee par l'exposition (un
    #    client a 10 contrats a mecaniquement plus de chances d'avoir EU un
    #    sinistre qu'un client a 2 contrats, sans etre plus risque par
    #    contrat) -- designait a tort le cluster "Premium" (le plus de
    #    contrats) comme "a risque".
    #  - moyenne des ratios individuels (ou frequence sinistres/contrat) :
    #    un seul client avec un ratio a plusieurs milliers de % (sinistre
    #    exceptionnel isole, voir mae_backend/main.py) ou un tres petit
    #    nombre de contrats peut a lui seul fausser la moyenne du cluster.
    # Le ratio agrege (somme/somme) est la MEME methode, deja validee, que
    # celle utilisee pour le ratio S/P global du portefeuille (62,5%,
    # calibre sur la reference FTUSA reelle -- voir 02_cleaning.py) :
    # pondere naturellement par l'exposition, insensible aux valeurs
    # individuelles extremes.
    client_df = client_df.copy()
    agg_cols = ['CA_TOTAL', 'NB_CONTRATS', 'PRIME_MOYENNE', 'BONUS_MALUS_MOY',
                'DUREE_MOY', 'SEXE_NUM', 'TOTAL_REGLEMENTS']
    if 'CAPITAUX_MOY' in client_df.columns:
        agg_cols.append('CAPITAUX_MOY')
    sums  = client_df.groupby('CLUSTER')[['CA_TOTAL', 'TOTAL_REGLEMENTS']].sum()
    stats = client_df.groupby('CLUSTER')[agg_cols].mean()
    stats['n'] = client_df.groupby('CLUSTER').size()
    stats['LOSS_RATIO'] = sums['TOTAL_REGLEMENTS'] / sums['CA_TOTAL']
    z = (stats.drop(columns='LOSS_RATIO') - stats.drop(columns='LOSS_RATIO').mean()) / stats.drop(columns='LOSS_RATIO').std()

    order = stats['CA_TOTAL'].sort_values(ascending=False).index.tolist()

    # (nom, classement des clusters du plus au moins extreme sur ce critere),
    # par ordre de priorite -- si le premier choix d'un critere est deja
    # reclame par un critere plus prioritaire, on retombe sur le 2e/3e...
    # choix du MEME critere plutot que d'abandonner completement la persona
    # (sinon un cluster gagnant sur 2 criteres a la fois fait disparaitre
    # purement et simplement l'un des deux personas, remplace par un
    # "Segment N" generique alors qu'un autre cluster lui correspond bien).
    def ranked(series):
        return series.sort_values(ascending=False).index.tolist()

    candidates = [
        ('Client a Risque',      ranked(stats['LOSS_RATIO'])),                    # ratio sinistres/primes agrege le plus eleve
        ('Client Premium',       ranked(z['CA_TOTAL'])),                          # CA le plus eleve
        ('Client Grand Contrat', ranked(z['PRIME_MOYENNE'] - z['NB_CONTRATS'])),  # peu de contrats, mais tres chers
        ('Client Capital Eleve', ranked(z['CAPITAUX_MOY'])) if 'CAPITAUX_MOY' in stats.columns else (None, []),
        ('Clientele Feminine',   ranked(-z['SEXE_NUM'])),                        # SEXE_NUM=0 code F -- ecart le plus extreme vers F
        ('Client Fidele',        ranked(z['DUREE_MOY'])),                        # duree de contrat la plus longue
        ('Client Economique',    ranked(z['n'] - z['DUREE_MOY'])),               # segment le plus gros, le plus court en duree
    ]

    name_map, label_map = {}, {}
    claimed = set()
    for nom, ranking in candidates:
        if nom is None:
            continue
        for cluster_id in ranking:
            if cluster_id not in claimed:
                name_map[int(cluster_id)] = nom
                claimed.add(cluster_id)
                break

    generic_rank = 0
    for cluster_id in order:
        if cluster_id not in name_map:
            generic_rank += 1
            name_map[int(cluster_id)] = f'Client Segment {generic_rank}'

    for rank, cluster_id in enumerate(order):
        nom = name_map[int(cluster_id)]
        label_map[int(cluster_id)] = f"Seg_{rank+1}_{nom.split()[-1]}"

    # Couleur assignee par RANG de CA (pas par cluster_id brut) -- garantit
    # que "Client Premium" a toujours la meme couleur dans le plot PCA et le
    # plot des profils, plutot que 2 couleurs differentes pour le meme segment.
    color_map = {int(cluster_id): PALETTE[rank % len(PALETTE)] for rank, cluster_id in enumerate(order)}
    return name_map, label_map, color_map


# =================================================================
# 5. VISUALISATION PCA 2D
# =================================================================
def plot_pca(X_scaled, labels, k, name_map, color_map):
    pca   = PCA(n_components=2, random_state=42)
    # Pour accélérer l'affichage des 72k points, on sous-échantillonne à 50k points max pour le scatter plot
    plot_sample_size = min(50000, len(X_scaled))
    np.random.seed(42)
    indices = np.random.choice(len(X_scaled), size=plot_sample_size, replace=False)

    X_scaled_sampled = X_scaled[indices]
    labels_sampled = labels[indices]

    X_pca = pca.fit_transform(X_scaled_sampled)
    explained = pca.explained_variance_ratio_ * 100
    total_explained = explained.sum()

    # Centroides calcules sur TOUTES les donnees (pas l'echantillon d'affichage),
    # puis projetes avec le meme PCA -- plus fiables que la moyenne des points affiches.
    centroids_scaled = np.array([X_scaled[labels == c].mean(axis=0) for c in range(k)])
    centroids_pca = pca.transform(centroids_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster_id in range(k):
        mask  = labels_sampled == cluster_id
        label = name_map.get(cluster_id, f'Segment {cluster_id+1}')
        color = color_map.get(cluster_id, PALETTE[cluster_id % len(PALETTE)])
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=label, alpha=0.35, s=12, edgecolors='none')
        ax.scatter(centroids_pca[cluster_id, 0], centroids_pca[cluster_id, 1],
                   c=color, s=260, marker='X', edgecolors='black', linewidths=1.3, zorder=5)

    # Titre honnete sur la part de variance reellement capturee en 2D (souvent
    # partielle avec 7-8 features) -- pas presente comme une vue complete.
    ax.set_title(
        f"Segmentation Clients — Projection PCA 2D ({total_explained:.0f}% de la variance totale)",
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
# 5b. VISUALISATION PCA 3D
# =================================================================
def plot_pca_3d(X_scaled, labels, k, name_map, color_map):
    pca = PCA(n_components=3, random_state=42)
    plot_sample_size = min(50000, len(X_scaled))
    np.random.seed(42)
    indices = np.random.choice(len(X_scaled), size=plot_sample_size, replace=False)

    X_scaled_sampled = X_scaled[indices]
    labels_sampled = labels[indices]

    X_pca = pca.fit_transform(X_scaled_sampled)
    explained = pca.explained_variance_ratio_ * 100
    total_explained = explained.sum()

    centroids_scaled = np.array([X_scaled[labels == c].mean(axis=0) for c in range(k)])
    centroids_pca = pca.transform(centroids_scaled)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_id in range(k):
        mask  = labels_sampled == cluster_id
        label = name_map.get(cluster_id, f'Segment {cluster_id+1}')
        color = color_map.get(cluster_id, PALETTE[cluster_id % len(PALETTE)])
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=color, label=label, alpha=0.3, s=10, edgecolors='none')
        ax.scatter(centroids_pca[cluster_id, 0], centroids_pca[cluster_id, 1], centroids_pca[cluster_id, 2],
                   c=color, s=220, marker='X', edgecolors='black', linewidths=1.2, depthshade=False)

    ax.set_title(
        f"Segmentation Clients — Projection PCA 3D ({total_explained:.0f}% de la variance totale)",
        fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel(f'Composante 1 ({explained[0]:.1f}%)', fontsize=9)
    ax.set_ylabel(f'Composante 2 ({explained[1]:.1f}%)', fontsize=9)
    ax.set_zlabel(f'Composante 3 ({explained[2]:.1f}%)', fontsize=9)
    ax.legend(fontsize=9, markerscale=2, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()

    out_path = 'plots/09b_clustering_pca_3d.png'
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Plot 9b — PCA 3D généré.")


# =================================================================
# 6. PROFIL DE CHAQUE SEGMENT
# =================================================================
def plot_segment_profiles(client_df, k, name_map, color_map):
    client_df = client_df.copy()
    client_df['SEG_LABEL'] = client_df['CLUSTER'].map(
        lambda c: name_map.get(int(c), f'Seg {c+1}')
    )
    order_ids = sorted(
        name_map.keys(),
        key=lambda c: client_df[client_df['CLUSTER'] == c]['CA_TOTAL'].mean(),
        reverse=True
    )
    order = [name_map[i] for i in order_ids]
    # Meme couleur par segment que dans le plot PCA (voir color_map).
    colors_ordered = [color_map[i] for i in order_ids]

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
            color=colors_ordered, edgecolor='white'
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
                   'AGE', 'BONUS_MALUS_MOY', 'NB_SINISTRES', 'TOTAL_REGLEMENTS']
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

    # k choisi par silhouette (voir plot_elbow) -- pas de bornage manuel a
    # une plage "actionnable" fixee a l'avance (remarque superviseur).
    k = plot_elbow(X_scaled)
    print(f"\n🎯 Nombre de clusters retenu (silhouette) : k={k}")

    # Entraînement
    model, labels = train_kmeans(X_scaled, k)
    client_df['CLUSTER'] = labels

    # Mapping cluster → nom métier + couleur (basé sur profil, voir fonction)
    name_map, label_map, color_map = build_cluster_name_map(client_df, k)

    # Visualisations
    plot_pca(X_scaled, labels, k, name_map, color_map)
    plot_pca_3d(X_scaled, labels, k, name_map, color_map)
    plot_segment_profiles(client_df, k, name_map, color_map)

    # Résumé et export
    print_summary(client_df, k, name_map)
    export_results(client_df, name_map)

    print("\n✅ TERMINÉ — plots/08, 09, 09b, 10 + outputs/segments_clients.csv")