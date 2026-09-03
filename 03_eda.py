import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# =================================================================
# CHEMINS ABSOLUS (Sécurité d'exécution)
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# =================================================================
# CONFIGURATION VISUELLE
# =================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
PASTEL_BLUE   = '#AEC6CF'
PASTEL_PINK   = '#FFB7C5'
PASTEL_PURPLE = '#C3B1E1'
PASTEL_GREEN  = '#B5EAD7'
PASTEL_ORANGE = '#FFDAB9'
PALETTE = [PASTEL_BLUE, PASTEL_PINK, PASTEL_PURPLE, PASTEL_GREEN, PASTEL_ORANGE,
           '#FFFACD', '#D4F1F4', '#E8D5B7', '#F0E6EF', '#C7CEEA']


# =================================================================
# CHARGEMENT DES DONNÉES
# =================================================================
def load_data():
    path_prod = os.path.join(BASE_DIR, "processed_data", "Production_Cleaned.csv")
    path_sini = os.path.join(BASE_DIR, "processed_data", "Sinistres_Cleaned.csv")

    if not os.path.exists(path_prod):
        print(f"❌ Fichier introuvable : {path_prod}")
        return None, None

    df_prod = pd.read_csv(path_prod, low_memory=False)
    df_sini = pd.read_csv(path_sini, low_memory=False) if os.path.exists(path_sini) else None

    # Conversion dates
    if 'DEBUT_PERI' in df_prod.columns:
        df_prod['DEBUT_PERI'] = pd.to_datetime(df_prod['DEBUT_PERI'], errors='coerce')
    if df_sini is not None and 'DATE_ACCIDENT' in df_sini.columns:
        df_sini['DATE_ACCIDENT'] = pd.to_datetime(df_sini['DATE_ACCIDENT'], errors='coerce')

    # CORRIGE (meme constat que 04_forecasting.py, remarque 2 superviseur) :
    # un filtre par ANNEE CALENDAIRE fixe ("target_year = 2025") ne correspond
    # plus a rien depuis le decalage de dates DYNAMIQUE de 02_cleaning.py --
    # la periode dense de ~12 mois de donnees s'etale desormais sur DEUX
    # annees calendaires (ex: aout 2025 a juillet 2026), donc filtrer sur
    # UNE SEULE annee ne recuperait qu'une partie de la periode dense et
    # laissait passer la longue traine eparse (0.05% du volume, artefact
    # d'export) qui la precede -- observe concretement : plot 3 (evolution
    # mensuelle) montrait un CA quasi nul de janvier a juillet puis un saut
    # brutal en aout, un artefact de donnees et non une tendance reelle.
    # Meme detection DONNEES-DRIVEE que 04_forecasting.py : on isole les
    # semaines a volume de transactions substantiel, sans presupposer
    # quelle(s) annee(s) calendaire(s) elles couvrent.
    weekly_counts = df_prod.groupby(df_prod['DEBUT_PERI'].dt.to_period('W')).size()
    dense_weeks = weekly_counts[weekly_counts > 100].index
    df_prod_filtered = df_prod[df_prod['DEBUT_PERI'].dt.to_period('W').isin(dense_weeks)].copy()

    if len(df_prod_filtered) == 0:
        print("❌ Erreur critique : aucune semaine a volume substantiel trouvee dans DEBUT_PERI.")
        return None, None

    period_start = df_prod_filtered['DEBUT_PERI'].min()
    period_end   = df_prod_filtered['DEBUT_PERI'].max()
    print(f"✅ Données chargées : {len(df_prod_filtered):,} contrats "
          f"(période dense {period_start:%Y-%m} à {period_end:%Y-%m})")

    # Sinistres alignes sur la MEME plage de dates que la periode dense
    # ci-dessus (pas une egalite sur une seule annee calendaire, pour la
    # meme raison) -- necessaire pour plot_sinistres_vs_primes.
    if df_sini is not None and 'DATE_ACCIDENT' in df_sini.columns:
        df_sini = df_sini[df_sini['DATE_ACCIDENT'].between(period_start, period_end)].copy()

    return df_prod_filtered, df_sini


# =================================================================
# PLOT 1 — TOP 10 AGENCES PAR CA
# =================================================================
def plot_top_agences(df):
    col = 'AGENCE'
    top = df.groupby(col)['PRIME_NETTE'].sum().sort_values(ascending=True).tail(10)
    # Periode dynamique (voir load_data()) plutot qu'une annee fixe "2025" --
    # la periode dense peut s'etaler sur 2 annees calendaires.
    period_label = f"{df['DEBUT_PERI'].min():%b %Y} – {df['DEBUT_PERI'].max():%b %Y}"

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(top.index, top.values, color=PASTEL_BLUE, edgecolor='white')

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + top.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.2f}M TND', va='center', fontsize=11, color='#444')

    ax.set_title(f"Top 10 des Agences — Chiffre d'Affaires ({period_label})",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Revenus (TND)', fontsize=12)
    # CORRIGE : ':.0f' arrondissait tous les ticks a "0M" (valeurs toutes
    # sous 0.5M TND) -- axe illisible malgre des etiquettes par barre
    # correctes en ':.2f'. Alignee sur la meme precision.
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.2f}M'))
    ax.set_xlim(0, top.values.max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_top_agences_ca.png'), dpi=150)
    plt.close()
    print("✅ Plot 1 — Top Agences généré.")


# =================================================================
# PLOT 2 — RÉPARTITION PAR BRANCHE
# =================================================================
def plot_repartition_branche(df):
    col = 'N_BRA'
    branche = df[col].value_counts()

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(branche)), bars_y := branche.values,
                  color=PALETTE[:len(branche)], edgecolor='white')

    for bar, val in zip(bars, bars_y):
        pct = val / branche.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bars_y) * 0.01,
                f'{pct:.1f}%', ha='center', fontsize=9, color='#444')

    ax.set_xticks(range(len(branche)))
    ax.set_xticklabels(branche.index, rotation=40, ha='right', fontsize=9)
    ax.set_title('Répartition du Portefeuille par Branche',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Nombre de Contrats')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_repartition_branche.png'), dpi=150)
    plt.close()
    print("✅ Plot 2 — Répartition Branche généré.")


# =================================================================
# PLOT 3 — ÉVOLUTION MENSUELLE CA (2025)
# =================================================================
def plot_evolution_mensuelle(df):
    df = df.copy()
    df['Annee_Mois'] = df['DEBUT_PERI'].dt.to_period('M').astype(str)
    evolution = df.groupby('Annee_Mois')['PRIME_NETTE'].sum().sort_index()
    # Meme exclusion que 04_forecasting.py : le mois a la frontiere de la
    # periode dense est partiel (coupure semaine/mois), pas comparable a un
    # mois complet -- sans ca il apparait a tort comme un quasi-zero.
    evolution = evolution[evolution > 1e6]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(range(len(evolution)), evolution.values, alpha=0.3, color=PASTEL_PURPLE)
    ax.plot(range(len(evolution)), evolution.values,
            marker='o', color=PASTEL_PURPLE, linewidth=3)

    # Annotations sur chaque point
    for i, (mois, val) in enumerate(evolution.items()):
        ax.annotate(f'{val/1e6:.1f}M',
                    xy=(i, val), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, color='#555')

    ax.set_xticks(range(len(evolution)))
    ax.set_xticklabels(evolution.index, rotation=45, ha='right', fontsize=10)
    # CORRIGE (meme bug que plot 1) : ':.0f' arrondissait tous les ticks a
    # "1M TND" (valeurs toutes entre 1.0M et 1.5M).
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M TND'))
    period_label = f"{df['DEBUT_PERI'].min():%b %Y} – {df['DEBUT_PERI'].max():%b %Y}"
    ax.set_title(f"Évolution du Chiffre d'Affaires Mensuel ({period_label})",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Primes Nettes (TND)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_tendance_temporelle.png'), dpi=150)
    plt.close()
    print("✅ Plot 3 — Évolution Mensuelle généré.")


# =================================================================
# PLOT 4 — SINISTRES VS PRIMES PAR MOIS
# =================================================================
def plot_sinistres_vs_primes(df_prod, df_sini):
    if df_sini is None or 'REGLEMENTS' not in df_sini.columns:
        print("⚠️  Sinistres manquants — Plot 4 ignoré.")
        return

    df_prod = df_prod.copy()
    df_sini = df_sini.copy()

    df_prod['Mois'] = df_prod['DEBUT_PERI'].dt.to_period('M').astype(str)
    # df_sini est deja aligne sur la meme plage de dates que df_prod par
    # load_data() -- ne pas re-filtrer sur une seule annee calendaire ici
    # (l'ancien filtre perdait la moitie de la periode dense, voir load_data()).
    df_sini['DATE_ACCIDENT'] = pd.to_datetime(df_sini['DATE_ACCIDENT'], errors='coerce')
    df_sini['Mois'] = df_sini['DATE_ACCIDENT'].dt.to_period('M').astype(str)

    primes    = df_prod.groupby('Mois')['PRIME_NETTE'].sum()
    sinistres = df_sini.groupby('Mois')['REGLEMENTS'].sum()

    combined = pd.DataFrame({'Primes': primes, 'Sinistres': sinistres}).dropna().sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(combined))
    width = 0.38
    b1 = ax.bar([i - width/2 for i in x], combined['Primes'],
                width=width, label='Primes Nettes', color=PASTEL_BLUE, edgecolor='white')
    b2 = ax.bar([i + width/2 for i in x], combined['Sinistres'],
                width=width, label='Sinistres Réglés', color=PASTEL_PINK, edgecolor='white')

    # Ratio S/P au-dessus de chaque paire
    for i, (_, row) in enumerate(combined.iterrows()):
        ratio = row['Sinistres'] / row['Primes'] * 100 if row['Primes'] > 0 else 0
        ax.text(i, max(row['Primes'], row['Sinistres']) + combined.max().max() * 0.02,
                f'{ratio:.0f}%', ha='center', fontsize=9, color='#c45c7a', fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(combined.index, rotation=45)
    # CORRIGE (meme bug que plots 1 et 3) : ':.0f' arrondissait tous les
    # ticks a "1M"/"0M" (Sinistres ~0.3-1.4M, Primes ~1.0-1.5M).
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax.set_ylim(0, combined.max().max() * 1.18)
    ax.set_title('Primes Collectées vs Sinistres Réglés par Mois (+ Ratio S/P)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Montant (TND)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_sinistres_vs_primes.png'), dpi=150)
    plt.close()
    print("✅ Plot 4 — Sinistres vs Primes généré.")


# =================================================================
# PLOT 5 — DISTRIBUTION DES PRIMES
# =================================================================
def plot_distribution_primes(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    data = df['PRIME_NETTE'][df['PRIME_NETTE'] < df['PRIME_NETTE'].quantile(0.99)]
    ax.hist(data, bins=50, color=PASTEL_GREEN, edgecolor='white')

    median_val = data.median()
    ax.axvline(median_val, color='#888', linestyle='--', linewidth=1.5)
    ax.text(median_val * 1.02, ax.get_ylim()[1] * 0.9,
            f'Médiane : {median_val:,.0f} TND', color='#555', fontsize=10)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_title('Distribution des Primes Nettes (99ème percentile)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Prime Nette (TND)')
    ax.set_ylabel('Nombre de Contrats')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_distribution_primes.png'), dpi=150)
    plt.close()
    print("✅ Plot 5 — Distribution des Primes généré.")


# =================================================================
# PLOT 6 — PROFIL CLIENT : ÂGE, SEXE, CSP
# =================================================================
def plot_profil_client(df):
    df = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    period_label = f"{df['DEBUT_PERI'].min():%b %Y} – {df['DEBUT_PERI'].max():%b %Y}"
    fig.suptitle(f'Profil des Assurés MAE ({period_label})', fontsize=16, fontweight='bold')

    # --- Âge ---
    if 'DT_NAISS' in df.columns:
        df['DT_NAISS'] = pd.to_datetime(df['DT_NAISS'], errors='coerce')
        # Age a la date de CE contrat, pas a une annee de reference unique
        # (l'ancien code prenait l'annee de la 1ere ligne du fichier, un
        # choix arbitraire qui biaisait l'age de tout contrat souscrit une
        # autre annee -- desormais significatif puisque la periode dense
        # couvre 2 annees calendaires, voir load_data()).
        df['AGE'] = df['DEBUT_PERI'].dt.year - df['DT_NAISS'].dt.year
        age_data = df['AGE'][(df['AGE'] >= 18) & (df['AGE'] <= 80)]
        axes[0].hist(age_data, bins=30, color=PASTEL_BLUE, edgecolor='white')
        axes[0].set_title('Distribution des Âges')
        axes[0].set_xlabel('Âge')
        axes[0].set_ylabel('Nombre de clients')
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    else:
        axes[0].text(0.5, 0.5, 'DT_NAISS\nnon disponible',
                     ha='center', va='center', transform=axes[0].transAxes)

    # --- Sexe ---
    if 'SEXE' in df.columns:
        sexe = df['SEXE'].value_counts()
        axes[1].bar(sexe.index, sexe.values,
                    color=[PASTEL_BLUE, PASTEL_PINK], edgecolor='white')
        for i, (idx, val) in enumerate(sexe.items()):
            axes[1].text(i, val + sexe.max() * 0.02,
                         f'{val/sexe.sum()*100:.1f}%', ha='center', fontsize=11)
        axes[1].set_title('Répartition par Sexe')
        axes[1].set_ylabel('Nombre de clients')
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    else:
        axes[1].text(0.5, 0.5, 'SEXE\nnon disponible',
                     ha='center', va='center', transform=axes[1].transAxes)

    # CSP
    if 'CSP' in df.columns:
        csp = df['CSP'].value_counts().head(6)
        axes[2].barh(csp.index, csp.values, color=PALETTE[:6], edgecolor='white')
        axes[2].set_title('Top 6 Professions (CSP)')
        axes[2].set_xlabel('Nombre de clients')
        axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    else:
        axes[2].text(0.5, 0.5, 'CSP\nnon disponible',
                     ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_profil_client.png'), dpi=150)
    plt.close()
    print("✅ Plot 6 — Profil Client généré.")


# =================================================================
# EXÉCUTION
# =================================================================
if __name__ == "__main__":
    print("🚀 Génération des analyses EDA — MAE Assurances 2025\n")
    
    # ÉTAPE SUPPLÉMENTAIRE : Force le nettoyage physique du dossier plots
    # Cela force Windows à réinitialiser la "Date de création" et détruit le cache.
    target_plots = [
        '01_top_agences_ca.png',
        '02_repartition_branche.png',
        '03_tendance_temporelle.png',
        '04_sinistres_vs_primes.png',
        '05_distribution_primes.png',
        '06_profil_client.png'
    ]
    
    for plot_name in target_plots:
        filepath = os.path.join(PLOTS_DIR, plot_name)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"🧹 Ancien plot supprimé du disque : {plot_name}")
            except Exception as e:
                print(f"⚠️  Impossible de supprimer {plot_name} (fichier peut-être verrouillé par Windows) : {e}")

    print("") # Ligne vide pour la clarté

    df_prod, df_sini = load_data()

    if df_prod is not None:
        plot_top_agences(df_prod)
        plot_repartition_branche(df_prod)
        plot_evolution_mensuelle(df_prod)
        plot_sinistres_vs_primes(df_prod, df_sini)
        plot_distribution_primes(df_prod)
        plot_profil_client(df_prod)

    print("\n✅ TERMINÉ — Tous les graphiques sont dans le dossier 'plots/'")