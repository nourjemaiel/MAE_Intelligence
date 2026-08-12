import pandas as pd
import mlflow
from ydata_profiling import ProfileReport
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# 1. CONFIGURATION DU PROJET
# =================================================================
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_PFE_Initial_Diagnostic")

RAW_DATA_FOLDER = "raw_data" 
FILE_PROD = "Base Production.csv"
FILE_SINI = "Base Sinistres.csv"

# =================================================================
# 2. FONCTION DE DIAGNOSTIC
# =================================================================
def run_diagnostic(file_path, report_name):
    if not os.path.exists(file_path):
        print(f"❌ Erreur : Le fichier est introuvable : {file_path}")
        return

    with mlflow.start_run(run_name=f"Diag_{report_name}"):
        print(f"\n--- 🔍 Analyse en cours : {report_name} ---")
        
        # --- LECTURE ---
        try:
            df = pd.read_csv(file_path, sep=';', encoding='latin-1')
            if df.shape[1] <= 1:
                df = pd.read_csv(file_path, sep=',', encoding='latin-1')
        except Exception as e:
            print(f"❌ Erreur de lecture : {e}")
            return

        # --- CORRECTION DES TYPES ---
        if 'CAPITAUX' in df.columns:
            df['CAPITAUX'] = pd.to_numeric(df['CAPITAUX'].astype(str).str.replace(',', '.'), errors='coerce')

        # BONUS_MALUS est un coefficient de risque ORDINAL/NUMERIQUE (traite
        # comme numerique partout ailleurs dans le pipeline -- BONUS_MALUS_MOY
        # dans 05_clustering.py), pas une categorie nominale. Le forcer en
        # string ici l'excluait silencieusement de la matrice de correlation
        # et du profil numerique ydata-profiling -- corrige : converti en
        # numerique comme CAPITAUX, retire de cols_categoriques ci-dessous.
        if 'BONUS_MALUS' in df.columns:
            df['BONUS_MALUS'] = pd.to_numeric(df['BONUS_MALUS'], errors='coerce')

        cols_dates = ['DEBUT_PERI', 'FIN_PERIOD', 'DT_NAISS', 'DATE_ACCIDENT']
        for col in cols_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

        cols_categoriques = ['N_POLICE', 'N_CLIENT', 'C_GARA', 'Region', 'N_BRA', 'AGENCE', 'SEXE']
        for col in cols_categoriques:
            if col in df.columns:
                df[col] = df[col].astype(str)

        print("✅ Types corrigés.")

        # --- MLOps : Metrics ---
        mlflow.log_metric("nombre_lignes", float(df.shape[0]))
        mlflow.log_metric("nombre_colonnes", float(df.shape[1]))

        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)

        # --- ÉCHANTILLONNAGE POUR ÉVITER LE CRASH ---
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)

        # =================================================================
        # 2.7 ANALYSE DE CORRÉLATION MANUELLE (LÉGÈRE)
        # =================================================================
        print(f"📈 Calcul de la matrice de corrélation...")
        # On ne garde que les colonnes numériques pour la corrélation
        df_numeric = df_sample.select_dtypes(include=['float64', 'int64'])
        
        if not df_numeric.empty:
            plt.figure(figsize=(12, 8))
            corr_matrix = df_numeric.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"Corrélations Numériques - {report_name}")
            
            # Sauvegarde de l'image
            corr_img_path = os.path.join(output_dir, f"correlation_{report_name}.png")
            plt.savefig(corr_img_path)
            plt.close() # Libère la mémoire
            
            # MLOps : Log de l'image
            mlflow.log_artifact(corr_img_path)
            print(f"✅ Matrice de corrélation sauvegardée.")

        # --- GÉNÉRATION DU RAPPORT (MINIMAL) ---
        print(f"📊 Génération du profil statistique (Mode Minimal)...")
        profile = ProfileReport(df_sample, title=f"Diagnostic MAE - {report_name}", minimal=True)
        
        report_path = os.path.join(output_dir, f"rapport_{report_name}.html")
        profile.to_file(report_path)
        mlflow.log_artifact(report_path)
        
        print(f"✅ Rapport HTML disponible : {report_path}")

# =================================================================
# 3. EXÉCUTION
# =================================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH_PROD = os.path.join(BASE_DIR, RAW_DATA_FOLDER, FILE_PROD)
    PATH_SINI = os.path.join(BASE_DIR, RAW_DATA_FOLDER, FILE_SINI)

    run_diagnostic(PATH_PROD, "Production_Auto")
    run_diagnostic(PATH_SINI, "Sinistres")

    print("\n🚀 TERMINÉ : Vérifie le dossier 'reports' ou lance 'mlflow ui'")