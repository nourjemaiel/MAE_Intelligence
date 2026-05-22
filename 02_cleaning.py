import pandas as pd
import mlflow
import os

# =================================================================
# 1. CONFIGURATION MLOPS
# =================================================================
mlflow.set_experiment("MAE_PFE_Final_Sync_2026")


def clean_augment_shift(file_path, output_name):
    if not os.path.exists(file_path):
        print(f"❌ Fichier introuvable : {file_path}")
        return

    try:
        with mlflow.start_run(run_name=f"Process_{output_name}", nested=True):
            print(f"\n--- ⚙️ Traitement de : {output_name} ---")

            # ----------------------------------------------------------------
            # 1. CHARGEMENT
            # ----------------------------------------------------------------
            df = pd.read_csv(file_path, sep=';', encoding='latin-1', low_memory=False)
            df = df.drop_duplicates()
            print(f"📥 Chargement : {len(df)} lignes uniques.")

            # ----------------------------------------------------------------
            # 2. AUGMENTATION (+20%)
            # ----------------------------------------------------------------
            extra = df.sample(frac=0.20, random_state=42)
            df = pd.concat([df, extra], ignore_index=True)
            print(f"🚀 Augmentation terminée : {len(df)} lignes au total.")

            # ----------------------------------------------------------------
            # 3. NETTOYAGE DES TYPES AVANT TOUT MAPPING
            #    Convertir les colonnes numériques mal lues comme float→int string
            # ----------------------------------------------------------------
            for col in ['N_BRA', 'AGENCE', 'N_CLIENT', 'N_POLICE', 'N_SINISTRE', 'C_GARA', 'N_BRA', 'CSP', 'BONUS_MALUS']:
                if col in df.columns:
                    # Supprimer le ".0" parasite introduit par pandas lors de la lecture
                    df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

            # ----------------------------------------------------------------
            # 4. MAPPING N_BRA → libellé (remplace la colonne en place)
            # ----------------------------------------------------------------
            if 'N_BRA' in df.columns:
                mapping_mae = {
                    '11':  'Tourisme',
                    '21':  'Transport_Prive_Inf_3.5T',
                    '22':  'Transport_Prive_Sup_3.5T',
                    '31':  'Transport_Public_Inf_3.5T',
                    '32':  'Transport_Public_Sup_3.5T',
                    '41':  'Taxi',
                    '42':  'Louage',
                    '43':  'Taxi_Plus_4_places',
                    '44':  'Taxi_Collectif',
                    '51':  'Transport_Personnel',
                    '52':  'Transport_Hotel_Agence',
                    '63':  'Transport_Agricole_Inf_3.5T',
                    '64':  'Transport_Agricole_Sup_3.5T',
                    '71':  'Auto_Ecole_Tourisme',
                    '72':  'Auto_Ecole_Utilitaire',
                    '91':  'Engin_Agricole_Prive',
                    '92':  'Engin_Agricole_Etablissement',
                    '93':  'Engin_de_Chantier',
                    '94':  'Transport_Rural',
                    '101': 'Ambulance',
                    '111': '2_Roues',
                    '116': '2_Roues',
                }
                df['N_BRA'] = df['N_BRA'].map(mapping_mae).fillna('Autre_ou_Inconnu')
                print(f"✅ N_BRA décodé en libellé (en place).")

            # ----------------------------------------------------------------
            # 5. MAPPING AGENCE → nom ville (remplace la colonne en place)
            # ----------------------------------------------------------------
            noms_villes = [
                "Tunis Centre", "Ariana", "Ben Arous", "Bizerte", "Béja",
                "Sousse", "Sfax Ville", "Monastir", "Nabeul", "Gabès"
            ]
            if 'AGENCE' in df.columns:
                unique_codes = sorted(df['AGENCE'].unique())
                map_agt = {code: noms_villes[i % len(noms_villes)] for i, code in enumerate(unique_codes)}
                df['AGENCE'] = df['AGENCE'].map(map_agt)
                print(f"✅ AGENCE décodée en nom ville (en place, {len(unique_codes)} agences).")

           
            # ----------------------------------------------------------------
            # 7. TIME SHIFT SÉLECTIF (décalage +4 ans vers 2025-2026)
            # ----------------------------------------------------------------
            date_columns_detected = []
            for col in df.columns:
                # Sauter les colonnes déjà converties ou non-objet
                if col in date_columns_detected:
                    continue

                is_date_col = False
                if df[col].dtype == 'object':
                    sample_vals = df[col].dropna()
                    if not sample_vals.empty:
                        sample = str(sample_vals.iloc[0])
                        # Heuristique : contient '/' ou '-' avec longueur typique d'une date
                        if ('/' in sample or '-' in sample) and 6 <= len(sample) <= 11:
                            is_date_col = True

                if is_date_col or pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Préserver les dates de naissance
                    if "NAISS" in col.upper():
                        print(f"🛡️  Colonne '{col}' ignorée (Date de Naissance préservée).")
                        continue

                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                    df[col] = df[col].fillna(pd.Timestamp('2021-01-01'))
                    df[col] = df[col] + pd.DateOffset(years=4)   # +4 ans exact (pas ~1461 jours)
                    date_columns_detected.append(col)
                    print(f"📅 Colonne '{col}' décalée → max {df[col].dt.year.max()}")

            # ----------------------------------------------------------------
            # 8. NETTOYAGE DES MONTANTS
            # ----------------------------------------------------------------
            montants = ['CAPITAUX', 'REGLEMENTS', 'SAP', 'PRIME_NETTE']
            for col in montants:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.', regex=False),
                            errors='coerce'
                        ).fillna(0)
                    )
            print(f"💰 Montants nettoyés : {[c for c in montants if c in df.columns]}")

            # ----------------------------------------------------------------
            # 9. SUPPRESSION DES DOUBLONS DE COLONNES
            #    Sécurité : si une colonne apparaît deux fois après concat/merge
            # ----------------------------------------------------------------
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"🔍 Colonnes finales ({len(df.columns)}) : {list(df.columns)}")

            # ----------------------------------------------------------------
            # 10. VALIDATION RAPIDE
            # ----------------------------------------------------------------
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if not cols_with_nulls.empty:
                print(f"⚠️  Valeurs nulles restantes :\n{cols_with_nulls}")

            # ----------------------------------------------------------------
            # 11. SAUVEGARDE
            # ----------------------------------------------------------------
            output_dir = "processed_data"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{output_name}_Cleaned.csv")
            df.to_csv(save_path, index=False, encoding='utf-8')

            mlflow.log_metric("total_lignes", float(len(df)))
            mlflow.log_metric("total_colonnes", float(len(df.columns)))
            mlflow.log_param("colonnes", str(list(df.columns)))
            print(f"\n💾 Succès : '{save_path}' généré ({len(df)} lignes, {len(df.columns)} colonnes).")

    except Exception as e:
        print(f"❌ Erreur sur {output_name} : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    clean_augment_shift(os.path.join(BASE_DIR, "raw_data", "Base Production.csv"), "Production")
    clean_augment_shift(os.path.join(BASE_DIR, "raw_data", "Base Sinistres.csv"), "Sinistres")