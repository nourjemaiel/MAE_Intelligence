import pandas as pd
import mlflow
import os
import math

# =================================================================
# 1. CONFIGURATION MLOPS
# =================================================================
mlflow.set_experiment("MAE_PFE_Final_Sync_2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================
# TABLE REGION → LOCALITE TUNISIENNE
# =================================================================
# Il n'existe pas de table officielle code -> région/localité (données
# synthétiques, consigne du superviseur de stage : décoder nous-mêmes).
# Convention retenue : les codes distincts sont triés puis répartis en 5
# macro-régions de taille égale (Grand Tunis, Nord, Centre, Sud, Sahel),
# et chaque code reçoit en plus le nom d'une localité tunisienne réelle et
# DISTINCTE appartenant à ce bucket, pour éviter d'afficher le même
# libellé de macro-région des dizaines de fois dans les filtres/graphes.
# C'est un choix de présentation assumé et documenté, pas une déduction
# géographique réelle. Décodé UNE SEULE FOIS ici (source de vérité) plutôt
# que recalculé à chaque requête dans l'API.
REGIONS_MACRO = ["Grand Tunis", "Nord", "Centre", "Sud", "Sahel"]

TUNISIA_LOCALITIES = {
    "Grand Tunis": [
        "Tunis","Ariana","La Marsa","Carthage","Le Bardo","Ben Arous",
        "Ezzahra","Mornag","Hammam Lif","Hammam Chott","Rades","Megrine",
        "La Goulette","Manouba","Den Den","Oued Ellil","Douar Hicher",
        "El Mourouj","Mnihla","Sidi Thabet","Kalaat Andalous","La Soukra",
        "Raoued","Sidi Hassine","El Omrane","Bab Souika",
    ],
    "Nord": [
        "Bizerte","Menzel Bourguiba","Mateur","Ras Jebel","Ghar El Melh",
        "Beja","Medjez El Bab","Testour","Nefza","Jendouba","Tabarka",
        "Ain Draham","Fernana","Bou Salem","Le Kef","Tajerouine",
        "Kalaat Senan","Sakiet Sidi Youssef","Siliana","Bargou","Gaafour",
        "Zaghouan","El Fahs","Nabeul","Hammamet","Kelibia",
    ],
    "Sahel": [
        "Sousse","Msaken","Kalaa Kebira","Kalaa Seghira","Hammam Sousse",
        "Akouda","Enfidha","Bouficha","Monastir","Moknine","Jemmal",
        "Ksar Hellal","Ksibet El Mediouni","Sayada","Bembla","Mahdia",
        "Ksour Essef","El Jem","Chebba","Melloulech","Souassi","Chorbane",
        "Bou Merdes","Ouled Chamekh","Hbira",
    ],
    "Centre": [
        "Kairouan","Sbikha","Haffouz","Nasrallah","Chebika","El Alaa",
        "Bouhajla","Oueslatia","Hajeb El Ayoun","Menzel Mehiri",
        "Sidi Bouzid","Regueb","Menzel Bouzaiane","Mezzouna","Bir El Hafey",
        "Ouled Haffouz","Meknassy","Souk Jedid","Kasserine","Sbeitla",
        "Feriana","Thala","Hassi El Ferid","Foussana","Jedelienne","El Ayoun",
    ],
    "Sud": [
        "Sfax","Sakiet Eddaier","Sakiet Ezzit","Jebeniana","Mahres","El Amra",
        "Bir Ali Ben Khalifa","Menzel Chaker","Gabes","Ghannouch","Mareth",
        "Metouia","El Hamma","Medenine","Ben Gardane","Zarzis","Houmt Souk",
        "Midoun","Ajim","Tataouine","Remada","Ghomrassen","Bir Lahmar",
        "Kebili","Douz","Souk Lahad","Tozeur","Nefta","Degache","Gafsa",
        "Metlaoui","Redeyef",
    ],
}


def build_region_label_map(codes):
    """
    Construit le dictionnaire {code_brut: "Nom localité"} pour TOUS les
    codes région distincts trouvés dans le fichier. Convention assumée
    (voir docstring du module) : buckets de taille égale par macro-région,
    puis une localité réelle distincte par code au sein de chaque bucket.
    """
    def sort_key(c):
        try:
            return (0, float(c))
        except (ValueError, TypeError):
            return (1, str(c))

    codes_sorted = sorted({str(c) for c in codes if c is not None and str(c).lower() != "nan"}, key=sort_key)
    n = len(codes_sorted)
    if n == 0:
        return {}

    buckets = len(REGIONS_MACRO)
    size = math.ceil(n / buckets)

    labels = {}
    for i, code in enumerate(codes_sorted):
        bucket_idx = min(i // size, buckets - 1)
        macro = REGIONS_MACRO[bucket_idx]
        localities = TUNISIA_LOCALITIES[macro]
        pos_in_bucket = i - bucket_idx * size
        locality = localities[pos_in_bucket % len(localities)]
        if pos_in_bucket >= len(localities):
            locality = f"{locality} {pos_in_bucket // len(localities) + 1}"
        labels[code] = locality
    return labels


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
            #    "Region" ajoutée ici : ses codes bruts subissent le même
            #    problème de ".0" parasite que N_BRA/AGENCE lors de la
            #    lecture pandas, ce qui casserait le mapping ci-dessous si
            #    on ne le nettoie pas AVANT de construire les buckets.
            # ----------------------------------------------------------------
            for col in ['N_BRA', 'AGENCE', 'N_CLIENT', 'N_POLICE', 'N_SINISTRE',
                        'C_GARA', 'CSP', 'BONUS_MALUS', 'Region']:
                if col in df.columns:
                    # Supprimer le ".0" parasite introduit par pandas lors de la lecture
                    df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

            # ----------------------------------------------------------------
            # 4-6. DECODAGE N_BRA / AGENCE / Region → libelles lisibles
            # ----------------------------------------------------------------
            # POURQUOI decoder ici plutot que de garder les codes bruts :
            # ces libelles servent la couche presentation/interaction (filtres
            # et graphes du dashboard, filtres en langage naturel de l agent —
            # ex. "genere un rapport pour Sfax Ville"), PAS l entrainement des
            # modeles. Verifie explicitement : 04_forecasting.py n utilise que
            # PRIME_NETTE + DEBUT_PERI (aucune reference a AGENCE/BRANCHE/
            # Region) ; 05_clustering.py ne les inclut pas non plus dans sa
            # liste features (NB_CONTRATS, CA_TOTAL, PRIME_MOYENNE,
            # BONUS_MALUS_MOY, AGE, DUREE_MOY, SEXE_NUM). Donc pas de conflit
            # avec le besoin d un modele en valeurs numeriques : si un futur
            # modele doit un jour utiliser branche/agence/region comme
            # predicteur, on ré-encode a ce moment-la (one-hot/label encoding)
            # a partir du libelle -- decoder une fois pour l humain puis
            # ré-encoder plus tard pour un modele specifique est une pratique
            # standard, pas une contradiction.
            #
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
            # 6. MAPPING REGION → localité tunisienne (remplace la colonne en place)
            #    Décodé UNE SEULE FOIS ici (source de vérité pour tout le
            #    pipeline : EDA, forecasting, clustering, API). L'API ne
            #    doit plus recalculer ce mapping à partir d'un échantillon
            #    en mémoire — voir main.py.
            # ----------------------------------------------------------------
            if 'Region' in df.columns:
                unique_region_codes = df['Region'].dropna().unique().tolist()
                region_map = build_region_label_map(unique_region_codes)
                df['Region'] = df['Region'].map(region_map).fillna('Region_Inconnue')
                print(f"✅ Region décodée en localité tunisienne (en place, {len(unique_region_codes)} régions).")
            else:
                print("⚠️  Colonne 'Region' absente de ce fichier — décodage ignoré.")

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
            # 8. NETTOYAGE DES MONTANTS + CONVERSION DES UNITÉS FINANCIÈRES
            # ----------------------------------------------------------------
            # METHODOLOGIE DU CHOIX DU DIVISEUR (revise -- l'ancienne
            # justification "ca correspond au CA global attendu" etait
            # circulaire : caler un diviseur sur un total agrege qu'on
            # attend deja n'est pas une preuve, c'est ajuster la reponse a
            # la question. Aucune documentation officielle MAE ne precise
            # l'unite brute des montants (pas de dictionnaire de donnees
            # fourni), donc a defaut on valide chaque diviseur candidat
            # contre des ORDRES DE GRANDEUR REALISTES du marche tunisien de
            # l'assurance auto, INDEPENDAMMENT sur les deux fichiers :
            #
            #   Diviseur   PRIME_NETTE (mediane)   REGLEMENTS (mediane)
            #      1         49 000 TND/an           370 000 TND/sinistre
            #    100            490 TND/an             3 700 TND/sinistre
            #   1000             49 TND/an               370 TND/sinistre
            #
            #   - /1    : une prime annuelle de 49 000 TND (~16 000 USD) est
            #     absurde -- superieure au prix de la plupart des vehicules
            #     assures. Ecarte.
            #   - /1000 : une prime de 49 TND/an (~16 USD) est irrealiste,
            #     bien en-dessous du cout reel d'une assurance meme basique
            #     en Tunisie. Ecarte.
            #   - /100  : 490 TND/an de prime et 3 700 TND de reglement
            #     median sont tous deux coherents avec des montants reels
            #     observes sur le marche tunisien -- et ce sur les DEUX
            #     fichiers INDEPENDAMMENT (primes ET sinistres), ce qui
            #     suggere une convention d'unite brute commune plutot qu'une
            #     coincidence. Retenu.
            #
            # Limite assumee : sans acces au systeme source MAE, ceci reste
            # une validation par plausibilite economique, pas une
            # confirmation documentaire de l'unite brute exacte. C'est
            # neanmoins une methode reproductible et falsifiable (voir
            # sanity check imprime plus bas), contrairement au calage sur
            # un total agrege.
            RAW_TO_DINARS = 100.0

            montants = ['CAPITAUX', 'REGLEMENTS', 'SAP', 'PRIME_NETTE']
            for col in montants:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.', regex=False),
                            errors='coerce'
                        ).fillna(0)
                    ) / RAW_TO_DINARS
            print(f"💰 Montants nettoyés et ajustés (/{RAW_TO_DINARS:.0f}) : {[c for c in montants if c in df.columns]}")

            # Sanity check IMPRIME (pas seulement un commentaire) : verifie a
            # chaque execution que le diviseur choisi produit toujours des
            # ordres de grandeur plausibles -- alerte si une future version
            # des donnees brutes rendrait ce choix caduc.
            if 'PRIME_NETTE' in df.columns:
                prime_med = df['PRIME_NETTE'].median()
                if not (100 <= prime_med <= 3000):
                    print(f"⚠️  ATTENTION : mediane PRIME_NETTE = {prime_med:,.0f} TND, "
                          f"hors de la plage plausible [100, 3000] TND/an -- revalider RAW_TO_DINARS.")
            if 'REGLEMENTS' in df.columns:
                regl_med = df['REGLEMENTS'].median()
                if regl_med > 0 and not (500 <= regl_med <= 50000):
                    print(f"⚠️  ATTENTION : mediane REGLEMENTS = {regl_med:,.0f} TND, "
                          f"hors de la plage plausible [500, 50000] TND/sinistre -- revalider RAW_TO_DINARS.")

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

            if 'PRIME_NETTE' in df.columns:
                print(f"🔎 Sanity check PRIME_NETTE : min={df['PRIME_NETTE'].min():,.2f}  "
                      f"mean={df['PRIME_NETTE'].mean():,.2f}  max={df['PRIME_NETTE'].max():,.2f}  "
                      f"(en TND, après division /{RAW_TO_DINARS:.0f})")

            # ----------------------------------------------------------------
            # 11. SAUVEGARDE
            #    Sécurité : ancré sur BASE_DIR pour éviter les doublons de répertoires
            #    "processed_data" d'un script à un autre.
            # ----------------------------------------------------------------
            output_dir = os.path.join(BASE_DIR, "processed_data")
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
    clean_augment_shift(os.path.join(BASE_DIR, "raw_data", "Base Production.csv"), "Production")
    clean_augment_shift(os.path.join(BASE_DIR, "raw_data", "Base Sinistres.csv"), "Sinistres")