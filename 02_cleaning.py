import pandas as pd
import mlflow
import os
import math
from datetime import datetime


def compute_dynamic_date_shift(raw_max_date, today=None):
    """
    Calcule le decalage a appliquer aux colonnes de dates pour que
    raw_max_date atterrisse sur le DERNIER JOUR DU MOIS PRECEDENT par
    rapport a AUJOURD'HUI -- recalcule a CHAQUE execution du script, jamais
    fige a une annee fixe.

    POURQUOI (remarque 1c, superviseur) : un decalage fixe en annees
    (l'ancienne version : +4 ans) place le dernier jour d'entrainement sur
    un 31 decembre. Or 31 decembre + 1 mois = janvier de l'annee suivante :
    le tout premier mois "previsionnel" est fige a janvier quelle que soit
    la date reelle d'execution. Des qu'on execute/presente le pipeline
    apres janvier de cette annee-la, une partie des mois "prevus" sont deja
    passes dans la realite -- une prevision qui predit le passe n'est pas
    logique. Aucun decalage en annees entieres ne peut resoudre ca : si les
    donnees brutes se terminent un 31 decembre, le mois suivant sera
    TOUJOURS janvier, qui que ce soit la date reelle d'aujourd'hui.
    Preuve : notons Y l'annee du 31 decembre decale. Il faut A LA FOIS
    31/12/Y <= aujourd'hui (donnees d'entrainement dans le passe) ET
    01/01/(Y+1) > aujourd'hui (le mois previsionnel pas encore arrive).
    La 1ere condition impose Y <= annee(aujourd'hui) ; la 2e impose
    Y >= annee(aujourd'hui). Les deux ne coincident que si aujourd'hui
    tombe EXACTEMENT le 31 decembre -- impossible a garantir a la date
    d'une soutenance ou d'une demo.

    SOLUTION : ne pas se limiter a un decalage en annees entieres. On
    decale de la quantite exacte necessaire pour que la derniere date
    d'entrainement tombe le dernier jour du mois precedant aujourd'hui,
    quel que soit le jour d'execution. Consequence acceptee : le mois
    calendaire associe a chaque transaction n'est plus forcement le meme
    qu'avant le decalage -- acceptable ici car l'ensemble du schema de
    dates est deja synthetique (aucune saisonnalite tunisienne reelle
    n'est encodee dans le choix du mois d'origine), documente depuis le
    depart. Ce qui compte est prevu par le pipeline (04_forecasting.py) :
    la prevision demarre toujours "le mois prochain" par rapport a
    aujourd'hui, plus jamais un mois deja ecoule -- vrai a CHAQUE execution,
    pas seulement au moment ou ce code a ete ecrit.
    """
    if today is None:
        today = pd.Timestamp(datetime.now().date())
    first_of_this_month = pd.Timestamp(today.year, today.month, 1)
    target_max_date = first_of_this_month - pd.Timedelta(days=1)  # dernier jour du mois precedent
    return target_max_date - raw_max_date

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


def clean_augment_shift(file_path, output_name, date_shift):
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
            # 7. TIME SHIFT SÉLECTIF (décalage DYNAMIQUE -- voir
            #    compute_dynamic_date_shift() : recalcule a chaque execution
            #    pour que l'entrainement se termine "le mois dernier" et la
            #    prevision demarre "le mois prochain" par rapport a
            #    aujourd'hui, quelle que soit la date reelle d'execution.
            #    Remplace l'ancien decalage fixe +4 ans (remarque 1c).
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
                    df[col] = df[col] + date_shift
                    date_columns_detected.append(col)
                    print(f"📅 Colonne '{col}' décalée → max {df[col].max().date()}")

            # ----------------------------------------------------------------
            # 8. NETTOYAGE DES MONTANTS + CONVERSION DES UNITÉS FINANCIÈRES
            # ----------------------------------------------------------------
            # METHODOLOGIE DU CHOIX DU DIVISEUR -- formule dérivée d'une
            # reference externe reelle, PAS d'un total agrege qu'on attend
            # deja (circulaire) et PAS arrondie a un chiffre "propre" par
            # convention supposee (la version precedente arrondissait 88,2
            # a 100 en supposant que les systemes financiers stockent des
            # multiplicateurs ronds -- hypothese non verifiee pour CE
            # systeme MAE precisement, donc abandonnee : on garde la valeur
            # derivee des donnees, pas une approximation esthetique).
            #
            # FORMULE :
            #   k = moyenne(PRIME_NETTE brute) / moyenne(prime auto reelle en Tunisie)
            #
            # Valeurs :
            #   moyenne(PRIME_NETTE brute, 400 000 lignes)  = 67 585,40
            #   prime auto moyenne en Tunisie (reference)   =    766,50 TND/an
            #     source : Managers.tn, 2021 -- "Combien coute en moyenne
            #     une assurance auto ?" (assurance.tn cite le meme ordre de
            #     grandeur pour le marche tunisien)
            #   => k = 67 585,40 / 766,50 ≈ 88,2 -- retenu tel quel (88)
            #
            # HYPOTHESE ALTERNATIVE TESTEE ET ECARTEE : un ecart aussi grand
            # pourrait venir d'un melange contrats individuels/flotte (une
            # flotte a une prime legitimement bien plus elevee, sans aucun
            # probleme d'unite). Verifie : Mono_Flotte=10 (individuel)
            # represente 399 961 lignes sur 400 000 (99,99%), et leur
            # moyenne seule (67 575,56) est quasi identique a la moyenne
            # globale (67 585,40) -- la quasi-totalite du portefeuille est
            # individuelle et presente deja cet ecart. Le melange flotte
            # n'explique donc pas l'ecart : ca reste bien une difference
            # d'unite brute, pas un artefact de composition du portefeuille.
            #
            # k≈88 n'est pas exactement 100, ecart residuel attendu et
            # explicable (reference externe de 2021 alors que les primes
            # augmentent depuis ; portefeuille MAE = mutuelle des
            # enseignants, pas un echantillon de la population generale) --
            # mais rien ne justifie de forcer ce chiffre vers 100, donc k=88
            # est utilise directement.
            #
            # Cette meme formule ecarte aussi explicitement l'option "ne
            # rien diviser" (k=1) : une prime moyenne reelle de 766,50 TND/an
            # representee par une valeur brute de 67 585,40 serait fausse
            # d'un facteur ~88, pas juste "haute" -- et ce facteur ne
            # s'explique pas par un melange de types de contrats (voir
            # verification ci-dessus), donc "garder tel quel" n'est pas une
            # option valide non plus.
            RAW_TO_DINARS = 88.0

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
    # Decalage calcule UNE SEULE FOIS a partir de DEBUT_PERI (Production),
    # puis reutilise pour les DEUX fichiers -- garantit que Production et
    # Sinistres restent sur la meme ligne de temps apres decalage (un
    # sinistre doit rester contemporain du contrat concerne).
    _prod_path = os.path.join(BASE_DIR, "raw_data", "Base Production.csv")
    _raw_debut_peri = pd.to_datetime(
        pd.read_csv(_prod_path, sep=';', encoding='latin-1', usecols=['DEBUT_PERI'])['DEBUT_PERI'],
        dayfirst=True, errors='coerce'
    )
    _raw_max_date = _raw_debut_peri.max()
    DATE_SHIFT = compute_dynamic_date_shift(_raw_max_date)
    print(f"📅 Decalage dynamique calcule : {_raw_max_date.date()} -> "
          f"{(_raw_max_date + DATE_SHIFT).date()} (base : aujourd'hui = {datetime.now().date()})")

    clean_augment_shift(_prod_path, "Production", DATE_SHIFT)
    clean_augment_shift(os.path.join(BASE_DIR, "raw_data", "Base Sinistres.csv"), "Sinistres", DATE_SHIFT)