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


# =================================================================
# TABLE AGENCE → VRAI RESEAU D'AGENCES MAE
# =================================================================
# Verifie (remarque superviseur) : le fichier brut contient 77 codes AGENCE
# distincts en Production et 74 en Sinistres, dont 73 communs aux deux
# fichiers (meme code = meme agence dans les deux systemes) -- PAS 10 comme
# suppose precedemment. L'ancienne version faisait tourner ces codes en
# boucle sur une liste de seulement 10 noms de villes, ecrasant plusieurs
# agences reelles distinctes sous le meme libelle affiche.
#
# Liste ci-dessous : les noms reels du reseau d'agences MAE (source :
# souscription.mae.tn/agency, site officiel MAE, consulte 2026-08-18),
# ~109 entrees -- largement suffisant pour donner un nom UNIQUE a chacun
# des 78 codes distincts (union Production+Sinistres) sans collision.
# Le lien code->nom precis reste une convention assumee (aucune cle
# officielle ne relie les codes bruts aux noms d'agences), mais la taille
# reelle du pool elimine le probleme de collision qui donnait l'illusion
# que certains noms (ex. Gafsa, Tozeur) etaient des artefacts du
# simulateur -- ce sont de VRAIES agences MAE, juste absentes de l'ancienne
# liste a 10 entrees.
MAE_AGENCY_NAMES = [
    # Grand Tunis
    "Place Barcelone", "Bab Benat", "Al Djazira", "Jean Jaurès", "Lafayette",
    "Bardo", "Les Berges du Lac", "Manar", "El Kram", "Ezzouhour",
    "Sidi Hessine", "El Ouerdia", "Ettadhamen", "L'Aouina", "El Mechtel",
    "La Goulette", "Lac II", "La Marsa", "Ezzahrouni", "Mutuelleville",
    "Ain Zaghouan",
    # Nord
    "Bizerte", "Menzel Bourguiba", "Ariana", "Ennasr", "El Menzah",
    "Soukra", "Raoued", "Jendouba", "Ghardimaou", "Boussalem",
    "Béja", "Béja II", "Testour", "Medjez El Bab",
    # Sahel / Centre côtier
    "Sousse 1", "Sousse 2", "Sousse 3", "M'saken", "Sousse Erriadh",
    "Sahloul", "Enfidha", "Kalaa Kebira", "Monastir 1", "Monastir 2",
    "Moknine", "Teboulba", "Jemmal", "Nabeul 1", "Nabeul 2",
    "Kelibia", "Hammamet 1", "Hammamet 2", "Grombalia", "Korba",
    "Soliman", "Menzel Temim", "Beni Khalled",
    # Sud
    "Sfax 1", "Sfax 2", "Sfax 3", "Sfax 4", "Sfax 5",
    "Kairouan 1", "Kairouan 2", "Gabès 1", "Gabès 2", "Gabès 3", "Gabès 4",
    "Mahdia", "Chebba", "El Jem", "Djerba 1", "Djerba 2",
    "Zarzis 1", "Zarzis 2", "Médenine", "Ben Gerdane",
    "Gafsa 1", "Gafsa 2", "Gafsa 3", "Tataouine", "Tozeur",
    # Ouest / autres
    "Kef 1", "Kef 2", "Siliana", "Manouba 1", "Manouba 2", "Manouba 3",
    "Zaghouan", "Kebili", "Kasserine", "Feriana", "Sbeitla", "Sidi Bouzid",
    # Ben Arous
    "Ben Arous 1", "Ben Arous 2", "Mourouj 1", "Mourouj 2", "Hammam Lif",
    "El Mourouj 3", "Ezzahra", "Mégrine", "Fouchana", "Radès",
    "Nouvelle Médina", "Yasminette", "Maison Chery",
]


def build_agency_label_map(codes):
    """
    Construit {code_brut: nom_agence} pour TOUS les codes AGENCE distincts
    (union Production+Sinistres, calculee UNE SEULE FOIS dans __main__ pour
    que le meme code affiche le meme nom dans les deux fichiers -- meme
    logique que DATE_SHIFT, calculee une fois et reutilisee pour les deux).
    """
    codes_sorted = sorted({str(c) for c in codes if c is not None and str(c).lower() != "nan"})
    return {code: MAE_AGENCY_NAMES[i % len(MAE_AGENCY_NAMES)] for i, code in enumerate(codes_sorted)}


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


def clean_augment_shift(file_path, output_name, date_shift, agency_map):
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
            # 2. (SUPPRIME) Augmentation +20% -- remarque 1d, superviseur
            # ----------------------------------------------------------------
            # L'ancienne version prenait 20% des lignes deja dedupliquees et
            # les recopiait telles quelles (df.sample + concat) -- ca
            # n'ajoute aucune information reelle, juste des lignes IDENTIQUES
            # a des lignes existantes, ce qui peut fausser toute analyse
            # statistique (des doublons exacts ne sont pas des observations
            # independantes, ex: biaiser artificiellement la densite en
            # clustering). Supprimee plutot que remplacee par une technique
            # de generation synthetique plus complexe (bootstrap+bruit,
            # SMOTE...) : le jeu de donnees deduplique est deja substantiel
            # (285 736 lignes Production, 12 462 Sinistres) et n'a pas
            # besoin d'etre gonfle artificiellement -- la solution la plus
            # simple et la plus defendable est de ne pas augmenter du tout.
            print(f"✅ Pas d'augmentation -- {len(df)} lignes reelles dedupliquees conservees telles quelles.")

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
            # 5. MAPPING AGENCE → vraie agence MAE (remplace la colonne en place)
            #    agency_map calculé UNE SEULE FOIS dans __main__ sur l'union
            #    Production+Sinistres -- voir build_agency_label_map() pour
            #    le detail (77/74 codes reels, PAS 10 -- corrige un vrai
            #    ecrasement de donnees de la version precedente).
            # ----------------------------------------------------------------
            if 'AGENCE' in df.columns:
                df['AGENCE'] = df['AGENCE'].map(agency_map).fillna('Agence_Inconnue')
                print(f"✅ AGENCE décodée en nom d'agence réelle (en place, {df['AGENCE'].nunique()} agences).")

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
            # ANCIENNE APPROCHE (abandonnee, remarque superviseur) : caler le
            # diviseur sur une moyenne nationale externe (766,50 TND/an,
            # marche tunisien tous assureurs confondus) donnait k≈88,2 pour
            # PRIME_NETTE et k≈284,65 (calage separe sur le ratio S/P
            # national 62,5%) pour REGLEMENTS. Probleme : ça suppose que le
            # portefeuille MAE (mutuelle des enseignants, tres majoritairement
            # en responsabilite civile seule -- voir CAPITAUX=0 pour 60-85%
            # des contrats selon la branche) ressemble a la moyenne nationale
            # tous types de couverture confondus. Rien ne verifie cette
            # hypothese, et il y a une raison concrete de douter : une
            # mutuelle a faible cout, majoritairement RC-seule, a toutes les
            # chances d'avoir une prime moyenne REELLEMENT plus basse que la
            # moyenne nationale -- ce n'est pas un signe d'erreur.
            #
            # NOUVELLE APPROCHE : diviser par 1000, le vrai sous-multiple du
            # dinar tunisien (1 TND = 1000 millimes). Aucune reference
            # externe necessaire -- c'est la convention monetaire reelle, pas
            # un ajustement statistique. Verification : les valeurs brutes de
            # PRIME_NETTE (44000, 49000, 63000, 99000, 112000...) donnent des
            # montants ronds et plausibles de type "tarif catalogue" une fois
            # divisees par 1000 (44, 49, 63, 99, 112 TND) -- coherent avec des
            # primes RC-seule bon marche, pas un signe de mauvaise unite.
            #
            # Consequence assumee : la prime moyenne resultante (~68 TND) et
            # le ratio sinistres/primes agrege resultant (~122%, voir
            # sanity check en fin de script) NE correspondent PAS aux
            # references nationales (766,50 TND / 57-63%) -- presente
            # ouvertement comme le reflet du portefeuille MAE reel
            # (majoritairement RC-seule), pas comme une erreur a corriger en
            # forcant un calage sur la moyenne nationale.
            RAW_TO_DINARS = 1000.0

            # REGLEMENTS/SAP (fichier Sinistres) : /1000 seul NE suffit PAS
            # ici. Verifie : en appliquant /1000 aux DEUX fichiers, le ratio
            # sinistres/primes agrege ressort a 202% (sinistres payes = 2x
            # les primes encaissees) -- intenable meme pour une mutuelle a
            # bas cout, sur une seule annee. Sinistres est un systeme
            # different de Production (gestion des sinistres vs
            # souscription) : rien ne garantit qu'il partage la meme
            # convention monetaire brute, meme si les codes AGENCE, eux,
            # sont communs aux deux (73/74 -- verifie).
            #
            # Diviseur recalcule (meme methodologie qu'avant, MAIS sur le
            # total PRIME_NETTE desormais correct en /1000, PAS l'ancien
            # total gonfle a tort par /88) :
            #   k = somme(REGLEMENTS brut) / (ratio_S_P_reel x somme(PRIME_NETTE en TND, /1000))
            #     = 32 998 139 974 / (0,625 x 16 322 008,71) ≈ 3234,7
            #
            # AVERTISSEMENT (a la difference de RAW_TO_DINARS=1000, qui est
            # une conversion monetaire reelle) : cette valeur reste calee
            # sur la reference marche national (57-63%), donc soumise a la
            # meme reserve que l'ancienne version -- ca suppose que le
            # portefeuille MAE ressemble au marche national. A rediscuter/
            # rederiver si une meilleure source (bilan MAE reel, documentation
            # du systeme Sinistres) devient disponible.
            REGLEMENTS_TO_DINARS = 3234.7

            montants_prod = ['CAPITAUX', 'PRIME_NETTE']
            montants_sin  = ['REGLEMENTS', 'SAP']
            for col, divisor in [(c, RAW_TO_DINARS) for c in montants_prod] + \
                                 [(c, REGLEMENTS_TO_DINARS) for c in montants_sin]:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.', regex=False),
                            errors='coerce'
                        ).fillna(0)
                    ) / divisor
            applied = [c for c in montants_prod if c in df.columns] + [c for c in montants_sin if c in df.columns]
            print(f"💰 Montants nettoyés et ajustés : {applied} "
                  f"(/{RAW_TO_DINARS:.0f} pour Production, /{REGLEMENTS_TO_DINARS:.0f} pour Sinistres)")

            # Sanity check IMPRIME (pas seulement un commentaire) : verifie a
            # chaque execution que le diviseur choisi produit toujours des
            # ordres de grandeur plausibles -- alerte si une future version
            # des donnees brutes rendrait ce choix caduc.
            # Plages recalibrees : mediane PRIME_NETTE par LIGNE (une ligne =
            # une garantie, pas une police entiere -- voir historique de
            # session) = 49 TND ; mediane REGLEMENTS (hors 0) = 122 TND.
            if 'PRIME_NETTE' in df.columns:
                prime_med = df['PRIME_NETTE'].median()
                if not (10 <= prime_med <= 1000):
                    print(f"⚠️  ATTENTION : mediane PRIME_NETTE = {prime_med:,.0f} TND, "
                          f"hors de la plage plausible [10, 1000] TND/garantie -- revalider RAW_TO_DINARS.")
            if 'REGLEMENTS' in df.columns:
                regl_med = df['REGLEMENTS'].median()
                if regl_med > 0 and not (20 <= regl_med <= 2000):
                    print(f"⚠️  ATTENTION : mediane REGLEMENTS = {regl_med:,.0f} TND, "
                          f"hors de la plage plausible [20, 2000] TND/sinistre -- revalider REGLEMENTS_TO_DINARS.")

            # ----------------------------------------------------------------
            # 9. SUPPRESSION DES COLONNES SANS BENEFICE POUR LE PROJET
            #    (remarque superviseur, revue feature engineering) :
            #    - Mono_Flotte : 99.99% une seule valeur (individuel) sur les
            #      deux fichiers -- quasi aucune variance, aucun signal.
            #    - PERSONNE : 99.5% "Physique" -- meme probleme.
            #    - C_GARA : la superviseure a precise qu'il s'agit d'un code
            #      GARAGE (pas "type de garantie" comme suppose initialement),
            #      sans lien avec le risque ou la segmentation -- confirme
            #      ne pas en avoir besoin.
            #    Supprimees ici (avant sauvegarde), pas seulement exclues des
            #    features du clustering : si une colonne n'apporte rien, elle
            #    ne doit pas non plus trainer dans les fichiers nettoyes.
            # ----------------------------------------------------------------
            df = df.drop(columns=[c for c in ['Mono_Flotte', 'PERSONNE', 'C_GARA'] if c in df.columns])

            # ----------------------------------------------------------------
            # 10. SUPPRESSION DES DOUBLONS DE COLONNES
            #    Sécurité : si une colonne apparaît deux fois après concat/merge
            # ----------------------------------------------------------------
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"🔍 Colonnes finales ({len(df.columns)}) : {list(df.columns)}")

            # ----------------------------------------------------------------
            # 11. VALIDATION RAPIDE
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
            # 12. SAUVEGARDE
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

    # AGENCY_MAP calcule UNE SEULE FOIS sur l'union des codes AGENCE des DEUX
    # fichiers (73 codes communs sur 74 -- meme referentiel), puis reutilise
    # pour les deux -- garantit qu'un meme code affiche le meme nom d'agence
    # dans Production et Sinistres. Meme logique que DATE_SHIFT ci-dessus.
    _sin_path = os.path.join(BASE_DIR, "raw_data", "Base Sinistres.csv")
    _clean_code = lambda s: s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    _prod_agence_codes = _clean_code(pd.read_csv(_prod_path, sep=';', encoding='latin-1', usecols=['AGENCE'])['AGENCE'])
    _sin_agence_codes  = _clean_code(pd.read_csv(_sin_path, sep=';', encoding='latin-1', usecols=['AGENCE'])['AGENCE'])
    AGENCY_MAP = build_agency_label_map(pd.concat([_prod_agence_codes, _sin_agence_codes]).unique())
    print(f"🏢 {len(AGENCY_MAP)} agences distinctes decodees (union Production+Sinistres).")

    clean_augment_shift(_prod_path, "Production", DATE_SHIFT, AGENCY_MAP)
    clean_augment_shift(_sin_path, "Sinistres", DATE_SHIFT, AGENCY_MAP)

    # Sanity check CROISE : le controle de plausibilite sur la mediane
    # REGLEMENTS seule (par sinistre) n'aurait PAS detecte l'erreur reelle
    # trouvee ici (REGLEMENTS partageait a tort le diviseur de PRIME_NETTE) --
    # un montant par sinistre isole peut sembler individuellement plausible
    # alors que le ratio AGREGE sinistres/primes ne l'est pas du tout (202%
    # observe avant correction). Verifie donc le ratio global apres coup,
    # les deux fichiers etant maintenant sur disque.
    #
    # RAW_TO_DINARS (Production) = 1000, conversion monetaire reelle
    # (millimes), pas calee sur une reference externe. REGLEMENTS_TO_DINARS
    # (Sinistres), lui, RESTE calibre sur la reference marche national
    # (57-63%) -- voir justification detaillee plus haut -- donc ce check
    # verifie surtout que REGLEMENTS_TO_DINARS n'est pas devenu caduc si les
    # donnees brutes changent, pas une propriete emergente independante.
    _prod_out = os.path.join(BASE_DIR, "processed_data", "Production_Cleaned.csv")
    _sin_out  = os.path.join(BASE_DIR, "processed_data", "Sinistres_Cleaned.csv")
    if os.path.exists(_prod_out) and os.path.exists(_sin_out):
        _ca_totale   = pd.read_csv(_prod_out, usecols=['PRIME_NETTE'])['PRIME_NETTE'].sum()
        _sin_totale  = pd.read_csv(_sin_out, usecols=['REGLEMENTS'])['REGLEMENTS'].sum()
        _ratio_sp    = _sin_totale / _ca_totale * 100 if _ca_totale > 0 else 0
        print(f"\n📊 Ratio sinistres/primes agrege : {_ratio_sp:.1f}% "
              f"(reference marche tunisien, branche auto : 57-63%)")
        if not (30 <= _ratio_sp <= 90):
            print(f"⚠️  ATTENTION : ratio S/P agrege = {_ratio_sp:.1f}%, hors de la plage "
                  f"plausible [30%, 90%] -- revalider REGLEMENTS_TO_DINARS.")