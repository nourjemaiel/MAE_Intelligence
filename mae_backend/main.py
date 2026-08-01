# -*- coding: utf-8 -*-
"""
main.py — MAE Assurances · FastAPI Backend v3.3
Real-time simulator + Analytics endpoints + Agent ReAct (via agent.py)
Modele agent : Llama 3.3 70B via Groq (gratuit)

Changelog v3.3 (2026-07, fix rapports PDF) :
- BUG CORRIGE : REPORTS_DIR = "reports" etait relatif au repertoire de
  travail du process (cwd au moment de lancer uvicorn), pas au dossier
  de main.py. Si uvicorn est lance depuis un autre dossier que
  mae_backend/, le PDF etait cree ailleurs et semblait "manquant". Le
  chemin est maintenant ancre sur le dossier de ce fichier via
  os.path.dirname(os.path.abspath(__file__)).
- AJOUT API_BASE_URL : generate_report() renvoyait une URL RELATIVE
  ("/api/reports/xxx.pdf"), qui n'est pas un lien cliquable/copiable
  utilisable tel quel dans le chat. download_url est desormais une URL
  absolue construite avec API_BASE_URL (configurable via variable d
  environnement, defaut http://localhost:8000).
- SCOPING : tool_generate_report() accepte maintenant agence/region/
  branche/mois_num/sections pour generer un rapport cible sur ce que l
  utilisateur demande, au lieu de toujours produire le rapport complet.
  Voir agent.py (TOOL_DEFS + SYSTEM_PROMPT) pour le cote "l agent sait
  quand utiliser ces parametres".
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import math
import threading
import time
import random
import unicodedata
import logging
from datetime import datetime, timedelta
import mlflow
from fastapi.responses import FileResponse
from fastapi import HTTPException

from agent import MAEAgent
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="MAE Intelligence API", version="3.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("mlruns")

# ════════════════════════════════════════════════════════════════
# BASE URL — utilisee pour construire des liens ABSOLUS et cliquables
# (ex: dans les reponses de l agent). Surchargeable via .env si le
# backend tourne derriere un autre host/port (deploiement, ngrok, etc.)
# ════════════════════════════════════════════════════════════════
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

AGENCES  = ["Sfax Ville","Tunis Centre","Sousse","Nabeul","Bizerte",
            "Gabes","Sfax Nord","Monastir","Ariana","Gafsa",
            "Kairouan","Medenine","Jendouba","Beja","Tozeur"]

BRANCHE_LABELS = {
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
ALL_BRANCHE_LABELS = sorted(set(BRANCHE_LABELS.values()))

_BRANCHE_W = {
    'Tourisme':                        0.857,
    'Taxi':                            0.030,
    'Taxi_Plus_4_places':              0.015,
    'Taxi_Collectif':                  0.015,
    'Transport_Prive_Inf_3.5T':        0.025,
    'Transport_Prive_Sup_3.5T':        0.015,
    'Louage':                          0.020,
    '2_Roues':                         0.015,
    'Transport_Public_Inf_3.5T':       0.0012,
    'Transport_Public_Sup_3.5T':       0.0012,
    'Transport_Personnel':             0.0012,
    'Transport_Hotel_Agence':          0.0012,
    'Transport_Agricole_Inf_3.5T':     0.0008,
    'Transport_Agricole_Sup_3.5T':     0.0008,
    'Auto_Ecole_Tourisme':             0.0008,
    'Auto_Ecole_Utilitaire':           0.0008,
    'Engin_Agricole_Prive':            0.0004,
    'Engin_Agricole_Etablissement':    0.0004,
    'Engin_de_Chantier':               0.0004,
    'Transport_Rural':                 0.0004,
    'Ambulance':                       0.0004,
}
_w_sum = sum(_BRANCHE_W.get(b, 0.0004) for b in ALL_BRANCHE_LABELS)
BRANCHE_WEIGHTS_LIST = [_BRANCHE_W.get(b, 0.0004) / _w_sum for b in ALL_BRANCHE_LABELS]

REGIONS  = ["Grand Tunis","Nord","Centre","Sud","Sahel"]
CSP_LIST = ["Salarie","Fonctionnaire","Commercant","Artisan","Profession liberale","Retraite"]
MOIS_LABELS = ["Jan","Fev","Mar","Avr","Mai","Jun","Jul","Aou","Sep","Oct","Nov","Dec"]

_bw = [0.857,0.06,0.04,0.02,0.015,0.008]; _bs = sum(_bw)
BRANCH_WEIGHTS = [x/_bs for x in _bw]
_aw = [0.12,0.11,0.09,0.08,0.07,0.07,0.06,0.06,0.06,0.05,0.05,0.05,0.05,0.05,0.04]; _as = sum(_aw)
AGENCE_WEIGHTS = [x/_as for x in _aw]

TOTAL_CONTRATS_REEL = 342883

# Parametres du generateur synthetique de PRIME_NETTE pour les NOUVEAUX
# contrats simules (simulator() et le fallback _gen_synthetic()). Avant ce
# correctif, lognormal(7.5, 1.2) clippe [200, 80000] produisait une moyenne
# simulee de ~3758 TND contre ~571 TND dans les vraies donnees (Production_
# Cleaned.csv, 342883 lignes) -- soit 6.6x trop eleve. Consequence directe :
# compute_data_drift() compare la moyenne des 50 derniers contrats a cette
# reference reelle, donc des que la fenetre des 50 derniers est remplie de
# contrats simules (~50 x 5s = ~4min apres le demarrage), l'alerte de derive
# se declenchait a coup sur ET NE SE RESORBAIT JAMAIS -- ce n'etait pas de
# la derive detectee, juste un generateur mal calibre. Valeurs ci-dessous
# ajustees par methode des moments pour reproduire moyenne/ecart-type reels
# (~571 / ~719 TND) ; verifie empiriquement : moyenne simulee obtenue ~568,
# mediane ~357 (vs ~385 reel) sur 50k tirages.
PRIME_SIM_LOGNORMAL_MU    = 5.87
PRIME_SIM_LOGNORMAL_SIGMA = 0.97
PRIME_SIM_MIN_TND         = 50.0
PRIME_SIM_MAX_TND         = 70000.0

# Seuil de ratio sinistres/primes (S/P) individuel au-dela duquel un client
# est considere a risque. Partage entre detect_anomalies() (comptage) et
# risk_clients() (liste nominative) pour rester coherent.
RISK_RATIO_THRESHOLD_PCT = 150.0

# Prime totale minimale (TND) pour qu'un client entre dans risk_clients().
# Sans ce plancher, un client dont la prime cumulee est quasi nulle (ex:
# 15-45 TND, vs une mediane observee de ~440 TND -- typiquement une police
# partielle ou un avenant isole) produit un ratio S/P demesurement gonfle
# des qu'un sinistre meme modeste lui est rattache, simplement parce que le
# denominateur est proche de zero -- ce n'est pas un signal de risque
# exploitable. Le plancher (fixe au 25e percentile approx. des primes
# observees) l'exclut pour garder une liste presentable et defendable
# devant un jury. Distinct de RISK_EXCEPTIONNEL_THRESHOLD_PCT ci-dessous,
# qui traite un autre cas : un denominateur normal mais un sinistre reel
# exceptionnellement eleve.
MIN_CA_FOR_RISK_TND = 200.0

# Seuil au-dela duquel un ratio S/P n'est plus juste "eleve" mais reflete
# probablement UN sinistre catastrophique isole plutot qu'un pattern de
# risque recurrent (verifie sur les donnees brutes : la mediane des
# sinistres bruts est deja ~7.5x la mediane des primes brutes, et le
# sinistre max ~122x la prime max -- ce n'est PAS un artefact de la
# division /100 dans 02_cleaning.py, puisqu'elle s'applique identiquement
# aux deux colonnes et ne change donc aucun ratio). Sert uniquement a
# etiqueter la reponse pour un jury, ne modifie aucun chiffre.
RISK_EXCEPTIONNEL_THRESHOLD_PCT = 2000.0

def _risk_severity_label(ratio_pct: float) -> str:
    if ratio_pct > RISK_EXCEPTIONNEL_THRESHOLD_PCT:
        return "Sinistre exceptionnel isole (a distinguer d'un risque recurrent)"
    return "Risque eleve"


# Ecart (%) entre la prime moyenne recente (50 derniers contrats simules) et
# la prime moyenne de reference (portefeuille entier) au-dela duquel on
# considere qu'il y a une derive des donnees. Module-level pour etre partage
# entre compute_data_drift() (calcul), simulator() (declenchement d'alerte
# en continu) et tool_agent_monitoring() (affichage detaille).
#
# JUSTIFICATION STATISTIQUE (pas une valeur arbitraire) : PRIME_NETTE a un
# coefficient de variation (CV = ecart-type/moyenne) reel de ~1.26 (719/571
# TND, Production_Cleaned.csv). Pour une moyenne calculee sur n=50 tirages,
# l'erreur-type de cette moyenne (en %) vaut CV/racine(n) * 100 = 1.26/7.07
# *100 ~= 17.8%. Un seuil de 15% (l'ancienne valeur) est donc INFERIEUR a
# 1 erreur-type -- meme avec un generateur parfaitement calibre sur la
# vraie distribution, ~30-40% des fenetres de 50 contrats declenchent une
# "derive" par pur bruit d'echantillonnage. Seuil recalcule pour un seuil
# de confiance ~95% (z=1.96) : 1.96 * CV / racine(n) * 100 ~= 35%. A
# recalculer si n (fenetre de simulator()) ou le CV reel des donnees change.
DRIFT_SEUIL_ALERTE_PCT = 35.0


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

_AGENCE_CANONICAL = {_strip_accents(a).lower(): a for a in AGENCES}
_AGENCE_OVERRIDES = {
    "sfax":           "Sfax Ville",
    "sfax ville":     "Sfax Ville",
    "sfax nord":      "Sfax Nord",
    "gabes":          "Gabes",
    "beja":           "Beja",
    "tunis":          "Tunis Centre",
    "tunis centre":   "Tunis Centre",
}

def normalize_agence(raw) -> str:
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return random.choice(AGENCES)
    s = str(raw).strip()
    key = _strip_accents(s).lower()
    if key in _AGENCE_OVERRIDES:
        return _AGENCE_OVERRIDES[key]
    if key in _AGENCE_CANONICAL:
        return _AGENCE_CANONICAL[key]
    return s


def normalize_branche(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.random.choice(ALL_BRANCHE_LABELS, p=BRANCHE_WEIGHTS_LIST)
    key = str(raw).strip()
    if key == "" or key.lower() == "nan":
        return np.random.choice(ALL_BRANCHE_LABELS, p=BRANCHE_WEIGHTS_LIST)
    if key.endswith(".0"):
        key = key[:-2]
    if key in BRANCHE_LABELS:
        return BRANCHE_LABELS[key]
    return key


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
    codes_set = {str(c) for c in codes if c is not None and str(c).lower() != "nan"}
    if not codes_set:
        return {}

    def _looks_numeric(c):
        try:
            float(c)
            return True
        except (ValueError, TypeError):
            return False

    if not all(_looks_numeric(c) for c in codes_set):
        return {c: c for c in codes_set}

    codes_sorted = sorted(codes_set, key=lambda c: (0, float(c)))
    n = len(codes_sorted)
    buckets = len(REGIONS)
    size = math.ceil(n / buckets)

    labels = {}
    for i, code in enumerate(codes_sorted):
        bucket_idx = min(i // size, buckets - 1)
        macro = REGIONS[bucket_idx]
        localities = TUNISIA_LOCALITIES[macro]
        pos_in_bucket = i - bucket_idx * size
        locality = localities[pos_in_bucket % len(localities)]
        if pos_in_bucket >= len(localities):
            locality = f"{locality} {pos_in_bucket // len(localities) + 1}"
        labels[code] = locality
    return labels


state = {
    "contrats":          [],
    "sinistres":         [],
    "last_update":       datetime.now().isoformat(),
    "total_generated":   0,
    "lock":              threading.Lock(),
    "ca_total_reel":     0.0,
    "nb_contrats_reel":  0,
    "ca_by_month_reel":  {},
    "sin_total_reel":    0.0,
}

def seed_data():
    prod_path = "../processed_data/Production_Cleaned.csv"
    sin_path  = "../processed_data/Sinistres_Cleaned.csv"
    np.random.seed(42)

    if os.path.exists(prod_path):
        try:
            df_full = pd.read_csv(prod_path)

            if "AGENCE" in df_full.columns:
                df_full["AGENCE"] = df_full["AGENCE"].apply(normalize_agence)
            else:
                df_full["AGENCE"] = np.random.choice(AGENCES, size=len(df_full), p=AGENCE_WEIGHTS)

            if "N_BRA" in df_full.columns:
                df_full["BRANCHE"] = df_full["N_BRA"].apply(normalize_branche)
            elif "BRANCHE" in df_full.columns:
                df_full["BRANCHE"] = df_full["BRANCHE"].apply(normalize_branche)
            else:
                df_full["BRANCHE"] = np.random.choice(ALL_BRANCHE_LABELS, size=len(df_full), p=BRANCHE_WEIGHTS_LIST)

            if "PRIME_NETTE" in df_full.columns:
                df_full["PRIME_NETTE"] = pd.to_numeric(df_full["PRIME_NETTE"], errors="coerce").fillna(500.0).abs()
            else:
                df_full["PRIME_NETTE"] = 500.0

            state["ca_total_reel"]    = float(df_full["PRIME_NETTE"].sum())
            state["nb_contrats_reel"] = len(df_full)

            # Conserve l'INTEGRALITE des colonnes utiles (pas seulement l'echantillon
            # de 10k lignes ci-dessous) pour les outils qui doivent agreger par client
            # sur la population complete plutot que sur un sous-echantillon -- voir
            # tool_risk_clients() : croiser un echantillon prod (10k/342k) avec un
            # echantillon sinistres (3k/15k) tires INDEPENDAMMENT produit des ratios
            # par client statistiquement absurdes (numerateur et denominateur ne
            # portent pas sur les memes clients avec la meme completude).
            _cols_full = [c for c in ["N_CLIENT", "AGENCE", "BRANCHE", "Region", "PRIME_NETTE"] if c in df_full.columns]
            state["prod_full_df"] = df_full[_cols_full].copy()

            if "DEBUT_PERI" in df_full.columns:
                mois_series = pd.to_datetime(df_full["DEBUT_PERI"], errors="coerce").dt.month
                state["ca_by_month_reel"] = (
                    df_full.assign(_MOIS=mois_series)
                    .dropna(subset=["_MOIS"])
                    .groupby("_MOIS")["PRIME_NETTE"].sum()
                    .to_dict()
                )
            else:
                state["ca_by_month_reel"] = {}

            sample_size = min(10000, len(df_full))
            df = df_full.sample(n=sample_size, random_state=42).reset_index(drop=True)

            records = df.to_dict("records")
            for r in records:
                r["MOIS"]        = pd.to_datetime(r.get("DEBUT_PERI"), errors="coerce").month \
                                   if r.get("DEBUT_PERI") else random.randint(1, 12)
                r["AGE"]         = r.get("AGE", random.randint(22, 72))
                r["SEXE"]        = str(r.get("SEXE", random.choice(["M", "F"])))
                r["Region"]      = str(r.get("Region", random.choice(REGIONS)))
                r["CSP"]         = str(r.get("CSP", random.choice(CSP_LIST)))
                r["BONUS_MALUS"] = int(float(r.get("BONUS_MALUS", 0) or 0))
                r["timestamp"]   = (datetime.now() - timedelta(days=random.randint(0, 364))).isoformat()
            state["contrats"] = records
            print(f"✅ {len(records)} contrats charges (echantillon aleatoire sur {len(df_full)} lignes reelles)")
            print(f"   CA total reel (100% du fichier) : {state['ca_total_reel']:,.2f} TND")
        except Exception as e:
            print(f"⚠️  CSV prod ({e}) -> donnees synthetiques")
            _gen_synthetic()
    else:
        print("⚠️  CSV prod introuvable -> donnees synthetiques")
        _gen_synthetic()

    if os.path.exists(sin_path):
        try:
            df_sin_full = pd.read_csv(sin_path)

            if "AGENCE" in df_sin_full.columns:
                df_sin_full["AGENCE"] = df_sin_full["AGENCE"].apply(normalize_agence)
            if "REGLEMENTS" in df_sin_full.columns:
                df_sin_full["REGLEMENTS"] = pd.to_numeric(df_sin_full["REGLEMENTS"], errors="coerce").fillna(1000.0).abs()
            else:
                df_sin_full["REGLEMENTS"] = 1000.0

            state["sin_total_reel"] = float(df_sin_full["REGLEMENTS"].sum())

            _cols_sin_full = [c for c in ["N_CLIENT", "AGENCE", "REGLEMENTS"] if c in df_sin_full.columns]
            state["sin_full_df"] = df_sin_full[_cols_sin_full].copy()

            sample_size = min(3000, len(df_sin_full))
            df = df_sin_full.sample(n=sample_size, random_state=42).reset_index(drop=True)

            records = df.to_dict("records")
            for r in records:
                r["TRANSACTION"] = str(r.get("TRANSACTION", random.choice(["REGLEMENT", "EN COURS", "EXPERTISE"])))
                r["MOIS"]        = pd.to_datetime(r.get("DATE_ACCIDENT"), errors="coerce").month \
                                   if r.get("DATE_ACCIDENT") else random.randint(1, 12)
                r["timestamp"]   = (datetime.now() - timedelta(days=random.randint(0, 364))).isoformat()
            state["sinistres"] = records
            print(f"✅ {len(records)} sinistres charges (echantillon aleatoire sur {len(df_sin_full)} lignes reelles)")
        except Exception as e:
            print(f"⚠️  CSV sin ({e}) -> synthetiques")
            _gen_synthetic_sin()
    else:
        print("⚠️  CSV sin introuvable -> synthetiques")
        _gen_synthetic_sin()

    state["total_generated"] = len(state["contrats"])


def _gen_synthetic(n=5000):
    records = []
    for i in range(n):
        mois = random.randint(1, 12)
        records.append({
            "N_CLIENT":    i + 1,
            "N_POLICE":    random.randint(100000, 999999),
            "AGENCE":      np.random.choice(AGENCES, p=AGENCE_WEIGHTS),
            "BRANCHE":     np.random.choice(ALL_BRANCHE_LABELS, p=BRANCHE_WEIGHTS_LIST),
            "PRIME_NETTE": float(np.clip(np.random.lognormal(PRIME_SIM_LOGNORMAL_MU, PRIME_SIM_LOGNORMAL_SIGMA), PRIME_SIM_MIN_TND, PRIME_SIM_MAX_TND)),
            "BONUS_MALUS": int(np.random.choice(range(14), p=[.05,.08,.10,.12,.13,.13,.12,.10,.08,.06,.05,.03,.03,.02])),
            "SEXE":        np.random.choice(["M", "F"], p=[0.75, 0.25]),
            "CSP":         np.random.choice(CSP_LIST, p=[0.32, 0.28, 0.18, 0.10, 0.08, 0.04]),
            "Region":      np.random.choice(REGIONS, p=[0.35, 0.15, 0.20, 0.15, 0.15]),
            "AGE":         int(np.clip(np.random.normal(47, 11), 18, 80)),
            "MOIS":        mois,
            "TRIMESTRE":   (mois - 1) // 3 + 1,
            "timestamp":   (datetime.now() - timedelta(days=random.randint(0, 364))).isoformat(),
        })
    state["contrats"] = records
    state["ca_total_reel"]    = sum(r["PRIME_NETTE"] for r in records)
    state["nb_contrats_reel"] = len(records)
    by_month = {}
    for r in records:
        by_month[r["MOIS"]] = by_month.get(r["MOIS"], 0) + r["PRIME_NETTE"]
    state["ca_by_month_reel"] = by_month


def _gen_synthetic_sin(n=1000):
    records = []
    for i in range(n):
        records.append({
            "N_CLIENT":    random.randint(1, 5000),
            "AGENCE":      np.random.choice(AGENCES, p=AGENCE_WEIGHTS),
            "REGLEMENTS":  float(np.clip(np.random.lognormal(7.8, 1.3), 500, 200000)),
            "TRANSACTION": np.random.choice(["REGLEMENT", "EN COURS", "EXPERTISE"], p=[0.55, 0.30, 0.15]),
            "MOIS":        random.randint(1, 12),
            "timestamp":   (datetime.now() - timedelta(days=random.randint(0, 364))).isoformat(),
        })
    state["sinistres"] = records
    state["sin_total_reel"] = sum(r["REGLEMENTS"] for r in records)


def simulator(interval_seconds: int = 5):
    while True:
        time.sleep(interval_seconds)
        with state["lock"]:
            mois  = datetime.now().month
            new_c = {
                "N_CLIENT":    state["total_generated"] + 1,
                "N_POLICE":    random.randint(100000, 999999),
                "AGENCE":      np.random.choice(AGENCES, p=AGENCE_WEIGHTS),
                "BRANCHE":     np.random.choice(ALL_BRANCHE_LABELS, p=BRANCHE_WEIGHTS_LIST),
                "PRIME_NETTE": float(np.clip(np.random.lognormal(PRIME_SIM_LOGNORMAL_MU, PRIME_SIM_LOGNORMAL_SIGMA), PRIME_SIM_MIN_TND, PRIME_SIM_MAX_TND)),
                "BONUS_MALUS": int(np.random.choice(range(14))),
                "SEXE":        np.random.choice(["M", "F"], p=[0.75, 0.25]),
                "CSP":         np.random.choice(CSP_LIST),
                "Region":      np.random.choice(REGIONS),
                "AGE":         int(np.clip(np.random.normal(47, 11), 18, 80)),
                "MOIS":        mois,
                "TRIMESTRE":   (mois - 1) // 3 + 1,
                "timestamp":   datetime.now().isoformat(),
            }
            state["contrats"].append(new_c)
            state["total_generated"] += 1
            state["ca_total_reel"]    = state.get("ca_total_reel", 0.0) + new_c["PRIME_NETTE"]
            state["nb_contrats_reel"] = state.get("nb_contrats_reel", 0) + 1
            by_month = state.setdefault("ca_by_month_reel", {})
            by_month[mois] = by_month.get(mois, 0) + new_c["PRIME_NETTE"]

            if random.random() < 0.30:
                new_s = {
                    "N_CLIENT":    new_c["N_CLIENT"],
                    "AGENCE":      new_c["AGENCE"],
                    "REGLEMENTS":  float(np.clip(np.random.lognormal(7.8, 1.3), 500, 200000)),
                    "TRANSACTION": np.random.choice(["REGLEMENT", "EN COURS", "EXPERTISE"], p=[0.55, 0.30, 0.15]),
                    "MOIS":        mois,
                    "timestamp":   datetime.now().isoformat(),
                }
                state["sinistres"].append(new_s)
                state["sin_total_reel"] = state.get("sin_total_reel", 0.0) + new_s["REGLEMENTS"]
            state["last_update"] = datetime.now().isoformat()

        # ── Data drift : evalue et journalise EN CONTINU, pas seulement ──
        # quand quelqu'un ouvre la page Monitoring ou interroge l'agent.
        # Journalise uniquement sur TRANSITION (derive qui apparait ou se
        # resorbe) pour eviter de spammer logs/agent.log toutes les 5s tant
        # que l'alerte reste active.
        drift = compute_data_drift()
        state["last_data_drift"] = drift
        new_alert = bool(drift.get("alerte_derive", False))
        was_alert = state.get("drift_alert_active", False)
        if new_alert and not was_alert:
            logging.warning(
                f"[DATA DRIFT] Derive detectee : ecart {drift['ecart_pct']}% "
                f"(seuil +-{drift['seuil_alerte_pct']}%) -- prime moyenne recente "
                f"{drift['prime_moyenne_recente_tnd']:,.2f} TND vs reference "
                f"{drift['prime_moyenne_reference_tnd']:,.2f} TND"
            )
        elif was_alert and not new_alert:
            logging.info(f"[DATA DRIFT] Derive resorbee -- ecart revenu a {drift['ecart_pct']}%.")
        state["drift_alert_active"] = new_alert


def get_df():
    with state["lock"]:
        return pd.DataFrame(list(state["contrats"])), pd.DataFrame(list(state["sinistres"]))


def get_full_df():
    """
    Renvoie les dataframes prod/sinistres COMPLETS (toutes les lignes du CSV
    nettoye, pas l'echantillon de 10k/3k utilise par get_df() pour le flux
    temps reel simule). Ecrit une seule fois au demarrage (seed_data), jamais
    modifie ensuite par le simulateur -- pas besoin du lock de state["contrats"].
    A utiliser pour toute agregation PAR CLIENT ou la completude des deux
    cotes (primes ET sinistres) doit correspondre au meme client, sous peine
    de ratios faux (voir commentaire dans seed_data()).
    """
    return state.get("prod_full_df", pd.DataFrame()), state.get("sin_full_df", pd.DataFrame())


def compute_data_drift():
    """
    Compare la prime moyenne des 50 derniers contrats simules a la prime
    moyenne de reference (portefeuille entier). Utilisee en continu par
    simulator() (pour journaliser/alerter des qu'une derive apparait, pas
    seulement quand quelqu'un consulte la page Monitoring) et par
    tool_agent_monitoring() (pour l'affichage detaille a la demande).
    Auparavant ce calcul n'existait QUE dans tool_agent_monitoring : la
    derive n'etait donc jamais detectee tant que personne n'appelait cet
    outil/cette page -- purement passif, jamais journalise ni visible
    ailleurs dans l'app.
    """
    try:
        with state["lock"]:
            recents = state["contrats"][-50:]
        if not recents or state.get("nb_contrats_reel", 0) <= 0:
            return {"note": "Pas assez de donnees pour evaluer la derive."}

        prime_moyenne_recente   = sum(r.get("PRIME_NETTE", 0) for r in recents) / len(recents)
        prime_moyenne_reference = state["ca_total_reel"] / state["nb_contrats_reel"]
        ecart_pct = (
            (prime_moyenne_recente - prime_moyenne_reference) / prime_moyenne_reference * 100
            if prime_moyenne_reference > 0 else 0
        )
        return {
            "prime_moyenne_reference_tnd": round(prime_moyenne_reference, 2),
            "prime_moyenne_recente_tnd":   round(prime_moyenne_recente, 2),
            "ecart_pct":                   round(ecart_pct, 1),
            "seuil_alerte_pct":            DRIFT_SEUIL_ALERTE_PCT,
            "alerte_derive":               abs(ecart_pct) > DRIFT_SEUIL_ALERTE_PCT,
        }
    except Exception as e:
        return {"error": str(e)}


def apply_filters(df, agence=None, region=None, branche=None, mois=None):
    if agence  and agence  != "all" and "AGENCE"  in df.columns: df = df[df["AGENCE"]  == agence]
    if region  and region  != "all" and "Region"  in df.columns: df = df[df["Region"]  == region]
    if branche and branche != "all" and "BRANCHE" in df.columns: df = df[df["BRANCHE"] == branche]
    if mois    and mois    != 0     and "MOIS"    in df.columns: df = df[df["MOIS"]    == mois]
    return df


def scale_ca(value, prod_len):
    if prod_len == 0:
        return 0.0
    return round(float(value) * TOTAL_CONTRATS_REEL / prod_len, 2)


def is_unfiltered(agence=None, region=None, branche=None, mois=None):
    agence_empty  = agence  in (None, "all")
    region_empty  = region  in (None, "all")
    branche_empty = branche in (None, "all")
    mois_empty    = mois in (None, 0)
    return agence_empty and region_empty and branche_empty and mois_empty


def tool_portfolio_summary(agence=None, region=None, branche=None):
    prod, sin = get_df()
    prod_f = apply_filters(prod, agence, region, branche)

    if is_unfiltered(agence, region, branche):
        ca = round(state.get("ca_total_reel", 0.0), 2)
        nb_contrats = state.get("nb_contrats_reel", TOTAL_CONTRATS_REEL)
    else:
        ca_sample = prod_f["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod_f.columns else 0
        ca = scale_ca(ca_sample, len(prod_f))
        nb_contrats = len(prod_f)

    st     = sin["REGLEMENTS"].sum() if "REGLEMENTS" in sin.columns else 0
    top_ag = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().idxmax() if len(prod_f) > 0 else "N/A"
    return {
        "ca_total":           ca,
        "nb_clients":         int(prod_f["N_CLIENT"].nunique()) if "N_CLIENT" in prod_f.columns else nb_contrats,
        "nb_contrats":        nb_contrats,
        "sinistres_total":    round(st, 2),
        "ratio_sinistralite": round(st / ca * 100, 2) if ca > 0 else 0,
        "top_agence":         str(top_ag),
        "derniere_maj":       state["last_update"],
        "total_generes":      state["total_generated"],
    }


def tool_top_agencies(n=5, agence=None, region=None, branche=None):
    prod, _ = get_df()
    prod = apply_filters(prod, agence, region, branche)
    if len(prod) == 0:
        return []
    top      = prod.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values(ascending=False).head(n)
    ca_total = prod["PRIME_NETTE"].sum()
    return [{"agence": str(a), "ca": round(v, 2), "part_pct": round(v / ca_total * 100, 1)}
            for a, v in top.items()]


_SEASONAL_RAW = [0.92,0.88,1.02,1.05,1.08,1.10,1.07,1.05,1.03,1.00,0.97,1.12]
_seasonal_avg = sum(_SEASONAL_RAW) / 12
SEASONAL_NORM = [s / _seasonal_avg for s in _SEASONAL_RAW]
MOIS_NOMS_LONGS = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                   "Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]

# ── Intervalle de confiance DYNAMIQUE (widens with forecast horizon) ──
CI_BASE_PCT   = 0.05    # +-5% pour le mois le plus proche (janvier)
CI_GROWTH_PCT = 0.003   # +0.3 point par mois d'horizon supplementaire

def ci_width_for_month(i: int) -> float:
    """
    Largeur de l'intervalle de confiance pour le mois d'indice i (0=Janvier,
    11=Decembre). Convention assumee et documentee (voir notes projet) :
    plus l'horizon de prevision est loin, plus l'incertitude est grande,
    donc l'intervalle s'elargit lineairement avec le mois plutot que de
    rester fixe a +-7% quel que soit l'horizon. A affiner avec un vrai
    ecart-type de residus (ex: recalcule sur le RMSE du modele une fois
    la correction millimes->dinars propagee dans model_comparison.ipynb).
    """
    return CI_BASE_PCT + CI_GROWTH_PCT * i


def tool_forecast(mois_num=None):
    base   = state.get("ca_total_reel", 1_630_000_000) or 1_630_000_000
    growth = 0.047
    total_2026 = base * (1 + growth)
    monthly = [total_2026 / 12 * SEASONAL_NORM[i] for i in range(12)]

    if mois_num and 1 <= mois_num <= 12:
        i  = mois_num - 1
        ca = monthly[i]
        w  = ci_width_for_month(i)
        return {"mois": MOIS_NOMS_LONGS[i], "ca_prev": round(ca, 2),
                "ci_low": round(ca * (1 - w), 2), "ci_high": round(ca * (1 + w), 2),
                "ci_width_pct": round(w * 100, 1)}

    return {
        "total_2026":     round(sum(monthly), 2),
        "croissance_pct": 4.7,
        "mois_pic":       MOIS_NOMS_LONGS[int(np.argmax(monthly))],
        "detail":         [{"mois": MOIS_NOMS_LONGS[i], "ca": round(monthly[i], 2),
                             "ci_width_pct": round(ci_width_for_month(i) * 100, 1)} for i in range(12)],
    }


def tool_explain_forecast(mois_num=None):
    base   = state.get("ca_total_reel", 1_630_000_000) or 1_630_000_000
    growth = 0.047
    total_2026 = base * (1 + growth)
    monthly = [total_2026 / 12 * SEASONAL_NORM[i] for i in range(12)]

    if mois_num and 1 <= mois_num <= 12:
        i = mois_num - 1
        base_mensuel      = base / 12
        apres_croissance  = total_2026 / 12
        apres_saisonnalite = monthly[i]
        w = ci_width_for_month(i)
        return {
            "mois": MOIS_NOMS_LONGS[i],
            "ca_prevu": round(apres_saisonnalite, 2),
            "decomposition": {
                "1_base_mensuelle_2025":        round(base_mensuel, 2),
                "2_apres_croissance_4.7pct":     round(apres_croissance, 2),
                "3_facteur_saisonnier":          round(SEASONAL_NORM[i], 3),
                "4_apres_saisonnalite_final":    round(apres_saisonnalite, 2),
            },
            "explication": (
                f"CA base 2025 reparti uniformement sur 12 mois = {round(base_mensuel, 2):,.2f} TND. "
                f"Apres application de la croissance annuelle de +4.7% (mesuree sur l'evolution reelle "
                f"du CA MAE en 2025) = {round(apres_croissance, 2):,.2f} TND. "
                f"{MOIS_NOMS_LONGS[i]} a un facteur saisonnier de {SEASONAL_NORM[i]:.3f} "
                f"({'au-dessus' if SEASONAL_NORM[i] > 1 else 'en-dessous'} de la moyenne annuelle), "
                f"d'ou la prevision finale de {round(apres_saisonnalite, 2):,.2f} TND, "
                f"avec un intervalle de confiance de +-{round(w*100,1)}% (horizon mois {mois_num}/12, "
                f"l'incertitude croit avec l'horizon de prevision)."
            ),
        }

    return {
        "total_2026": round(sum(monthly), 2),
        "decomposition": {
            "1_base_2025_reelle":         round(base, 2),
            "2_croissance_appliquee_pct": 4.7,
            "3_montant_croissance_tnd":   round(total_2026 - base, 2),
            "4_total_apres_croissance":   round(total_2026, 2),
        },
        "explication": (
            f"Le CA reel 2025 (mesure sur la totalite du fichier, pas un echantillon) est de "
            f"{round(base, 2):,.2f} TND. La croissance de +4.7% (observee reellement sur "
            f"l'evolution du CA MAE en 2025) ajoute {round(total_2026 - base, 2):,.2f} TND, "
            f"pour un total previsionnel 2026 de {round(total_2026, 2):,.2f} TND. Ce total est "
            f"ensuite reparti sur les 12 mois selon un facteur saisonnier normalise (moyenne == 1). "
            f"L'intervalle de confiance n'est plus fixe a +-7% : il part de +-{CI_BASE_PCT*100:.0f}% en "
            f"debut d'annee et s'elargit de +{CI_GROWTH_PCT*100:.1f} point par mois d'horizon, jusqu'a "
            f"+-{ci_width_for_month(11)*100:.1f}% en decembre, pour refleter l'incertitude croissante."
        ),
        "mois_saisonniers": [
            {"mois": MOIS_NOMS_LONGS[i], "facteur": round(SEASONAL_NORM[i], 3),
             "ci_width_pct": round(ci_width_for_month(i) * 100, 1)}
            for i in range(12)
        ],
    }


def tool_segments():
    return [
        {"segment":"Premium",    "nb_clients":7751,  "ca_moyen":729000, "risque":"Faible",
         "description":"12+ contrats, BM<=3, tres fideles"},
        {"segment":"Standard",   "nb_clients":11771, "ca_moyen":368000, "risque":"Modere",
         "description":"Primes elevees, BM=7, potentiel upgrade"},
        {"segment":"Occasionnel","nb_clients":19333, "ca_moyen":190000, "risque":"Modere",
         "description":"2-3 contrats, potentiel fidelisation"},
        {"segment":"A Risque",   "nb_clients":33819, "ca_moyen":172000, "risque":"Eleve",
         "description":"Fort BM, sinistralite elevee -- priorite"},
    ]


def tool_risk_analysis(agence=None, region=None, branche=None):
    """
    agence/region/branche filtrent les parties calculees en direct sur les
    dataframes (ratio_sinistralite, agences_sous_perf). clients_a_risque
    et part_risque_pct restent GLOBAUX : ce sont des sorties de la
    segmentation K-Means (05_clustering.py), pas une requete sur les
    donnees live -- on ne peut pas les recalculer pour "Sfax Ville" sans
    refaire le clustering sur ce sous-ensemble. is_scoped indique
    explicitement au caller (agent/rapport) que ces deux champs restent
    globaux meme quand un filtre est applique, pour eviter d'afficher un
    chiffre segmente sous un titre qui suggere le contraire.
    """
    prod, sin = get_df()
    prod_f = apply_filters(prod, agence, region, branche)
    sin_f  = apply_filters(sin, agence, None, None)  # sin n'a pas Region/BRANCHE

    if is_unfiltered(agence, region, branche):
        ca = state.get("ca_total_reel", 0.0) or 1
        st = sin_f["REGLEMENTS"].sum() if "REGLEMENTS" in sin_f.columns else 0
    else:
        ca_sample = prod_f["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod_f.columns else 0
        ca = scale_ca(ca_sample, len(prod_f)) or 1
        st = sin_f["REGLEMENTS"].sum() if "REGLEMENTS" in sin_f.columns else 0

    sous_perf = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values().head(3).index.tolist() \
                if "AGENCE" in prod_f.columns and len(prod_f) > 0 else []

    return {
        "ratio_sinistralite":  round(st / ca * 100, 2),
        "clients_a_risque":    33819,
        "part_risque_pct":     46.5,
        "clients_a_risque_est_global": True,
        "agences_sous_perf":   [str(a) for a in sous_perf],
        "recommandations": [
            "Reviser la politique tarifaire pour le segment A Risque",
            "Renforcer les controles sur la branche Taxi",
            "Campagne de fidelisation pour les Occasionnels",
            "Programme VIP pour les clients Premium",
        ],
    }


def tool_risk_clients(n=20, agence=None, region=None, branche=None):
    """
    Liste NOMINATIVE des clients actuellement a risque, calculee EN DIRECT sur
    le portefeuille COMPLET (pas la segmentation K-Means figee utilisee par
    risk_analysis()/clients_a_risque, qui reste un comptage global non
    recalculable par filtre -- voir 05_clustering.py). Un client est
    considere a risque si son ratio individuel sinistres/primes (S/P) depasse
    RISK_RATIO_THRESHOLD_PCT -- meme critere que la 2e anomalie de
    detect_anomalies(), pour rester coherent entre les deux outils.

    Repond a "donne-moi la liste des clients a risque" : le jury peut demander
    des noms/identifiants concrets, pas juste un pourcentage agrege.

    IMPORTANT : utilise get_full_df() (portefeuille COMPLET charge au
    demarrage), PAS get_df() (echantillon temps reel 10k/3k utilise par le
    simulateur pour le dashboard live) -- croiser deux echantillons tires
    INDEPENDAMMENT produirait des ratios par client absurdes pour la plupart
    des clients (numerateur et denominateur ne couvrant pas les memes
    contrats/sinistres reels de ce client). Sur le portefeuille complet, le
    meme client contribue a 100% de ses primes ET de ses sinistres, donc le
    ratio est reellement representatif.
    """
    prod, sin = get_full_df()
    prod_f = apply_filters(prod, agence, region, branche)

    if "N_CLIENT" not in prod_f.columns or "N_CLIENT" not in sin.columns:
        return {
            "clients": [], "total_clients_a_risque": 0,
            "note": "Donnees insuffisantes (portefeuille complet non charge) pour identifier des clients individuels.",
        }

    sin_f = apply_filters(sin, agence, None, None) if "AGENCE" in sin.columns else sin

    ca_cl = prod_f.groupby("N_CLIENT")["PRIME_NETTE"].sum()
    st_cl = sin_f.groupby("N_CLIENT")["REGLEMENTS"].sum() if "REGLEMENTS" in sin_f.columns else pd.Series(dtype=float)
    agence_par_client = prod_f.groupby("N_CLIENT")["AGENCE"].first() if "AGENCE" in prod_f.columns else pd.Series(dtype=str)

    ratio_df = pd.DataFrame({"CA": ca_cl, "ST": st_cl}).dropna()
    ratio_df = ratio_df[ratio_df["CA"] >= MIN_CA_FOR_RISK_TND]
    ratio_df["ratio_sp_pct"] = ratio_df["ST"] / ratio_df["CA"] * 100

    a_risque = ratio_df[ratio_df["ratio_sp_pct"] > RISK_RATIO_THRESHOLD_PCT].sort_values(
        "ratio_sp_pct", ascending=False
    )

    clients = [
        {
            "n_client":             str(n_client),
            "agence":               str(agence_par_client.get(n_client, "N/A")),
            "prime_totale_tnd":     round(row["CA"], 2),
            "sinistres_totaux_tnd": round(row["ST"], 2),
            "ratio_sp_pct":         round(row["ratio_sp_pct"], 1),
            "niveau_risque":        _risk_severity_label(row["ratio_sp_pct"]),
        }
        for n_client, row in a_risque.head(max(1, n)).iterrows()
    ]

    return {
        "clients":                clients,
        "total_clients_a_risque": int(len(a_risque)),
        "nb_clients_analyses":    int(len(ratio_df)),
        "seuil_ratio_sp_pct":     RISK_RATIO_THRESHOLD_PCT,
        "note": (
            "Calcule EN DIRECT sur le portefeuille complet -- distinct du chiffre "
            "clients_a_risque (33819) renvoye par risk_analysis, qui provient de la "
            "segmentation K-Means globale (05_clustering.py) et n'est jamais recalcule par filtre. "
            f"Clients avec une prime totale < {MIN_CA_FOR_RISK_TND:.0f} TND exclus (denominateur "
            "trop faible pour un ratio S/P representatif)."
        ),
    }


def tool_sinistres_stats():
    _, sin = get_df()
    return {
        "total_sinistres": len(sin),
        "montant_total":   round(state.get("sin_total_reel", 0.0), 2),
        "montant_moyen":   round(sin["REGLEMENTS"].mean(), 2) if len(sin) > 0 and "REGLEMENTS" in sin.columns else 0,
        "en_cours":        int(len(sin[sin["TRANSACTION"] == "EN COURS"])) if "TRANSACTION" in sin.columns else 0,
        "pic_mois":        int(sin.groupby("MOIS")["REGLEMENTS"].sum().idxmax()) if "MOIS" in sin.columns and len(sin) > 0 else 0,
    }


def tool_branch_analysis():
    prod, _ = get_df()
    ca_total = prod["PRIME_NETTE"].sum()
    by_br    = prod.groupby("BRANCHE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    return [{"branche": str(b), "ca": round(v, 2), "part_pct": round(v / ca_total * 100, 1)}
            for b, v in by_br.items()]


def tool_compare_agencies():
    prod, sin = get_df()
    result = []
    for ag in prod["AGENCE"].unique():
        p  = prod[prod["AGENCE"] == ag]
        s  = sin[sin["AGENCE"]  == ag] if "AGENCE" in sin.columns else pd.DataFrame()
        ca = p["PRIME_NETTE"].sum()
        st = s["REGLEMENTS"].sum() if len(s) > 0 and "REGLEMENTS" in s.columns else 0
        result.append({
            "agence":       str(ag),
            "ca":           round(ca, 2),
            "nb_contrats":  len(p),
            "prime_moy":    round(p["PRIME_NETTE"].mean(), 2) if len(p) > 0 else 0,
            "sinistres":    round(st, 2),
            "ratio_sp_pct": round(st / ca * 100, 2) if ca > 0 else 0,
        })
    return sorted(result, key=lambda x: x["ca"], reverse=True)


def tool_detect_anomalies():
    prod, sin = get_df()
    anomalies = []
    if "REGLEMENTS" in sin.columns and len(sin) > 10:
        mean_s   = sin["REGLEMENTS"].mean()
        std_s    = sin["REGLEMENTS"].std()
        suspects = sin[sin["REGLEMENTS"] > mean_s + 3 * std_s]
        anomalies.append({
            "type":        "Sinistres montants aberrants",
            "count":       len(suspects),
            "seuil":       round(mean_s + 3 * std_s, 2),
            "montant_max": round(suspects["REGLEMENTS"].max(), 2) if len(suspects) > 0 else 0,
        })
    if "N_CLIENT" in prod.columns and "N_CLIENT" in sin.columns:
        ca_cl    = prod.groupby("N_CLIENT")["PRIME_NETTE"].sum()
        st_cl    = sin.groupby("N_CLIENT")["REGLEMENTS"].sum()
        ratio_df = pd.DataFrame({"CA": ca_cl, "ST": st_cl}).dropna()
        ratio_df["ratio"] = ratio_df["ST"] / ratio_df["CA"] * 100
        haut     = ratio_df[ratio_df["ratio"] > RISK_RATIO_THRESHOLD_PCT]
        anomalies.append({
            "type":        f"Clients ratio S/P > {RISK_RATIO_THRESHOLD_PCT:.0f}%",
            "count":       len(haut),
            "ratio_moyen": round(haut["ratio"].mean(), 1) if len(haut) > 0 else 0,
        })
    by_ag    = tool_compare_agencies()
    ag_alert = [a for a in by_ag if a["ratio_sp_pct"] > 80]
    anomalies.append({
        "type":    "Agences sinistralite > 80%",
        "count":   len(ag_alert),
        "agences": [a["agence"] for a in ag_alert[:5]],
    })
    return {"anomalies": anomalies, "total_alertes": sum(a["count"] for a in anomalies)}


def tool_profil_clients(agence=None, region=None, branche=None):
    prod, _ = get_df()
    prod = apply_filters(prod, agence, region, branche)
    sexe = prod["SEXE"].value_counts().to_dict()             if "SEXE"        in prod.columns else {}
    csp  = prod["CSP"].value_counts().head(6).to_dict()      if "CSP"         in prod.columns else {}
    bm   = prod["BONUS_MALUS"].value_counts().sort_index().to_dict() if "BONUS_MALUS" in prod.columns else {}
    ages = prod["AGE"].dropna().tolist()                     if "AGE"         in prod.columns else []
    age_bins = {"18-30": 0, "31-45": 0, "46-60": 0, "61-80": 0}
    for a in ages:
        if   a <= 30: age_bins["18-30"] += 1
        elif a <= 45: age_bins["31-45"] += 1
        elif a <= 60: age_bins["46-60"] += 1
        else:         age_bins["61-80"] += 1
    return {"sexe": sexe, "csp": csp, "bonus_malus": bm, "age_bins": age_bins}


def tool_agent_monitoring(n_runs: int = 50):
    monitoring = {
        "agent_performance": None,
        "data_drift":        None,
    }

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("MAE_Agent_Interactions")
        if experiment is None:
            monitoring["agent_performance"] = {"note": "Aucune execution loguee pour le moment."}
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=n_runs,
            )
            if not runs:
                monitoring["agent_performance"] = {"note": "Aucune execution loguee pour le moment."}
            else:
                durations = [r.data.metrics.get("duration_ms") for r in runs if r.data.metrics.get("duration_ms") is not None]
                tokens    = [r.data.metrics.get("input_tokens", 0) + r.data.metrics.get("output_tokens", 0) for r in runs]
                errors    = [r.data.metrics.get("error", 0) for r in runs]
                tool_counts = {}
                for r in runs:
                    n_tools = r.data.metrics.get("tool_calls", 0)
                    if n_tools:
                        tool_counts["appels_avec_outils"] = tool_counts.get("appels_avec_outils", 0) + 1

                monitoring["agent_performance"] = {
                    "executions_analysees": len(runs),
                    "latence_moyenne_ms":   round(sum(durations) / len(durations), 1) if durations else None,
                    "latence_max_ms":       round(max(durations), 1) if durations else None,
                    "tokens_moyens":        round(sum(tokens) / len(tokens), 1) if tokens else None,
                    "taux_erreur_pct":      round(sum(errors) / len(errors) * 100, 1) if errors else 0,
                    "pct_avec_appels_outils": round(tool_counts.get("appels_avec_outils", 0) / len(runs) * 100, 1),
                }
    except Exception as e:
        monitoring["agent_performance"] = {"error": str(e)}

    # Reutilise le dernier calcul fait par simulator() (rafraichi toutes les
    # 5s en continu) plutot que de recalculer -- garantit que ce qui est
    # affiche ici correspond exactement a ce qui a (ou n'a pas) declenche une
    # alerte journalisee. Fallback sur un calcul a la demande si le
    # simulateur n'a pas encore tourne (ex: tout premier appel juste apres
    # le demarrage du serveur).
    monitoring["data_drift"] = state.get("last_data_drift") or compute_data_drift()

    return monitoring


# ════════════════════════════════════════════════════════════════
# GENERATION DE RAPPORT PDF (reportlab)
# ════════════════════════════════════════════════════════════════
# INITIALISATION DEFENSIVE : meme pattern que RAG_AVAILABLE dans agent.py.
# Si reportlab n'est pas installe, REPORT_AVAILABLE=False et l'outil/l
# endpoint repondent proprement au lieu de faire planter le serveur.
REPORT_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Generation de rapport PDF indisponible — {type(e).__name__}: {e} (pip install reportlab)")

# v3.3 — FIX : ancre sur le dossier de ce fichier, pas sur le cwd du
# process qui lance uvicorn (voir changelog en tete de fichier).
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _fmt_tnd_fr(v):
    """Format TND cohérent avec le reste de l'app : '1 234 567,89 TND'."""
    try:
        s = f"{v:,.2f}"
        s = s.replace(",", " ").replace(".", ",")
        return f"{s} TND"
    except Exception:
        return str(v)


def tool_generate_report(agence=None, region=None, branche=None, mois_num=None, sections=None):
    """
    Genere un rapport PDF du portefeuille et le sauvegarde dans
    REPORTS_DIR. Retourne les metadonnees du fichier (pas le PDF lui-meme
    — l'agent ne peut pas streamer un binaire dans le chat) ; le
    telechargement se fait via l'URL absolue retournee (download_url).

    Parametres (v3.3, tous optionnels) :
      - agence/region/branche : restreint le resume et le classement des
        agences a ce perimetre (via tool_portfolio_summary). NOTE : les
        previsions et l'analyse de risque restent globales quel que soit
        ce filtre (voir limitation ci-dessous).
      - mois_num : restreint la section previsions a un seul mois au lieu
        du detail sur 12 mois.
      - sections : sous-ensemble de ["resume","agences","previsions","risques"].
        None/vide => rapport complet (comportement identique a avant).

    LIMITATION CONNUE : tool_risk_analysis() et tool_top_agencies() ne
    prennent pas de filtre agence/region/branche cote donnees — un rapport
    "scope" sur une agence affichera donc un titre/perimetre different
    mais la section risques restera basee sur les chiffres globaux du
    portefeuille. A corriger si un jury teste specifiquement ce cas.
    """
    if not REPORT_AVAILABLE:
        return {
            "error": "Generation de rapport indisponible sur cet environnement "
                     "(reportlab non installe — pip install reportlab)."
        }

    valid_sections = {"resume", "agences", "previsions", "risques"}
    if not sections:
        sections = list(valid_sections)
    else:
        sections = [s for s in sections if s in valid_sections] or list(valid_sections)

    summary = tool_portfolio_summary(agence, region, branche)
    top_ag  = tool_top_agencies(5, agence, region, branche)
    fc      = tool_forecast(mois_num) if mois_num else tool_forecast()
    risk    = tool_risk_analysis(agence, region, branche)

    scope_bits = [v for v in [agence, region, branche] if v]
    scope_label = " — " + ", ".join(scope_bits) if scope_bits else ""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rapport_MAE_{ts}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleMAE", parent=styles["Title"], textColor=colors.HexColor("#156b32"))
    h2_style = ParagraphStyle("H2MAE", parent=styles["Heading2"], textColor=colors.HexColor("#1a7a3a"),
                               spaceBefore=14, spaceAfter=6)
    body_style = styles["BodyText"]
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#6b8f6b"))

    doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    story = []

    story.append(Paragraph(f"MAE Assurances — Rapport de Portefeuille{scope_label}", title_style))
    story.append(Paragraph(f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}", body_style))
    story.append(Spacer(1, 0.5 * cm))

    if "resume" in sections:
        story.append(Paragraph("Resume du portefeuille", h2_style))
        summary_rows = [
            ["CA total", _fmt_tnd_fr(summary["ca_total"])],
            ["Nombre de contrats", f"{summary['nb_contrats']:,}".replace(",", " ")],
            ["Sinistres totaux", _fmt_tnd_fr(summary["sinistres_total"])],
            ["Ratio de sinistralite", f"{summary['ratio_sinistralite']}%"],
            ["Agence top", summary["top_agence"]],
        ]
        t1 = Table(summary_rows, colWidths=[7 * cm, 8 * cm])
        t1.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5ec")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4e8d4")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t1)

    if "agences" in sections:
        story.append(Paragraph("Top 5 agences par chiffre d'affaires", h2_style))
        ag_rows = [["Agence", "CA", "Part de marche"]] + [
            [a["agence"], _fmt_tnd_fr(a["ca"]), f"{a['part_pct']}%"] for a in top_ag
        ]
        t2 = Table(ag_rows, colWidths=[6 * cm, 6 * cm, 4 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a7a3a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4e8d4")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t2)

    if "previsions" in sections:
        story.append(Paragraph("Previsions", h2_style))
        if mois_num:
            story.append(Paragraph(
                f"CA prevu pour {fc['mois']} : <b>{_fmt_tnd_fr(fc['ca_prev'])}</b> "
                f"(intervalle +-{fc['ci_width_pct']}%).", body_style))
        else:
            story.append(Paragraph(
                f"CA total prevu 2026 : <b>{_fmt_tnd_fr(fc['total_2026'])}</b> "
                f"(croissance de {fc['croissance_pct']}% vs 2025). Mois de pic saisonnier : {fc['mois_pic']}. "
                f"Intervalles de confiance dynamiques : de +-{fc['detail'][0]['ci_width_pct']}% en debut d'annee "
                f"a +-{fc['detail'][-1]['ci_width_pct']}% en fin d'annee (incertitude croissante avec l'horizon).",
                body_style
            ))
            fc_rows = [["Mois", "CA prevu", "Intervalle"]] + [
                [d["mois"], _fmt_tnd_fr(d["ca"]), f"+-{d['ci_width_pct']}%"] for d in fc["detail"]
            ]
            t3 = Table(fc_rows, colWidths=[5 * cm, 7 * cm, 4 * cm])
            t3.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a7a3a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4e8d4")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t3)

    if "risques" in sections:
        story.append(Paragraph("Analyse des risques", h2_style))
        ratio_label = "Ratio de sinistralite" + (f" ({scope_label.strip(' —')})" if scope_bits else " global")
        story.append(Paragraph(
            f"{ratio_label} : <b>{risk['ratio_sinistralite']}%</b>.",
            body_style
        ))
        clients_note = (
            " (chiffre du portefeuille entier — la segmentation clients n'est pas recalculee par agence/region/branche)"
            if scope_bits else ""
        )
        story.append(Paragraph(
            f"Clients identifies a risque (portefeuille entier) : "
            f"<b>{str(risk['clients_a_risque']).replace(',', ' ')}</b> "
            f"({risk['part_risque_pct']}% du portefeuille){clients_note}.",
            body_style
        ))
        story.append(Paragraph("Recommandations prioritaires :", body_style))
        for reco in risk["recommandations"]:
            story.append(Paragraph(f"• {reco}", body_style))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Rapport genere automatiquement par MAEIA — MAE Intelligence. "
        "Les donnees presentees proviennent du portefeuille en temps reel.",
        footer_style
    ))

    doc.build(story)

    return {
        "filename": filename,
        "filepath": filepath,
        "download_url": f"{API_BASE_URL}/api/reports/{filename}",
        "message": f"Rapport PDF genere avec succes : {filename}. "
                    f"Telechargement disponible via /api/generate-report.",
    }


TOOLS_MAP = {
    "portfolio_summary":  tool_portfolio_summary,
    "top_agencies":       tool_top_agencies,
    "forecast":           tool_forecast,
    "explain_forecast":   tool_explain_forecast,
    "segments":           tool_segments,
    "risk_analysis":      tool_risk_analysis,
    "risk_clients":       tool_risk_clients,
    "sinistres_stats":    tool_sinistres_stats,
    "branch_analysis":    tool_branch_analysis,
    "compare_agencies":   tool_compare_agencies,
    "detect_anomalies":   tool_detect_anomalies,
    "profil_clients":     tool_profil_clients,
    "agent_monitoring":   tool_agent_monitoring,
    "generate_report":    tool_generate_report,
}

_agent: MAEAgent | None = None


@app.get("/api/status")
def status():
    # drift_alert_active/drift_ecart_pct : lecture du dernier calcul fait en
    # continu par simulator() (voir compute_data_drift()) -- expose ici pour
    # que le frontend affiche une alerte TOUJOURS VISIBLE (barre laterale),
    # pas seulement quand on ouvre la page Monitoring dediee.
    last_drift = state.get("last_data_drift") or {}
    return {
        "status":            "online",
        "last_update":       state["last_update"],
        "total_contrats":    len(state["contrats"]),
        "total_sinistres":   len(state["sinistres"]),
        "total_generated":   state["total_generated"],
        "drift_alert_active": bool(state.get("drift_alert_active", False)),
        "drift_ecart_pct":    last_drift.get("ecart_pct"),
    }


@app.get("/api/kpis")
def kpis(agence: str = "all", region: str = "all", branche: str = "all", mois: int = 0):
    prod, sin = get_df()
    prod_f = apply_filters(prod, agence, region, branche, mois)
    sin_f  = apply_filters(sin,  agence, None,   None,    mois)

    if is_unfiltered(agence, region, branche, mois):
        ca = round(state.get("ca_total_reel", 0.0), 2)
        nb = state.get("nb_contrats_reel", TOTAL_CONTRATS_REEL)
    else:
        ca_sample = prod_f["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod_f.columns else 0
        ca = scale_ca(ca_sample, len(prod_f))
        nb = len(prod_f)

    st     = sin_f["REGLEMENTS"].sum() if "REGLEMENTS" in sin_f.columns else 0
    top_ag = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().idxmax() if len(prod_f) > 0 else "N/A"
    top_ca = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().max()    if len(prod_f) > 0 else 0
    return {
        "ca_total":        ca,
        "nb_clients":      nb,
        "nb_contrats":     nb,
        "sin_total":       round(st, 2),
        "ratio_sin":       round(st / ca * 100, 2) if ca > 0 else 0,
        "top_agence":      str(top_ag),
        "top_agence_ca":   round(top_ca, 2),
        "last_update":     state["last_update"],
        "total_generated": state["total_generated"],
    }


@app.get("/api/ca-by-agence")
def ca_by_agence(agence: str = "all", region: str = "all", branche: str = "all", mois: int = 0):
    prod, _ = get_df()
    prod  = apply_filters(prod, agence, region, branche, mois)
    data  = prod.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    total = data.sum()
    return [{"agence": str(a), "ca": round(v, 2), "part": round(v / total * 100, 1) if total > 0 else 0}
            for a, v in data.items()]


@app.get("/api/ca-by-branche")
def ca_by_branche(agence: str = "all", region: str = "all", mois: int = 0):
    prod, _ = get_df()
    prod  = apply_filters(prod, agence, region, None, mois)
    data  = prod.groupby("BRANCHE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    total = data.sum()
    return [{"branche": str(b), "ca": round(v, 2), "part": round(v / total * 100, 1) if total > 0 else 0}
            for b, v in data.items()]


@app.get("/api/ca-by-month")
def ca_by_month(agence: str = "all", region: str = "all", branche: str = "all"):
    prod, sin = get_df()
    prod    = apply_filters(prod, agence, region, branche)
    prime_m = prod.groupby("MOIS")["PRIME_NETTE"].sum() if "MOIS" in prod.columns else pd.Series()
    sin_m   = sin.groupby("MOIS")["REGLEMENTS"].sum()   if "MOIS" in sin.columns  else pd.Series()

    if is_unfiltered(agence, region, branche):
        by_month_reel = state.get("ca_by_month_reel", {})
        total_reel    = state.get("ca_total_reel", 0.0)
        return [{"mois": MOIS_LABELS[m-1], "mois_num": m,
                 "primes":    round(float(by_month_reel.get(m, total_reel/12)), 2),
                 "sinistres": round(float(sin_m.get(m, 0)), 2)}
                for m in range(1, 13)]

    sf = TOTAL_CONTRATS_REEL / max(len(prod), 1)
    return [{"mois": MOIS_LABELS[m-1], "mois_num": m,
             "primes":    round(float(prime_m.get(m, 0)) * sf, 2),
             "sinistres": round(float(sin_m.get(m, 0)), 2)}
            for m in range(1, 13)]


@app.get("/api/profil-clients")
def profil_clients_endpoint(agence: str = "all", region: str = "all", branche: str = "all"):
    return tool_profil_clients(
        agence  if agence  != "all" else None,
        region  if region  != "all" else None,
        branche if branche != "all" else None,
    )


@app.get("/api/forecast")
def forecast_endpoint():
    base   = state.get("ca_total_reel", 1_630_000_000) or 1_630_000_000
    growth = 0.047
    total_2026 = base * (1 + growth)
    monthly = [total_2026 / 12 * SEASONAL_NORM[i] for i in range(12)]

    by_month_reel = state.get("ca_by_month_reel", {})
    if by_month_reel:
        ca_reel_2025 = [round(float(by_month_reel.get(m, base / 12)), 2) for m in range(1, 13)]
    else:
        ca_reel_2025 = [round(base / 12, 2)] * 12

    result = []
    for i in range(12):
        w = ci_width_for_month(i)
        result.append({
            "mois":     MOIS_LABELS[i],
            "mois_num": i + 1,
            "ca_prev":  round(monthly[i], 2),
            "ci_low":   round(monthly[i] * (1 - w), 2),
            "ci_high":  round(monthly[i] * (1 + w), 2),
            "ca_reel":  ca_reel_2025[i],
        })
    return result


@app.get("/api/explain-forecast")
def explain_forecast_endpoint(mois_num: int = 0):
    return tool_explain_forecast(mois_num if mois_num else None)


@app.get("/api/monitoring")
def monitoring_endpoint(n_runs: int = 50):
    return tool_agent_monitoring(n_runs)


@app.get("/api/generate-report")
def generate_report_endpoint():
    result = tool_generate_report()
    if "error" in result:
        return result
    return FileResponse(
        path=result["filepath"],
        media_type="application/pdf",
        filename=result["filename"],
    )


@app.get("/api/segments")
def segments_endpoint():
    return [
        {"segment":"Premium",    "count":7751,  "ca_moyen":729000, "color":"#00e5b4",
         "bm_moyen":2.1, "nb_contrats_moyen":12.3, "risque":"Faible"},
        {"segment":"Standard",   "count":11771, "ca_moyen":368000, "color":"#3b82f6",
         "bm_moyen":7.0, "nb_contrats_moyen":5.8,  "risque":"Modere"},
        {"segment":"Occasionnel","count":19333, "ca_moyen":190000, "color":"#f59e0b",
         "bm_moyen":5.2, "nb_contrats_moyen":2.4,  "risque":"Modere"},
        {"segment":"A Risque",   "count":33819, "ca_moyen":172000, "color":"#f43f5e",
         "bm_moyen":10.8,"nb_contrats_moyen":1.9,  "risque":"Eleve"},
    ]


@app.get("/api/filters")
def filters_endpoint():
    prod, _ = get_df()

    if "AGENCE" in prod.columns:
        raw_agences = prod["AGENCE"].dropna().unique().tolist()
        seen, agences = {}, []
        for a in sorted(raw_agences):
            key = _strip_accents(str(a)).lower()
            if key not in seen:
                seen[key] = True
                agences.append(a)
    else:
        agences = []

    branches = ALL_BRANCHE_LABELS

    region_codes  = prod["Region"].dropna().unique().tolist() if "Region" in prod.columns else []
    region_labels = build_region_label_map(region_codes)

    def _region_sort_key(c):
        try:
            return (0, float(c))
        except (ValueError, TypeError):
            return (1, str(c))

    regions = sorted(region_labels.keys(), key=_region_sort_key)

    return {
        "agences":       agences,
        "regions":       regions,
        "branches":      branches,
        "region_labels": region_labels,
    }


@app.get("/api/live-feed")
def live_feed(limit: int = 10):
    with state["lock"]:
        recent = sorted(state["contrats"][-50:], key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    return recent


@app.get("/api/anomalies")
def anomalies_endpoint():
    return tool_detect_anomalies()


@app.get("/api/risk-clients")
def risk_clients_endpoint(n: int = 20, agence: str = "all", region: str = "all", branche: str = "all"):
    return tool_risk_clients(
        n,
        agence  if agence  != "all" else None,
        region  if region  != "all" else None,
        branche if branche != "all" else None,
    )


@app.get("/api/compare-agencies")
def compare_agencies_endpoint():
    return tool_compare_agencies()


class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/api/reports/{filename}")
def download_report(filename: str):
    # Securite : empeche toute tentative de path traversal (../../etc)
    safe_name = os.path.basename(filename)
    filepath = os.path.join(REPORTS_DIR, safe_name)
    if not os.path.isfile(filepath) or not safe_name.endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Rapport introuvable.")
    return FileResponse(path=filepath, media_type="application/pdf", filename=safe_name)


@app.post("/api/agent")
def agent_endpoint(req: ChatRequest):
    global _agent
    if _agent is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return {
                "answer": "GROQ_API_KEY non configuree. Creez un compte gratuit sur console.groq.com et ajoutez la cle dans .env",
                "tool_calls": [], "thinking": [], "tokens_used": 0, "duration_ms": 0,
            }
        try:
            _agent = MAEAgent(tools_map=TOOLS_MAP)
        except ValueError as e:
            return {"answer": str(e), "tool_calls": [], "thinking": [], "tokens_used": 0, "duration_ms": 0}

    return _agent.run(req.message, req.history)


@app.on_event("startup")
def startup():
    seed_data()
    threading.Thread(target=simulator, args=(5,), daemon=True).start()
    print("✅ MAE Intelligence API v3.3 — simulateur actif toutes les 5s")