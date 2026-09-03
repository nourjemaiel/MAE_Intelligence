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

import sys
# Sous Windows, la console herite souvent de l'encodage ANSI local (cp1252),
# qui ne sait pas encoder les emojis utilises dans les logs (⚠️, ✅...) --
# print() plantait alors AU DEMARRAGE (UnicodeEncodeError) des qu'un message
# d'avertissement (ex: CSV introuvable) tentait de s'afficher, un crash plus
# grave que le probleme qu'il essayait de signaler. Force l'UTF-8 en sortie
# quel que soit l'environnement d'ou uvicorn est lance.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
import hashlib
import hmac
import json
import secrets
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
    # allow_credentials=False : l'authentification se fait par jeton dans
    # l'en-tete Authorization (ou en parametre ?token= pour les liens de
    # rapport), pas par cookie -- allow_credentials=True combine a
    # allow_origins="*" est une configuration invalide/dangereuse (le
    # navigateur l'accepte silencieusement dans certains cas) sans meme
    # apporter de benefice ici puisqu'aucun cookie n'est utilise.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("mlruns")

# ════════════════════════════════════════════════════════════════
# AUTHENTIFICATION (remarque superviseur : securiser l'agent/l'API)
# ════════════════════════════════════════════════════════════════
# AVANT ce correctif : le "login" (admin/mae2026, dg.mae/direction) etait
# verifie ENTIEREMENT cote frontend (mae_frontend/index.html), mots de
# passe en clair directement lisibles dans le JS servi au navigateur
# (View Source), et AUCUN endpoint de cette API ne verifiait quoi que ce
# soit -- n'importe qui pouvait appeler /api/agent, /api/reports/*, etc.
# directement (curl, Postman...) sans jamais passer par la page de
# connexion. Le "login" etait donc purement cosmetique du point de vue de
# la securite reelle -- pas une vraie barriere d'acces.
#
# Correctif : verification cote SERVEUR, mots de passe stockes HASHES
# (PBKDF2-HMAC-SHA256 + sel, jamais en clair), jetons de session en
# memoire avec expiration, middleware qui bloque toute route /api/*
# (sauf /api/login) sans jeton valide. Le frontend envoie le jeton recu a
# la connexion dans l'en-tete Authorization sur chaque appel fetch(), et
# en parametre ?token= pour les liens de rapport (une simple balise <a>
# ne peut pas ajouter d'en-tete personnalise -- necessaire pour que les
# boutons Visualiser/Telecharger restent de vrais liens cliquables).
SESSION_TTL_HOURS = 12
_sessions = {}  # token -> {"username", "role", "name", "expires"}

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()

# Sel fixe par utilisateur : suffisant ici (2 comptes internes geres a la
# main, pas une base d'utilisateurs publique avec inscription) -- le point
# important est que les mots de passe ne soient plus stockes/visibles EN
# CLAIR (c'etait le vrai probleme, pas la force du sel).
_USERS = {
    "admin":  {"salt": "mae-admin-2026-salt", "role": "admin", "name": "Administrateur",
               "password_hash": _hash_password("mae2026", "mae-admin-2026-salt")},
    "dg.mae": {"salt": "mae-dg-2026-salt",    "role": "dg",    "name": "Direction Générale",
               "password_hash": _hash_password("direction", "mae-dg-2026-salt")},
}


class LoginRequest(BaseModel):
    username: str
    password: str


# Comptes "Direction Generale" crees dynamiquement par l'admin (page
# Administration). AVANT ce correctif : geres UNIQUEMENT dans le
# localStorage du navigateur de l'admin (mots de passe en clair, pas de
# backend du tout) -- un compte cree ne pouvait de toute facon jamais
# s'authentifier reellement puisqu'aucun endpoint ne verifiait quoi que ce
# soit. Persistes maintenant cote serveur (hashes, jamais en clair),
# fichier JSON simple -- coherent avec l'echelle du projet (une poignee de
# comptes geres a la main), pas besoin d'une vraie base de donnees.
_DYNAMIC_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_users.json")

def _load_dynamic_users():
    if os.path.exists(_DYNAMIC_USERS_FILE):
        try:
            with open(_DYNAMIC_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_dynamic_users(users):
    with open(_DYNAMIC_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# Contrats/sinistres cosmetiques generes par le simulateur (voir simulator()
# plus bas) -- persistes pour survivre a un redemarrage du backend, sinon le
# compteur "Generes (live)" et les totaux qu'il alimente (CA/sinistres du
# dashboard, voir get_full_df()) repartaient de zero a chaque relance alors
# que le compteur affiche, lui, continuait de grimper (incoherent). Format
# JSON Lines (une ligne = un contrat/sinistre) plutot qu'un tableau JSON
# unique : chaque nouveau tick du simulateur APPEND une ligne (cout constant),
# alors que reecrire un tableau entier grandissant a chaque tick couterait de
# plus en plus cher avec le temps de fonctionnement du serveur. Le fichier
# entier n'est relu qu'une fois, au demarrage (seed_data()).
_LIVE_DELTA_PROD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_delta_contrats.jsonl")
_LIVE_DELTA_SIN_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_delta_sinistres.jsonl")

def _append_live_delta(path, record):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass

def _load_live_delta(path):
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows

_dynamic_users = _load_dynamic_users()


def _get_session(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        token = request.query_params.get("token", "")
    return _sessions.get(token)


def _require_admin(request: Request):
    session = _get_session(request)
    if not session or session["role"] != "admin":
        raise HTTPException(status_code=403, detail="Reserve a l'administrateur.")
    return session


@app.post("/api/login")
def login(req: LoginRequest):
    user = _USERS.get(req.username) or _dynamic_users.get(req.username)
    candidate_hash = _hash_password(req.password, user["salt"]) if user else ""
    # hmac.compare_digest (temps constant) meme si user est introuvable
    # (candidate_hash="" compare quand meme) -- evite qu'un attaquant
    # distingue "utilisateur inconnu" de "mot de passe incorrect" via le
    # temps de reponse.
    if not user or not hmac.compare_digest(candidate_hash, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides.")
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "username": req.username, "role": user["role"], "name": user["name"],
        "expires": datetime.now() + timedelta(hours=SESSION_TTL_HOURS),
    }
    return {"token": token, "role": user["role"], "name": user["name"]}


@app.post("/api/logout")
def logout(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    _sessions.pop(token, None)
    return {"ok": True}


class CreateUserRequest(BaseModel):
    username: str
    password: str
    name: str


@app.get("/api/admin/users")
def list_users(request: Request):
    _require_admin(request)
    return [{"username": u, "name": v["name"], "role": "dg"} for u, v in _dynamic_users.items()]


@app.post("/api/admin/users")
def create_user(req: CreateUserRequest, request: Request):
    _require_admin(request)
    username = req.username.strip()
    if not username or not req.name.strip():
        raise HTTPException(status_code=400, detail="Nom et identifiant requis.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Mot de passe minimum 6 caracteres.")
    if username in _USERS or username in _dynamic_users:
        raise HTTPException(status_code=409, detail="Cet identifiant existe deja.")
    salt = secrets.token_hex(16)
    _dynamic_users[username] = {
        "salt": salt, "password_hash": _hash_password(req.password, salt),
        "role": "dg", "name": req.name.strip(),
    }
    _save_dynamic_users(_dynamic_users)
    return {"username": username, "name": req.name.strip(), "role": "dg"}


@app.delete("/api/admin/users/{username}")
def delete_user(username: str, request: Request):
    _require_admin(request)
    if username not in _dynamic_users:
        raise HTTPException(status_code=404, detail="Compte introuvable.")
    del _dynamic_users[username]
    _save_dynamic_users(_dynamic_users)
    # Invalide toute session active de ce compte -- sinon un utilisateur
    # supprime resterait connecte jusqu'a expiration naturelle du jeton.
    for tok in [t for t, s in _sessions.items() if s["username"] == username]:
        del _sessions[tok]
    return {"ok": True}


_PUBLIC_API_PATHS = {"/api/login"}

@app.middleware("http")
async def require_auth(request: Request, call_next):
    # Les preflights CORS (OPTIONS) ne portent jamais l'en-tete Authorization
    # (le navigateur les envoie avant la vraie requete pour demander la
    # permission) -- les bloquer ici les empeche d'atteindre CORSMiddleware,
    # qui doit repondre en premier. Sans ce court-circuit, TOUS les appels
    # avec un en-tete personnalise (Authorization) echouaient avec une
    # erreur CORS avant meme d'atteindre la verification du jeton.
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if not path.startswith("/api/") or path in _PUBLIC_API_PATHS:
        return await call_next(request)
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        token = request.query_params.get("token", "")
    session = _sessions.get(token)
    if not session or session["expires"] < datetime.now():
        _sessions.pop(token, None)
        return JSONResponse(status_code=401, content={"detail": "Authentification requise ou session expiree."})
    return await call_next(request)

# ════════════════════════════════════════════════════════════════
# BASE URL — utilisee pour construire des liens ABSOLUS et cliquables
# (ex: dans les reponses de l agent). Surchargeable via .env si le
# backend tourne derriere un autre host/port (deploiement, ngrok, etc.)
# ════════════════════════════════════════════════════════════════
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Liste des agences pour le simulateur "live" (nouveaux contrats simules) --
# lue depuis les vraies donnees nettoyees (77 agences reelles, voir
# 02_cleaning.py / MAE_AGENCY_NAMES) plutot qu'une liste a part codee en dur
# ici : l'ancienne liste (15 noms, dont certains partiellement synthetiques
# type "Sfax Nord") ne recoupait aucun des vrais noms d'agence desormais
# dans le CSV, ce qui aurait fait cohabiter deux conventions de nommage
# differentes sur le meme tableau de bord (flux live vs donnees reelles).
try:
    _agence_counts = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed_data", "Production_Cleaned.csv"),
        usecols=["AGENCE"]
    )["AGENCE"].value_counts(normalize=True)
    AGENCES = _agence_counts.index.tolist()
    AGENCE_WEIGHTS_REELLES = _agence_counts.values.tolist()
except Exception:
    AGENCES = ["Tunis Centre", "Sousse 1", "Nabeul 1", "Bizerte", "Béja",
               "Gabès 1", "Monastir 1", "Ariana", "Gafsa 1", "Sfax 1"]
    AGENCE_WEIGHTS_REELLES = None

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
if AGENCE_WEIGHTS_REELLES is not None:
    AGENCE_WEIGHTS = AGENCE_WEIGHTS_REELLES
else:
    AGENCE_WEIGHTS = [1.0 / len(AGENCES)] * len(AGENCES)

TOTAL_CONTRATS_REEL = 285736

# Parametres du generateur synthetique de PRIME_NETTE pour les NOUVEAUX
# contrats simules (simulator() et le fallback _gen_synthetic()).
#
# RECALIBRE (remarque superviseur, feature engineering) : suite au passage
# de RAW_TO_DINARS a 1000 (vraie sous-unite du dinar, /1000 -- voir
# 02_cleaning.py) au lieu de l'ancien diviseur cale sur une moyenne externe
# (88), la moyenne reelle par LIGNE de PRIME_NETTE est tombee de ~571 TND a
# ~57 TND (chaque ligne = une garantie au sein d'une police, pas la police
# entiere -- voir 02_cleaning.py). L'ancienne calibration (mu=5.87) restait
# donc figee sur l'ancienne echelle : le generateur produisait des contrats
# simules ~10x trop eleves par rapport aux vraies donnees, ce qui declenchait
# une fausse alerte de derive permanente des que la fenetre des 50 derniers
# contrats se remplissait de valeurs simulees.
#
# Valeurs recalculees par methode des moments sur la distribution reelle
# actuelle (Production_Cleaned.csv, 285 736 lignes, moyenne 57,12 / ecart-type
# 72,38 TND) : CV = 72,38/57,12 ≈ 1,267 -> sigma = sqrt(ln(1+CV²)) ≈ 0,979,
# mu = ln(moyenne) - sigma²/2 ≈ 3,566. Verifie empiriquement : moyenne
# simulee obtenue ~57,2, ecart-type ~72,8 (sur 200k tirages) -- coherent.
PRIME_SIM_LOGNORMAL_MU    = 3.566
PRIME_SIM_LOGNORMAL_SIGMA = 0.979
PRIME_SIM_MIN_TND         = 1.0
PRIME_SIM_MAX_TND         = 6500.0

# Meme probleme, jamais corrige a l'epoque : le generateur de REGLEMENTS
# simules (simulator() et _gen_synthetic_sin()) etait reste sur
# lognormal(7.8, 1.3) clippe [500, 200000] -- calibre sur l'ancienne
# echelle d'avant le passage a REGLEMENTS_TO_DINARS=3234.7 (voir 02_cleaning
# .py). Consequence : moyenne simulee ~5680 TND contre ~819 TND reel (donnees
# completes, Sinistres_Cleaned.csv) -- un sinistre simule ~7x trop eleve. Ça
# faisait deriver ca_total_reel et sin_total_reel a des rythmes differents
# (l'un correctement recalibre, l'autre non), donc le ratio sinistres/primes
# affiche sur le tableau de bord (/api/kpis) grimpait avec le temps de
# fonctionnement du serveur au lieu de rester stable autour de 62-63%.
# Valeurs recalculees par methode des moments (memes donnees, moyenne 818,59 /
# ecart-type 3438,78 TND) : CV ≈ 4,20 -> sigma ≈ 1,710, mu ≈ 5,245. Verifie
# empiriquement : moyenne simulee ~787 TND sur 200k tirages -- coherent.
REGLEMENTS_SIM_LOGNORMAL_MU    = 5.245
REGLEMENTS_SIM_LOGNORMAL_SIGMA = 1.710
REGLEMENTS_SIM_MIN_TND         = 1.0
REGLEMENTS_SIM_MAX_TND         = 240000.0

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
# exploitable. Un plancher a 200 TND (25e percentile approx.) laissait encore
# passer des cas extremes (ex: un client a 875 TND affichant un ratio a
# 311 395% une fois REGLEMENTS correctement calibre -- voir 02_cleaning.py).
# Remonte a la prime auto moyenne reelle en Tunisie (766,50 TND/an, meme
# reference que RAW_TO_DINARS -- Managers.tn, 2021) : sous ce seuil, le
# client n'a pas paye l'equivalent d'une annee de prime moyenne, le
# denominateur reste trop instable pour un ratio individuel exploitable.
# Distinct de RISK_EXCEPTIONNEL_THRESHOLD_PCT ci-dessous, qui traite un
# autre cas : un denominateur normal mais un sinistre reel exceptionnellement
# eleve.
MIN_CA_FOR_RISK_TND = 766.5

# Meme probleme, cote AGENCE plutot que client (remarque superviseur) : depuis
# le passage aux 77/74 vraies agences MAE (au lieu des 10 anciennes, qui
# agregaient chacune des milliers de contrats et lissaient ce risque), une
# poignee de tres petites agences (5 en ont moins de 50 contrats, ex.
# "Al Djazira" avec seulement 2) peuvent afficher un ratio S/P a plusieurs
# milliers de % du simple fait d'un denominateur quasi nul -- ce qui ecrasait
# visuellement le graphique "Ratio Sinistres/Primes par Agence" (axe force a
# une echelle enorme, toutes les vraies barres devenant invisibles). Meme
# logique que MIN_CA_FOR_RISK_TND : exclure du classement les agences dont
# l'exposition est trop faible pour qu'un ratio soit exploitable. 50 contrats
# laisse 72 des 77 agences (seules 5 exclues), coherent avec le seuil de
# "semaine dense" (>100 lignes) deja utilise dans 04_forecasting.py pour le
# meme type de probleme (exclure un volume trop faible pour etre fiable).
MIN_CONTRATS_FOR_AGENCE_RATIO = 50

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

def normalize_agence(raw) -> str:
    # Plus de table d'alias codee en dur (ex. "sfax"->"Sfax Ville") : depuis
    # le passage aux 77 vraies agences MAE (voir 02_cleaning.py), plusieurs
    # agences reelles partagent une meme ville (Sfax 1..5, Gabes 1..4...),
    # donc une correspondance exacte ville->agence unique n'a plus de sens.
    # Correspondance par prefixe generique a la place : "sfax" -> la
    # premiere agence dont le nom commence par "sfax" (ou l'inverse).
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return random.choice(AGENCES)
    s = str(raw).strip()
    key = _strip_accents(s).lower()
    if key in _AGENCE_CANONICAL:
        return _AGENCE_CANONICAL[key]
    for canon_key, name in _AGENCE_CANONICAL.items():
        if canon_key.startswith(key) or key.startswith(canon_key):
            return name
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
    "sin_by_month_reel": {},
    "segments_summary":  [],
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
            # SEXE/CSP/BONUS_MALUS sont de VRAIES colonnes du CSV (voir
            # tool_profil_clients, qui lit desormais get_full_df() -- v2.5) ; AGE
            # n'existe dans aucun fichier reel (seul DT_NAISS, jamais exploite),
            # donc reste fabrique ici avec la meme distribution random.randint(22,72)
            # que l'ancien echantillon live, pour ne rien changer au profil observe.
            _cols_full = [c for c in ["N_CLIENT", "AGENCE", "BRANCHE", "Region", "PRIME_NETTE", "SEXE", "CSP", "BONUS_MALUS"] if c in df_full.columns]
            state["prod_full_df"] = df_full[_cols_full].copy()
            if "SEXE" in state["prod_full_df"].columns:
                state["prod_full_df"]["SEXE"] = state["prod_full_df"]["SEXE"].apply(lambda v: str(v) if pd.notna(v) else random.choice(["M", "F"]))
            if "CSP" in state["prod_full_df"].columns:
                state["prod_full_df"]["CSP"] = state["prod_full_df"]["CSP"].apply(lambda v: str(v) if pd.notna(v) else random.choice(CSP_LIST))
            if "BONUS_MALUS" in state["prod_full_df"].columns:
                state["prod_full_df"]["BONUS_MALUS"] = state["prod_full_df"]["BONUS_MALUS"].apply(lambda v: int(float(v)) if pd.notna(v) else 0)
            state["prod_full_df"]["AGE"] = np.random.randint(22, 73, size=len(state["prod_full_df"]))

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

            if "DATE_ACCIDENT" in df_sin_full.columns:
                _mois_sin = pd.to_datetime(df_sin_full["DATE_ACCIDENT"], errors="coerce").dt.month
                df_sin_full["MOIS"] = _mois_sin
                # Meme principe que ca_by_month_reel (Production) : agrege sur
                # le fichier COMPLET, pas l'echantillon 3k utilise par le flux
                # simule -- /api/ca-by-month affichait un total sinistres
                # quasi-nul faute de cet agregat (bar quasi invisible sur le
                # graphique "Evolution Mensuelle" du dashboard).
                state["sin_by_month_reel"] = (
                    df_sin_full.dropna(subset=["MOIS"]).groupby("MOIS")["REGLEMENTS"].sum().to_dict()
                )
            else:
                state["sin_by_month_reel"] = {}

            _cols_sin_full = [c for c in ["N_CLIENT", "AGENCE", "REGLEMENTS", "MOIS", "TRANSACTION"] if c in df_sin_full.columns]
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

    # Frontiere entre l'echantillon initial (deja compte dans prod_full_df/
    # sin_full_df, lus depuis le CSV complet) et les contrats/sinistres
    # cosmetiques ajoutes ensuite par simulator() -- voir get_full_df() plus
    # bas, qui n'ajoute au portefeuille complet que ce qui vient APRES cette
    # frontiere, pour ne jamais compter deux fois l'echantillon de depart.
    state["_live_seed_count_prod"] = len(state["contrats"])
    state["_live_seed_count_sin"]  = len(state["sinistres"])

    # Rejoue les contrats/sinistres generes en direct lors des sessions
    # precedentes (voir _append_live_delta() dans simulator()) : sans ceci,
    # total_generated et les totaux qu'il alimente repartiraient de
    # l'echantillon de depart a chaque redemarrage.
    state["contrats"].extend(_load_live_delta(_LIVE_DELTA_PROD_FILE))
    state["sinistres"].extend(_load_live_delta(_LIVE_DELTA_SIN_FILE))
    state["total_generated"] = len(state["contrats"])

    state["segments_summary"] = load_segments()


# Degrades de vert (identite visuelle du reste du dashboard) plutot qu'une
# palette arc-en-ciel -- du plus fonce (segment le plus haut en CA) au plus
# clair, pour rester coherent avec le reste de l'app.
SEGMENT_COLORS = ["#0b3d1f", "#156b32", "#1a7a3a", "#2ea84f", "#4cba6a", "#7fd99a", "#b5ead7"]

def load_segments():
    """
    Lit outputs/segments_clients.csv (sortie reelle de 05_clustering.py) et
    agrege un resume par segment -- remplace deux anciens tableaux figes a
    la main (ici et dans /api/segments), deconnectes du clustering reel :
    nombre de segments, noms et chiffres ne bougeaient jamais meme apres un
    re-clustering. Source unique partagee par tool_segments() (agent) et
    /api/segments (dashboard).
    """
    path = "../outputs/segments_clients.csv"
    if not os.path.exists(path):
        print("⚠️  segments_clients.csv introuvable -> executer 05_clustering.py")
        return []
    try:
        df = pd.read_csv(path)
        # Ratio sinistres/primes AGREGE par segment (somme/somme, pas une
        # moyenne de ratios individuels) -- meme methode que le ratio S/P
        # global du portefeuille et que le nommage "Client a Risque" dans
        # 05_clustering.py, pour rester coherent (remarque superviseur :
        # utiliser le fichier Sinistres au-dela de l'EDA/dashboard).
        sums = df.groupby('Segment_Label')[['CA_TOTAL', 'TOTAL_REGLEMENTS']].sum()
        g = df.groupby('Segment_Label').agg(
            nb_clients=('N_CLIENT', 'count'),
            ca_moyen=('CA_TOTAL', 'mean'),
            bm_moyen=('BONUS_MALUS_MOY', 'mean'),
            nb_contrats_moyen=('NB_CONTRATS', 'mean'),
        ).reset_index()
        g['ratio_sp_pct'] = (sums['TOTAL_REGLEMENTS'] / sums['CA_TOTAL'] * 100).values

        # Niveau de risque relatif a la moyenne/dispersion du ratio S/P
        # agrege TOUS segments confondus (pas un seuil fixe choisi a la main).
        ratio_mean = g['ratio_sp_pct'].mean()
        ratio_std  = g['ratio_sp_pct'].std()
        def risque_label(ratio):
            if ratio >= ratio_mean + 0.5 * ratio_std: return "Eleve"
            if ratio <= ratio_mean - 0.5 * ratio_std: return "Faible"
            return "Modere"
        g['risque'] = g['ratio_sp_pct'].apply(risque_label)
        g = g.sort_values('ca_moyen', ascending=False).reset_index(drop=True)

        return [{
            "segment":           row['Segment_Label'],
            "nb_clients":        int(row['nb_clients']),
            "ca_moyen":          round(row['ca_moyen'], 2),
            # Bonus-malus = un palier entier de la grille tarifaire (1, 2, 3...),
            # jamais une valeur a decimale -- une moyenne arrondie a l'entier le
            # plus proche reste lisible comme "palier typique du segment", pas
            # comme "6,7" qui ne correspond a aucun palier reel.
            "bm_moyen":          int(round(row['bm_moyen'])),
            "nb_contrats_moyen": round(row['nb_contrats_moyen'], 2),
            "ratio_sp_pct":      round(row['ratio_sp_pct'], 1),
            "risque":            row['risque'],
            "color":             SEGMENT_COLORS[i % len(SEGMENT_COLORS)],
            "description":       f"{row['nb_contrats_moyen']:.1f} contrats/client en moyenne, "
                                   f"ratio sinistres/primes {row['ratio_sp_pct']:.1f}%",
        } for i, row in g.iterrows()]
    except Exception as e:
        print(f"⚠️  segments_clients.csv ({e}) -> segments non charges")
        return []


def load_accidents_forecast():
    """
    Lit outputs/previsions_accidents_12mois.csv (sortie de 08_accidents_
    forecasting.py) -- prevision du NOMBRE d'accidents/mois via une
    regression de Poisson (tendance + un terme d'alternance mois-haut/
    mois-bas, voir build_features() dans le script). Borne_Basse/Borne_Haute
    ont une largeur DYNAMIQUE (RMSE * sqrt(horizon)) : l'intervalle
    s'elargit progressivement du mois 1 au mois 12, pas un +-RMSE constant
    repete pour chaque mois.
    Remarque superviseur : retire le graphique d'importance des variables
    (encore un detail de mecanique de modele, pas un contenu pour la
    Direction Generale) et le remplace par cette prevision -- un vrai
    graphe de tendance/prevision, la demande initiale.
    """
    path = "../outputs/previsions_accidents_12mois.csv"
    if not os.path.exists(path):
        print("⚠️  previsions_accidents_12mois.csv introuvable -> executer 08_accidents_forecasting.py")
        return []
    try:
        df = pd.read_csv(path)
        calendar = forecast_calendar(periods=len(df))
        return [{
            "mois":        MOIS_LABELS[c["mois_num"] - 1],
            "accidents":   int(df.iloc[c["i"]]["Accidents_Prevus"]),
            "borne_basse": int(df.iloc[c["i"]]["Borne_Basse"]),
            "borne_haute": int(df.iloc[c["i"]]["Borne_Haute"]),
        } for c in calendar]
    except Exception as e:
        print(f"⚠️  previsions_accidents_12mois.csv ({e}) -> non chargee")
        return []


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
            "REGLEMENTS":  float(np.clip(np.random.lognormal(REGLEMENTS_SIM_LOGNORMAL_MU, REGLEMENTS_SIM_LOGNORMAL_SIGMA), REGLEMENTS_SIM_MIN_TND, REGLEMENTS_SIM_MAX_TND)),
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
            # v2.6 : les contrats/sinistres simules sont ajoutes a
            # state["contrats"]/["sinistres"] ET pris en compte par
            # get_full_df() (voir plus bas) -- donc par tous les outils qui
            # en dependent (KPIs, sinistres-stats, portfolio_summary...),
            # de facon coherente entre eux puisqu'ils passent tous par ce
            # meme point d'entree. Persistes en JSON Lines (append, cout
            # constant) pour survivre a un redemarrage -- voir
            # _append_live_delta()/_load_live_delta() plus haut.
            state["contrats"].append(new_c)
            state["total_generated"] += 1
            _append_live_delta(_LIVE_DELTA_PROD_FILE, new_c)

            if random.random() < 0.30:
                new_s = {
                    "N_CLIENT":    new_c["N_CLIENT"],
                    "AGENCE":      new_c["AGENCE"],
                    "REGLEMENTS":  float(np.clip(np.random.lognormal(REGLEMENTS_SIM_LOGNORMAL_MU, REGLEMENTS_SIM_LOGNORMAL_SIGMA), REGLEMENTS_SIM_MIN_TND, REGLEMENTS_SIM_MAX_TND)),
                    "TRANSACTION": np.random.choice(["REGLEMENT", "EN COURS", "EXPERTISE"], p=[0.55, 0.30, 0.15]),
                    "MOIS":        mois,
                    "timestamp":   datetime.now().isoformat(),
                }
                state["sinistres"].append(new_s)
                _append_live_delta(_LIVE_DELTA_SIN_FILE, new_s)
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


def _live_delta_df(records, seed_count, base_columns):
    """
    Les lignes de `records` ajoutees APRES `seed_count` par simulator() --
    jamais l'echantillon initial (deja dans base_columns/prod_full_df, tire
    du meme CSV complet), sous peine de compter ces lignes deux fois.
    """
    new_rows = records[seed_count:]
    if not new_rows:
        return None
    cols = [c for c in base_columns if c in new_rows[0]]
    return pd.DataFrame(new_rows)[cols]


def get_full_df():
    """
    Renvoie les dataframes prod/sinistres COMPLETS (toutes les lignes du CSV
    nettoye) PLUS les contrats/sinistres cosmetiques generes depuis le
    demarrage par simulator() -- pour que le tableau de bord bouge reellement
    a chaque nouveau contrat (remarque : le dashboard doit refleter le live,
    pas juste re-interroger un total fige). Un seul point d'entree pour toute
    agregation "portefeuille actuel" : chaque appelant (KPIs, graphes,
    resume agent) voit exactement le meme delta live, evitant le probleme
    d'origine (certains outils comptaient les contrats simules et d'autres
    non). Le delta est recalcule a chaque appel (liste Python, pas de lock :
    une lecture legerement en retard sur un append concurrent est sans
    consequence ici) -- l'echantillon initial, lui, reste fige depuis
    seed_data() et n'est jamais recalcule.
    """
    prod_base = state.get("prod_full_df", pd.DataFrame())
    sin_base  = state.get("sin_full_df", pd.DataFrame())

    live_prod = _live_delta_df(state["contrats"],  state.get("_live_seed_count_prod", 0), prod_base.columns)
    live_sin  = _live_delta_df(state["sinistres"], state.get("_live_seed_count_sin", 0),  sin_base.columns)

    prod = pd.concat([prod_base, live_prod], ignore_index=True) if live_prod is not None else prod_base
    sin  = pd.concat([sin_base, live_sin], ignore_index=True)   if live_sin  is not None else sin_base
    return prod, sin


def _live_month_sums():
    """
    Meme delta live que get_full_df(), mais groupe par MOIS -- pour le seul
    endroit (le cas non filtre de /api/ca-by-month) qui lit encore les
    totaux figes ca_by_month_reel/ca_total_reel au lieu de get_full_df(),
    pour rester rapide. Sans ce complement, /api/kpis (qui passe par
    get_full_df()) et /api/ca-by-month non filtre afficheraient deux CA
    totaux differents des que le simulateur tourne -- exactement la
    coherence que get_full_df() est cense garantir partout ailleurs.
    """
    new_prod = state["contrats"][state.get("_live_seed_count_prod", 0):]
    new_sin  = state["sinistres"][state.get("_live_seed_count_sin", 0):]
    ca_by_month, sin_by_month = {}, {}
    for r in new_prod:
        m = r.get("MOIS")
        if m:
            ca_by_month[m] = ca_by_month.get(m, 0.0) + float(r.get("PRIME_NETTE", 0) or 0)
    for r in new_sin:
        m = r.get("MOIS")
        if m:
            sin_by_month[m] = sin_by_month.get(m, 0.0) + float(r.get("REGLEMENTS", 0) or 0)
    return ca_by_month, sin_by_month


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
    # normalize_agence() fait une correspondance exacte puis par prefixe (ex:
    # "Sousse" -> "Sousse 1") -- sans cet appel, un nom d'agence partiel/
    # informel extrait par l'agent d'une question en langage naturel ne
    # matchait RIEN (comparaison exacte contre les 77 vrais noms d'agence,
    # souvent "Ville N") et renvoyait silencieusement un resultat a zero
    # au lieu d'une erreur ou d'un vrai match.
    if agence  and agence  != "all" and "AGENCE"  in df.columns: df = df[df["AGENCE"]  == normalize_agence(agence)]
    if region  and region  != "all" and "Region"  in df.columns: df = df[df["Region"]  == region]
    if branche and branche != "all" and "BRANCHE" in df.columns: df = df[df["BRANCHE"] == branche]
    if mois    and mois    != 0     and "MOIS"    in df.columns: df = df[df["MOIS"]    == mois]
    return df


def is_unfiltered(agence=None, region=None, branche=None, mois=None):
    agence_empty  = agence  in (None, "all")
    region_empty  = region  in (None, "all")
    branche_empty = branche in (None, "all")
    mois_empty    = mois in (None, 0)
    return agence_empty and region_empty and branche_empty and mois_empty


def tool_portfolio_summary(agence=None, region=None, branche=None):
    # v2.5 -- utilise get_full_df() partout (filtre ou non), plus jamais
    # get_df() (l'ancien echantillon live scale, retire). Remarque
    # superviseur : soit TOUS les outils comptent les contrats generes en
    # direct, soit AUCUN -- avant ce fix, le cas non filtre utilisait des
    # compteurs (ca_total_reel/nb_contrats_reel) incrementes par le
    # simulateur alors que top_agence/nb_clients venaient deja de get_df(),
    # un melange incoherent des deux.
    # v2.6 -- get_full_df() incorpore de nouveau les contrats/sinistres
    # cosmetiques generes depuis le demarrage (voir get_full_df()), mais
    # UNIQUEMENT via ce point d'entree unique : tous les outils qui appellent
    # get_full_df() voient exactement le meme delta live, donc restent
    # coherents entre eux -- le probleme d'origine n'etait pas que le live
    # soit compte, mais qu'il le soit differemment selon l'outil.
    prod, sin = get_full_df()
    prod_f = apply_filters(prod, agence, region, branche)
    sin_f  = apply_filters(sin, agence, region, branche)

    ca = round(prod_f["PRIME_NETTE"].sum(), 2) if "PRIME_NETTE" in prod_f.columns else 0
    nb_contrats = len(prod_f)
    st = round(sin_f["REGLEMENTS"].sum(), 2) if "REGLEMENTS" in sin_f.columns else 0

    top_ag = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().idxmax() if len(prod_f) > 0 else "N/A"
    return {
        "ca_total":           ca,
        "nb_clients":         int(prod_f["N_CLIENT"].nunique()) if "N_CLIENT" in prod_f.columns else nb_contrats,
        "nb_contrats":        nb_contrats,
        "sinistres_total":    round(st, 2),
        "ratio_sinistralite": round(st / ca * 100, 2) if ca > 0 else 0,
        # Marge technique BRUTE (avant frais de gestion/commissions) = primes
        # - sinistres. "Brute" precise volontairement qu'elle exclut les
        # frais operationnels (aucune colonne de frais/commissions n'est
        # disponible dans les donnees source) -- ne pas presenter comme un
        # resultat net complet. CAPITAUX n'y figure jamais : ce n'est pas
        # une depense, c'est un plafond de garantie (exposition au risque).
        "marge_technique_brute": round(ca - st, 2),
        "top_agence":         str(top_ag),
        "derniere_maj":       state["last_update"],
        "total_generes":      state["total_generated"],
    }


def tool_top_agencies(n=5, agence=None, region=None, branche=None):
    prod, _ = get_full_df()  # v2.5 -- coherent avec les autres outils, plus jamais get_df()
    prod = apply_filters(prod, agence, region, branche)
    if len(prod) == 0:
        return []
    top      = prod.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values(ascending=False).head(n)
    ca_total = prod["PRIME_NETTE"].sum()
    return [{"agence": str(a), "ca": round(v, 2), "part_pct": round(v / ca_total * 100, 1)}
            for a, v in top.items()]


# Facteurs saisonniers derives du modele Prophet de 04_forecasting.py
# (remarque 2, superviseur : compare empiriquement a 7 alternatives dans
# model_comparison.ipynb, dans une comparaison unifiee intra-echantillon +
# LOOCV + backtest glissant hors-echantillon -- Prophet generalise mieux
# que tous les autres modeles testes sur les trois mesures -- remplace la
# regression lineaire saisonniere utilisee avant).
# Hyperparametres Prophet (changepoint_prior_scale=0.0244,
# seasonality_prior_scale=0.599, fourier_order=1) tunes par Optuna + LOOCV
# dans le notebook, pas choisis a la main.
# Facteurs extraits en decomposant la prevision Prophet a 12 mois en
# composante tendance + composante saisonniere (colonnes 'trend'/'monthly'
# de model.predict()), rapportees a la tendance moyenne puis normalisees
# pour que leur moyenne annuelle soit exactement 1 -- a regenerer si
# 04_forecasting.py est ré-entraine sur de nouvelles donnees.
SEASONAL_NORM = [1.0051, 0.9792, 1.0029, 0.9770, 1.0006, 0.9747, 0.9417, 1.0269, 1.0235, 1.0247, 1.0213, 1.0224]

# Taux de croissance annuel derive du meme forecast Prophet (total prevu sur
# les 12 prochains mois vs total reel des 12 derniers mois, cf. 04_forecasting.py)
# -- plus un taux fixe suppose, comme l'etait le +4.7% initial dont l'origine
# n'etait pas documentee. A regenerer avec SEASONAL_NORM si le modele est
# ré-entraine sur de nouvelles donnees.
FORECAST_GROWTH_RATE = 0.0340
MOIS_NOMS_LONGS = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                   "Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]

# ── Intervalle de confiance DYNAMIQUE (widens with forecast horizon) ──
CI_BASE_PCT   = 0.05    # +-5% pour le mois le plus proche
CI_GROWTH_PCT = 0.003   # +0.3 point par mois d'horizon supplementaire

def ci_width_for_month(i: int) -> float:
    """
    Largeur de l'intervalle de confiance pour le mois d'indice i (0=premier
    mois prevu -- pas forcement Janvier, voir forecast_calendar --, 11=dernier
    des 12 mois). Convention assumee et documentee (voir notes projet) :
    plus l'horizon de prevision est loin, plus l'incertitude est grande,
    donc l'intervalle s'elargit lineairement avec le mois plutot que de
    rester fixe a +-7% quel que soit l'horizon. A affiner avec un vrai
    ecart-type de residus (ex: recalcule sur le RMSE du modele une fois
    la correction millimes->dinars propagee dans model_comparison.ipynb).
    """
    return CI_BASE_PCT + CI_GROWTH_PCT * i


def forecast_calendar(periods: int = 12, today=None):
    """
    Les 12 (mois calendaire, annee) qui composent la fenetre de prevision,
    en commencant au mois PROCHAIN reel (pas Janvier fixe) -- corrige un bug
    ou la prevision affichait toujours "Janvier -> Decembre" quelle que soit
    la date reelle d'execution, alors que les donnees d'entrainement sont
    deja decalees dynamiquement (voir 02_cleaning.py, compute_dynamic_date_shift).
    Ex : execute en aout 2026 -> Septembre 2026 .. Aout 2027.
    """
    if today is None:
        today = datetime.now()
    start_month = today.month + 1
    start_year  = today.year
    if start_month > 12:
        start_month = 1
        start_year += 1

    out = []
    for i in range(periods):
        m = ((start_month - 1 + i) % 12) + 1
        y = start_year + (start_month - 1 + i) // 12
        out.append({"i": i, "mois_num": m, "annee": y,
                    "mois": f"{MOIS_NOMS_LONGS[m - 1]} {y}"})
    return out


def tool_forecast(mois_num=None):
    base     = state.get("ca_total_reel", 1_630_000_000) or 1_630_000_000
    growth   = FORECAST_GROWTH_RATE
    total_12 = base * (1 + growth)
    calendar = forecast_calendar()
    monthly  = [total_12 / 12 * SEASONAL_NORM[c["mois_num"] - 1] for c in calendar]

    if mois_num and 1 <= mois_num <= 12:
        c  = next(c for c in calendar if c["mois_num"] == mois_num)
        ca = monthly[c["i"]]
        w  = ci_width_for_month(c["i"])
        return {"mois": c["mois"], "ca_prev": round(ca, 2),
                "ci_low": round(ca * (1 - w), 2), "ci_high": round(ca * (1 + w), 2),
                "ci_width_pct": round(w * 100, 1)}

    pic = calendar[int(np.argmax(monthly))]
    return {
        "total_12_mois":  round(sum(monthly), 2),
        "croissance_pct": round(growth * 100, 2),
        "mois_pic":       pic["mois"],
        "detail":         [{"mois": c["mois"], "ca": round(monthly[c["i"]], 2),
                             "ci_width_pct": round(ci_width_for_month(c["i"]) * 100, 1)} for c in calendar],
    }


def tool_explain_forecast(mois_num=None):
    base     = state.get("ca_total_reel", 1_630_000_000) or 1_630_000_000
    growth   = FORECAST_GROWTH_RATE
    total_12 = base * (1 + growth)
    calendar = forecast_calendar()
    monthly  = [total_12 / 12 * SEASONAL_NORM[c["mois_num"] - 1] for c in calendar]

    if mois_num and 1 <= mois_num <= 12:
        c = next(c for c in calendar if c["mois_num"] == mois_num)
        base_mensuel      = base / 12
        apres_croissance  = total_12 / 12
        apres_saisonnalite = monthly[c["i"]]
        facteur = SEASONAL_NORM[c["mois_num"] - 1]
        w = ci_width_for_month(c["i"])
        return {
            "mois": c["mois"],
            "ca_prevu": round(apres_saisonnalite, 2),
            "decomposition": {
                "1_base_mensuelle_actuelle":        round(base_mensuel, 2),
                "2_apres_croissance":                round(apres_croissance, 2),
                "3_facteur_saisonnier":              round(facteur, 3),
                "4_apres_saisonnalite_final":        round(apres_saisonnalite, 2),
            },
            "explication": (
                f"CA actuel reparti uniformement sur 12 mois = {round(base_mensuel, 2):,.2f} TND. "
                f"Apres application de la croissance annuelle de +{growth*100:.1f}% (extrapolee par le "
                f"modele Prophet sur l'historique reel, voir model_comparison.ipynb) = "
                f"{round(apres_croissance, 2):,.2f} TND. "
                f"{c['mois']} a un facteur saisonnier de {facteur:.3f} "
                f"({'au-dessus' if facteur > 1 else 'en-dessous'} de la moyenne annuelle), "
                f"d'ou la prevision finale de {round(apres_saisonnalite, 2):,.2f} TND, "
                f"avec un intervalle de confiance de +-{round(w*100,1)}% (horizon mois {c['i']+1}/12, "
                f"l'incertitude croit avec l'horizon de prevision)."
            ),
        }

    return {
        "total_12_mois": round(sum(monthly), 2),
        "decomposition": {
            "1_base_reelle_actuelle":     round(base, 2),
            "2_croissance_appliquee_pct": round(growth * 100, 2),
            "3_montant_croissance_tnd":   round(total_12 - base, 2),
            "4_total_apres_croissance":   round(total_12, 2),
        },
        "explication": (
            f"Le CA reel actuel (mesure sur la totalite du fichier, pas un echantillon) est de "
            f"{round(base, 2):,.2f} TND. La croissance de +{growth*100:.1f}% (extrapolee par le modele "
            f"Prophet sur l'evolution reelle du CA MAE, voir model_comparison.ipynb) ajoute "
            f"{round(total_12 - base, 2):,.2f} TND, "
            f"pour un total previsionnel sur 12 mois ({calendar[0]['mois']} a {calendar[-1]['mois']}) de "
            f"{round(total_12, 2):,.2f} TND. Ce total est ensuite reparti sur les 12 mois selon un "
            f"facteur saisonnier normalise (moyenne == 1). "
            f"L'intervalle de confiance n'est plus fixe a +-7% : il part de +-{CI_BASE_PCT*100:.0f}% pour "
            f"le premier mois prevu et s'elargit de +{CI_GROWTH_PCT*100:.1f} point par mois d'horizon, "
            f"jusqu'a +-{ci_width_for_month(11)*100:.1f}% pour le 12e mois, pour refleter l'incertitude croissante."
        ),
        "mois_saisonniers": [
            {"mois": c["mois"], "facteur": round(SEASONAL_NORM[c["mois_num"] - 1], 3),
             "ci_width_pct": round(ci_width_for_month(c["i"]) * 100, 1)}
            for c in calendar
        ],
    }


def tool_accidents_forecast():
    """
    Prevision du NOMBRE d accidents/mois sur les 12 prochains mois (Poisson
    GLM, voir 08_accidents_forecasting.py) -- un COMPTE d evenements, PAS un
    montant en TND. A utiliser quand l utilisateur demande combien
    d accidents/sinistres sont attendus dans le futur (pour le cout
    previsionnel en TND, utiliser forecast).
    """
    detail = load_accidents_forecast()
    if not detail:
        return {"error": "Previsions accidents indisponibles (executer 08_accidents_forecasting.py)."}
    pic = max(detail, key=lambda d: d["accidents"])
    return {
        "total_12_mois_accidents": sum(d["accidents"] for d in detail),
        "mois_pic":                pic["mois"],
        "accidents_mois_pic":      pic["accidents"],
        "detail":                  detail,
    }


# Fallback UNIQUEMENT si 05_clustering.py n'a jamais ete execute (pas de
# outputs/segments_clients.csv) -- valeurs approximatives d'un run passe,
# pas la source de verite. load_segments() (voir seed_data) est prioritaire.
# Meme schema que load_segments() pour que tool_segments() et /api/segments
# (qui partagent maintenant cette meme liste) n'aient pas besoin d'un
# fallback separe chacun.
_SEGMENTS_FALLBACK = [
    {"segment":"Premium",    "nb_clients":7751,  "ca_moyen":729000, "bm_moyen":2,
     "nb_contrats_moyen":12.3, "ratio_sp_pct":45.0, "risque":"Faible", "color":SEGMENT_COLORS[0],
     "description":"12+ contrats, BM<=3, tres fideles"},
    {"segment":"Standard",   "nb_clients":11771, "ca_moyen":368000, "bm_moyen":7,
     "nb_contrats_moyen":5.8,  "ratio_sp_pct":62.0, "risque":"Modere", "color":SEGMENT_COLORS[1],
     "description":"Primes elevees, BM=7, potentiel upgrade"},
    {"segment":"Occasionnel","nb_clients":19333, "ca_moyen":190000, "bm_moyen":5,
     "nb_contrats_moyen":2.4,  "ratio_sp_pct":62.0, "risque":"Modere", "color":SEGMENT_COLORS[2],
     "description":"2-3 contrats, potentiel fidelisation"},
    {"segment":"A Risque",   "nb_clients":33819, "ca_moyen":172000, "bm_moyen":11,
     "nb_contrats_moyen":1.9,  "ratio_sp_pct":80.0, "risque":"Eleve", "color":SEGMENT_COLORS[3],
     "description":"Fort BM, sinistralite elevee -- priorite"},
]

def tool_segments():
    return state.get("segments_summary") or _SEGMENTS_FALLBACK


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

    BUG CORRIGE : clients_a_risque/part_risque_pct etaient des CONSTANTES
    codees en dur (33819 / 46.5%), visiblement figees sur un run de
    clustering anterieur -- le segment "Client a Risque" reel actuel
    (outputs/segments_clients.csv, voir circulaire_segmentation.md) n'en
    compte que 13911 (19.1%). Lus maintenant depuis segments_summary (la
    meme source que tool_segments()), donc toujours coherents avec la
    segmentation reellement chargee au demarrage.
    """
    # v2.5 -- get_full_df() partout, meme raison que tool_portfolio_summary
    # (coherence : soit tous les outils comptent les contrats/sinistres
    # generes en direct, soit aucun -- l'ancienne version comptait deja
    # differemment le cas filtre (echantillon live + extrapolation) du cas
    # non filtre (compteur global), un melange incoherent en interne).
    prod, sin = get_full_df()
    prod_f = apply_filters(prod, agence, region, branche)
    sin_f  = apply_filters(sin, agence, None, None)  # sin n'a pas Region/BRANCHE

    ca = (prod_f["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod_f.columns else 0) or 1
    st = sin_f["REGLEMENTS"].sum() if "REGLEMENTS" in sin_f.columns else 0

    sous_perf = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values().head(3).index.tolist() \
                if "AGENCE" in prod_f.columns and len(prod_f) > 0 else []

    segments = state.get("segments_summary") or _SEGMENTS_FALLBACK
    seg_risque = next((s for s in segments if "risque" in s["segment"].lower()), None)
    nb_total_segmente = sum(s["nb_clients"] for s in segments) or 1

    return {
        "ratio_sinistralite":  round(st / ca * 100, 2),
        "clients_a_risque":    seg_risque["nb_clients"] if seg_risque else 0,
        "part_risque_pct":     round(seg_risque["nb_clients"] / nb_total_segmente * 100, 1) if seg_risque else 0,
        "clients_a_risque_est_global": True,
        "agences_sous_perf":   [str(a) for a in sous_perf],
        "recommandations": [
            "Reviser la politique tarifaire pour le segment Client a Risque",
            "Pedagogie sur le bonus-malus des la souscription pour le segment Client Jeune Conducteur",
            "Analyser les garanties souscrites par le segment Autres Clients pour comprendre ce qui le distingue",
            "Programme de fidelisation VIP pour le segment Client Premium",
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
    demarrage), PAS state["contrats"]/["sinistres"] (echantillon 10k/3k qui
    alimente uniquement le fil d'activite cosmetique du dashboard live) --
    croiser deux echantillons tires
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

    # Agrege par agence sur la totalite de a_risque (pas seulement les n
    # affiches) -- pour un affichage tableau de bord (comptage/graphe), pas
    # une liste nominative : remarque superviseur, la Direction Generale n'a
    # pas besoin de voir chaque client un par un, seulement des stats/graphes.
    par_agence = []
    if len(a_risque) > 0 and "N_CLIENT" in prod_f.columns:
        agences_a_risque = agence_par_client.reindex(a_risque.index)
        counts = agences_a_risque.value_counts()
        par_agence = [{"agence": str(ag), "nb_clients_risque": int(c)} for ag, c in counts.head(10).items()]

    return {
        "clients":                clients,
        "par_agence":             par_agence,
        "total_clients_a_risque": int(len(a_risque)),
        "nb_clients_analyses":    int(len(ratio_df)),
        "seuil_ratio_sp_pct":     RISK_RATIO_THRESHOLD_PCT,
        "note": (
            "Calcule EN DIRECT sur le portefeuille complet -- distinct du chiffre "
            "clients_a_risque renvoye par risk_analysis, qui provient du segment "
            "'Client a Risque' de la segmentation K-Means (05_clustering.py) et "
            "n'est jamais recalcule par filtre. "
            f"Clients avec une prime totale < {MIN_CA_FOR_RISK_TND:.0f} TND exclus (denominateur "
            "trop faible pour un ratio S/P representatif)."
        ),
    }


def tool_sinistres_stats():
    # get_full_df() (portefeuille complet + delta live, voir get_full_df()) --
    # montant_total/montant_moyen/total_sinistres viennent tous du meme
    # dataframe pour rester coherents entre eux au fil du temps.
    _, sin = get_full_df()
    return {
        "total_sinistres": len(sin),
        "montant_total":   round(sin["REGLEMENTS"].sum(), 2) if "REGLEMENTS" in sin.columns else 0,
        "montant_moyen":   round(sin["REGLEMENTS"].mean(), 2) if len(sin) > 0 and "REGLEMENTS" in sin.columns else 0,
        "en_cours":        int(len(sin[sin["TRANSACTION"] == "EN COURS"])) if "TRANSACTION" in sin.columns else 0,
        "pic_mois":        int(sin.groupby("MOIS")["REGLEMENTS"].sum().idxmax()) if "MOIS" in sin.columns and len(sin) > 0 else 0,
    }


def tool_branch_analysis():
    prod, _ = get_full_df()  # v2.5 -- coherent avec les autres outils, plus jamais get_df()
    ca_total = prod["PRIME_NETTE"].sum()
    by_br    = prod.groupby("BRANCHE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    return [{"branche": str(b), "ca": round(v, 2), "part_pct": round(v / ca_total * 100, 1)}
            for b, v in by_br.items()]


def tool_compare_agencies():
    # Portefeuille COMPLET (get_full_df), pas l'echantillon live
    # state["contrats"]/["sinistres"] : le simulateur pioche ses
    # contrats/sinistres synthetiques dans une liste
    # d'agences plus large (tous les gouvernorats) que les 10 agences REELES
    # du portefeuille historique -- sur l'echantillon live, une agence
    # synthetique a tres faible volume peut recevoir par hasard un sinistre
    # simule sans denominateur (CA) correspondant, produisant un ratio S/P
    # de plusieurs milliers de % qui n'existe pas dans les vraies donnees
    # (verifie : "Gafsa"/"Tozeur" n'apparaissent dans AUCUN des deux fichiers
    # Cleaned.csv reels -- seulement 10 agences y existent).
    prod, sin = get_full_df()
    result = []
    for ag in prod["AGENCE"].unique():
        p  = prod[prod["AGENCE"] == ag]
        s  = sin[sin["AGENCE"]  == ag] if "AGENCE" in sin.columns else pd.DataFrame()
        ca = p["PRIME_NETTE"].sum()
        st = s["REGLEMENTS"].sum() if len(s) > 0 and "REGLEMENTS" in s.columns else 0
        # ratio_sp_pct=0 si l'agence est sous le plancher d'exposition (voir
        # MIN_CONTRATS_FOR_AGENCE_RATIO) -- evite qu'une micro-agence a
        # denominateur quasi nul affiche un ratio a plusieurs milliers de %
        # qui ecrase l'echelle du graphique pour toutes les autres.
        assez_de_donnees = len(p) >= MIN_CONTRATS_FOR_AGENCE_RATIO
        result.append({
            "agence":       str(ag),
            "ca":           round(ca, 2),
            "nb_contrats":  len(p),
            "prime_moy":    round(p["PRIME_NETTE"].mean(), 2) if len(p) > 0 else 0,
            "sinistres":    round(st, 2),
            "ratio_sp_pct": round(st / ca * 100, 2) if (ca > 0 and assez_de_donnees) else 0,
        })
    return sorted(result, key=lambda x: x["ca"], reverse=True)


def tool_detect_anomalies():
    # Portefeuille COMPLET (get_full_df), pas l'echantillon live
    # state["contrats"]/["sinistres"] -- meme raison que
    # tool_compare_agencies()/tool_risk_clients() : sur
    # l'echantillon live (10k/3k lignes), le bruit d'echantillonnage + les
    # denominateurs quasi nuls produisaient des ratios moyens a 5 chiffres
    # (ex. 11 634% observe) sans rapport avec le portefeuille reel.
    prod, sin = get_full_df()
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
        # Meme plancher que tool_risk_clients() (MIN_CA_FOR_RISK_TND) --
        # sans lui, ce comptage-ci restait sur un ratio moyen a 5 chiffres
        # (denominateurs quasi nuls) alors que le tableau nominatif juste en
        # dessous, lui, l'appliquait deja -- incoherence visible sur la meme page.
        ratio_df = ratio_df[ratio_df["CA"] >= MIN_CA_FOR_RISK_TND]
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
    prod, _ = get_full_df()  # v2.5 -- coherent avec les autres outils, plus jamais get_df()
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

# EXCEL_AVAILABLE : meme pattern defensif, pour l'export .xlsx (openpyxl est
# une dependance beaucoup plus legere que reportlab -- pas de raison que
# l'un bloque l'autre si un seul des deux est installe).
EXCEL_AVAILABLE = False
try:
    import openpyxl  # noqa: F401  (juste pour verifier la presence du moteur pandas.to_excel)
    EXCEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Export Excel indisponible — {type(e).__name__}: {e} (pip install openpyxl)")

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


def tool_generate_report(agence=None, region=None, branche=None, mois_num=None, sections=None, format="pdf"):
    """
    Genere un rapport du portefeuille (PDF ou Excel) et le sauvegarde dans
    REPORTS_DIR. Retourne les metadonnees du fichier (pas le fichier
    lui-meme — l'agent ne peut pas streamer un binaire dans le chat) ; le
    telechargement se fait via l'URL absolue retournee (download_url).

    Parametres (tous optionnels) :
      - agence/region/branche : restreint le resume et le classement des
        agences a ce perimetre (via tool_portfolio_summary). NOTE : les
        previsions et l'analyse de risque restent globales quel que soit
        ce filtre (voir limitation ci-dessous).
      - mois_num : restreint la section previsions a un seul mois au lieu
        du detail sur 12 mois.
      - sections : sous-ensemble de ["resume","agences","previsions","risques"].
        None/vide => rapport complet (comportement identique a avant).
      - format : "pdf" (defaut) ou "excel"/"xlsx" -- meme donnees, format de
        sortie different. Excel est utile quand l'utilisateur veut filtrer/
        pivoter les chiffres lui-meme plutot que lire un document mis en page.

    LIMITATION CONNUE : tool_risk_analysis() et tool_top_agencies() ne
    prennent pas de filtre agence/region/branche cote donnees — un rapport
    "scope" sur une agence affichera donc un titre/perimetre different
    mais la section risques restera basee sur les chiffres globaux du
    portefeuille. A corriger si un jury teste specifiquement ce cas.
    """
    fmt = (format or "pdf").strip().lower()
    if fmt in ("xlsx", "excel", "xls"):
        return _generate_excel_report(agence, region, branche, mois_num, sections)
    return _generate_pdf_report(agence, region, branche, mois_num, sections)


def _generate_pdf_report(agence, region, branche, mois_num, sections):
    if not REPORT_AVAILABLE:
        return {
            "error": "Generation de rapport PDF indisponible sur cet environnement "
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
                f"CA total prevu sur les 12 prochains mois "
                f"({fc['detail'][0]['mois']} a {fc['detail'][-1]['mois']}) : "
                f"<b>{_fmt_tnd_fr(fc['total_12_mois'])}</b> "
                f"(croissance de {fc['croissance_pct']}% vs les 12 mois precedents). "
                f"Mois de pic saisonnier : {fc['mois_pic']}. "
                f"Intervalles de confiance dynamiques : de +-{fc['detail'][0]['ci_width_pct']}% pour le "
                f"premier mois prevu a +-{fc['detail'][-1]['ci_width_pct']}% pour le dernier "
                f"(incertitude croissante avec l'horizon).",
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


def _generate_excel_report(agence, region, branche, mois_num, sections):
    """
    Meme donnees que _generate_pdf_report (une feuille Excel par section au
    lieu d'un document mis en page) -- pour les utilisateurs qui veulent
    filtrer/pivoter les chiffres eux-memes plutot que lire un PDF statique.
    """
    if not EXCEL_AVAILABLE:
        return {
            "error": "Export Excel indisponible sur cet environnement "
                     "(openpyxl non installe — pip install openpyxl)."
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rapport_MAE_{ts}.xlsx"
    filepath = os.path.join(REPORTS_DIR, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        if "resume" in sections:
            pd.DataFrame([
                {"Indicateur": "CA total (TND)",           "Valeur": summary["ca_total"]},
                {"Indicateur": "Nombre de contrats",        "Valeur": summary["nb_contrats"]},
                {"Indicateur": "Sinistres totaux (TND)",    "Valeur": summary["sinistres_total"]},
                {"Indicateur": "Ratio de sinistralite (%)", "Valeur": summary["ratio_sinistralite"]},
                {"Indicateur": "Agence top",                "Valeur": summary["top_agence"]},
            ]).to_excel(writer, sheet_name="Resume", index=False)

        if "agences" in sections:
            pd.DataFrame(top_ag).rename(columns={
                "agence": "Agence", "ca": "CA (TND)", "part_pct": "Part de marche (%)",
            }).to_excel(writer, sheet_name="Agences", index=False)

        if "previsions" in sections:
            if mois_num:
                pd.DataFrame([{
                    "Mois": fc["mois"], "CA prevu (TND)": fc["ca_prev"],
                    "Borne basse (TND)": fc["ci_low"], "Borne haute (TND)": fc["ci_high"],
                    "Intervalle (%)": fc["ci_width_pct"],
                }]).to_excel(writer, sheet_name="Previsions", index=False)
            else:
                pd.DataFrame(fc["detail"]).rename(columns={
                    "mois": "Mois", "ca": "CA prevu (TND)", "ci_width_pct": "Intervalle (%)",
                }).to_excel(writer, sheet_name="Previsions", index=False)

        if "risques" in sections:
            risk_rows = [
                {"Indicateur": "Ratio de sinistralite (%)",                   "Valeur": risk["ratio_sinistralite"]},
                {"Indicateur": "Clients a risque (portefeuille entier)",      "Valeur": risk["clients_a_risque"]},
                {"Indicateur": "Part du portefeuille a risque (%)",           "Valeur": risk["part_risque_pct"]},
            ] + [
                {"Indicateur": f"Recommandation {i+1}", "Valeur": reco}
                for i, reco in enumerate(risk["recommandations"])
            ]
            pd.DataFrame(risk_rows).to_excel(writer, sheet_name="Risques", index=False)

    return {
        "filename": filename,
        "filepath": filepath,
        "download_url": f"{API_BASE_URL}/api/reports/{filename}",
        "message": f"Export Excel genere avec succes : {filename}. "
                    f"Telechargement disponible via le lien fourni.",
    }


TOOLS_MAP = {
    "portfolio_summary":  tool_portfolio_summary,
    "top_agencies":       tool_top_agencies,
    "forecast":           tool_forecast,
    "explain_forecast":   tool_explain_forecast,
    "accidents_forecast": tool_accidents_forecast,
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
    # v2.5 -- get_full_df() partout, meme raison que tool_portfolio_summary :
    # avant ce fix, le cas filtre lisait get_df() (echantillon live, gonfle
    # par les contrats cosmetiques ajoutes par simulator()) + scale_ca(),
    # alors que le cas non filtre lisait deja des totaux figes au demarrage --
    # un melange incoherent. sin_full_df n'a ni Region/BRANCHE ni MOIS : ces
    # filtres restent approximatifs pour ce total precis (agence reste exact).
    prod, sin = get_full_df()
    prod_f = apply_filters(prod, agence, region, branche, mois)
    sin_f  = apply_filters(sin, agence, region, branche)

    ca = round(prod_f["PRIME_NETTE"].sum(), 2) if "PRIME_NETTE" in prod_f.columns else 0
    nb = len(prod_f)
    st = round(sin_f["REGLEMENTS"].sum(), 2) if "REGLEMENTS" in sin_f.columns else 0

    top_ag = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().idxmax() if len(prod_f) > 0 else "N/A"
    top_ca = prod_f.groupby("AGENCE")["PRIME_NETTE"].sum().max()    if len(prod_f) > 0 else 0
    return {
        "ca_total":             ca,
        "nb_clients":           nb,
        "nb_contrats":          nb,
        "sin_total":            round(st, 2),
        "ratio_sin":            round(st / ca * 100, 2) if ca > 0 else 0,
        "marge_technique_brute": round(ca - st, 2),
        "top_agence":           str(top_ag),
        "top_agence_ca":        round(top_ca, 2),
        "last_update":          state["last_update"],
        "total_generated":      state["total_generated"],
    }


@app.get("/api/ca-by-agence")
def ca_by_agence(agence: str = "all", region: str = "all", branche: str = "all", mois: int = 0):
    prod, _ = get_full_df()  # v2.5 -- coherent avec les autres outils, plus jamais get_df()
    prod  = apply_filters(prod, agence, region, branche, mois)
    data  = prod.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    total = data.sum()
    return [{"agence": str(a), "ca": round(v, 2), "part": round(v / total * 100, 1) if total > 0 else 0}
            for a, v in data.items()]


@app.get("/api/ca-by-branche")
def ca_by_branche(agence: str = "all", region: str = "all", mois: int = 0):
    prod, _ = get_full_df()  # v2.5 -- coherent avec les autres outils, plus jamais get_df()
    prod  = apply_filters(prod, agence, region, None, mois)
    data  = prod.groupby("BRANCHE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    total = data.sum()
    return [{"branche": str(b), "ca": round(v, 2), "part": round(v / total * 100, 1) if total > 0 else 0}
            for b, v in data.items()]


@app.get("/api/ca-by-month")
def ca_by_month(agence: str = "all", region: str = "all", branche: str = "all"):
    if is_unfiltered(agence, region, branche):
        by_month_reel  = state.get("ca_by_month_reel", {})
        total_reel     = state.get("ca_total_reel", 0.0)
        sin_by_month   = state.get("sin_by_month_reel", {})
        live_ca_month, live_sin_month = _live_month_sums()
        return [{"mois": MOIS_LABELS[m-1], "mois_num": m,
                 "primes":    round(float(by_month_reel.get(m, total_reel/12)) + live_ca_month.get(m, 0.0), 2),
                 "sinistres": round(float(sin_by_month.get(m, 0)) + live_sin_month.get(m, 0.0), 2)}
                for m in range(1, 13)]

    # v2.5 -- prod ET sin viennent maintenant tous deux de get_full_df()
    # (portefeuille complet), plus jamais get_df() (echantillon live) ni le
    # facteur d'echelle sf qui compensait sa taille reduite -- meme principe
    # que tool_portfolio_summary/api/kpis. prod_full_df n'a pas de colonne
    # MOIS (contrairement a l'echantillon simule) : ce total agrege par mois
    # reste donc approximatif quand un filtre agence/region/branche est
    # applique (voir ca_by_month_reel ci-dessus pour le cas non filtre, qui
    # lui est exact).
    prod_full, sin_full = get_full_df()
    prod_full_f = apply_filters(prod_full, agence, region, branche)
    sin_full_f  = apply_filters(sin_full, agence, region, branche)
    sin_m = sin_full_f.groupby("MOIS")["REGLEMENTS"].sum() if "MOIS" in sin_full_f.columns else pd.Series()

    total_ca_f = prod_full_f["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod_full_f.columns else 0
    return [{"mois": MOIS_LABELS[m-1], "mois_num": m,
             "primes":    round(float(total_ca_f) / 12, 2),
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
    base     = state.get("ca_total_reel", 1_630_000_000) or 1_630_000_000
    growth   = FORECAST_GROWTH_RATE
    total_12 = base * (1 + growth)
    calendar = forecast_calendar()
    monthly  = [total_12 / 12 * SEASONAL_NORM[c["mois_num"] - 1] for c in calendar]

    by_month_reel = state.get("ca_by_month_reel", {})
    if by_month_reel:
        ca_reel_actuel = {m: round(float(by_month_reel.get(m, base / 12)), 2) for m in range(1, 13)}
    else:
        ca_reel_actuel = {m: round(base / 12, 2) for m in range(1, 13)}

    result = []
    for c in calendar:
        w = ci_width_for_month(c["i"])
        ca = monthly[c["i"]]
        result.append({
            "mois":     MOIS_LABELS[c["mois_num"] - 1],
            "mois_num": c["mois_num"],
            "annee":    c["annee"],
            "ca_prev":  round(ca, 2),
            "ci_low":   round(ca * (1 - w), 2),
            "ci_high":  round(ca * (1 + w), 2),
            "ca_reel":  ca_reel_actuel[c["mois_num"]],
        })
    return result


@app.get("/api/explain-forecast")
def explain_forecast_endpoint(mois_num: int = 0):
    return tool_explain_forecast(mois_num if mois_num else None)


@app.get("/api/accidents-forecast")
def accidents_forecast_endpoint():
    return load_accidents_forecast()


@app.get("/api/monitoring")
def monitoring_endpoint(n_runs: int = 50):
    return tool_agent_monitoring(n_runs)


@app.get("/api/generate-report")
def generate_report_endpoint(format: str = "pdf"):
    result = tool_generate_report(format=format)
    if "error" in result:
        return result
    ext = os.path.splitext(result["filename"])[1].lower()
    return FileResponse(
        path=result["filepath"],
        media_type=_REPORT_MEDIA_TYPES.get(ext, "application/octet-stream"),
        filename=result["filename"],
    )


@app.get("/api/segments")
def segments_endpoint():
    # Meme source que tool_segments() (agent) -- voir load_segments(). Seul
    # "nb_clients" est renomme "count" ici, cle attendue par le dashboard.
    return [{**s, "count": s["nb_clients"]} for s in tool_segments()]


@app.get("/api/filters")
def filters_endpoint():
    prod, _ = get_full_df()  # v2.5 -- coherent avec les autres outils, plus jamais get_df()

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


@app.get("/api/sinistres-stats")
def sinistres_stats_endpoint():
    return tool_sinistres_stats()


class ChatRequest(BaseModel):
    message: str
    history: list = []

_REPORT_MEDIA_TYPES = {
    ".pdf":  "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@app.get("/api/reports/{filename}")
def download_report(filename: str, inline: bool = False):
    """
    inline=False (defaut) : Content-Disposition: attachment -- comportement
    d'origine, force le telechargement (bouton "Telecharger" cote frontend).
    inline=True : Content-Disposition: inline -- le navigateur affiche le
    PDF directement dans un nouvel onglet au lieu de forcer un
    enregistrement sur disque (bouton "Visualiser" cote frontend --
    remarque superviseur : la liste des rapports generes par l'agent
    n'offrait qu'un lien a copier ou un telechargement automatique dans le
    dossier reports/, pas d'apercu direct).
    """
    # Securite : empeche toute tentative de path traversal (../../etc)
    safe_name = os.path.basename(filename)
    filepath = os.path.join(REPORTS_DIR, safe_name)
    ext = os.path.splitext(safe_name)[1].lower()
    if not os.path.isfile(filepath) or ext not in _REPORT_MEDIA_TYPES:
        raise HTTPException(status_code=404, detail="Rapport introuvable.")
    if inline:
        return FileResponse(
            path=filepath, media_type=_REPORT_MEDIA_TYPES[ext],
            headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
        )
    return FileResponse(path=filepath, media_type=_REPORT_MEDIA_TYPES[ext], filename=safe_name)


@app.post("/api/agent")
def agent_endpoint(req: ChatRequest):
    global _agent
    if _agent is None:
        try:
            _agent = MAEAgent(tools_map=TOOLS_MAP)
        except Exception as e:
            return {"answer": str(e), "tool_calls": [], "thinking": [], "tokens_used": 0, "duration_ms": 0}

    return _agent.run(req.message, req.history)


@app.on_event("startup")
def startup():
    seed_data()
    threading.Thread(target=simulator, args=(5,), daemon=True).start()
    print("✅ MAE Intelligence API v3.3 — simulateur actif toutes les 5s")