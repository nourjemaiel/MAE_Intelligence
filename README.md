# MAE Intelligence

Plateforme d'aide à la décision pour MAE Assurances (Mutuelle Automobile des
Enseignants de Tunisie) : pipeline de données → API temps réel + agent
conversationnel → tableau de bord. Construite sur le portefeuille réel MAE
2025 (342 883 contrats).

## Architecture

```
Pipeline de données  →  API + Agent (mae_backend/)  →  Dashboard (mae_frontend/)
   01 → 05                FastAPI + MAEIA (Groq)          React (fichier unique)
```

- **Pipeline** (`01_diagnostic.py` → `05_clustering.py`) : nettoie les CSV
  bruts, prévoit le CA 2026, segmente les clients. Voir le détail de chaque
  étape plus bas.
- **Backend** (`mae_backend/`) : API FastAPI qui simule un flux temps réel
  (nouveau contrat toutes les 5s) et expose 15 outils analytiques, utilisés
  à la fois par le dashboard et par l'agent `MAEIA` (Llama 3.3 70B via Groq).
- **Frontend** (`mae_frontend/index.html`) : React + Recharts en un seul
  fichier HTML, aucune étape de build.

## Prérequis

- Python 3.11
- Un navigateur (le frontend n'a besoin de rien d'autre)
- Une clé API Groq gratuite ([console.groq.com](https://console.groq.com)) pour l'agent

## Installation & lancement

### 1. Pipeline de données (à exécuter une fois, dans l'ordre)

```bash
pip install -r requirements.txt
python 01_diagnostic.py     # profil statistique des CSV bruts (reports/)
python 02_cleaning.py       # nettoyage + enrichissement (processed_data/)
python 03_eda.py            # 6 graphiques d'analyse (plots/)
python 04_forecasting.py    # prévision CA 2026 (outputs/, plots/07_*)
python 05_clustering.py     # segmentation clients (outputs/, plots/08-10_*)
```

### 2. Backend (API + agent)

```bash
cd mae_backend
pip install -r requirements.txt   # dependances minimales de l'API (pas le requirements.txt racine)
```

Créer `mae_backend/.env` avec :
```
GROQ_API_KEY=votre_cle_groq
```

Puis :
```bash
uvicorn main:app --reload
```
API disponible sur `http://localhost:8000` (documentation interactive sur `/docs`).

### 3. Frontend

Ouvrir `mae_frontend/index.html` directement dans un navigateur — aucune
compilation requise. Si le backend tourne sur un autre host/port que
`localhost:8000`, ajouter `?api=http://host:port/api` à l'URL du frontend
plutôt que d'éditer le fichier.

## Tests

```bash
cd mae_backend
pytest test_main.py -v
```

## MLOps & observabilité

- Chaque exécution du pipeline et chaque interaction de l'agent est
  journalisée dans MLflow (`mlruns/`) — lancer `mlflow ui` pour explorer.
- La dérive des données est surveillée en continu (pas seulement à la
  demande), journalisée dans `mae_backend/logs/agent.log`, et visible en
  permanence dans la barre latérale du dashboard + la page **Monitoring**.

## À savoir avant d'évaluer ce projet

- Les libellés région/agence sont des **conventions de présentation**
  documentées dans le code (voir l'en-tête de `02_cleaning.py`), pas une
  correspondance géographique vérifiée — aucune table officielle n'était
  disponible.
- Les documents métier utilisés par le RAG de l'agent
  (`mae_backend/business_docs/`) sont **illustratifs**, rédigés pour ce
  projet en l'absence de documents MAE officiels. Remplaçables sans
  changement de code.
- L'authentification du frontend est un **prototype** (comptes en dur,
  session en localStorage) — à durcir avant tout déploiement réel.

## Structure du dépôt

```
01-05_*.py              Pipeline de données (diagnostic → clustering)
model_comparison.ipynb  Comparaison Régression Linéaire / SARIMA / Prophet
mae_backend/
  main.py               API FastAPI (KPIs, simulateur temps réel, endpoints)
  agent.py              Agent MAEIA (ReAct + Groq + RAG + mémoire long terme)
  business_docs/        Documents métier indexés pour le RAG
  test_main.py           Tests unitaires (pytest)
mae_frontend/
  index.html             Dashboard React (fichier unique, sans build)
raw_data/, processed_data/, outputs/, plots/, reports/
                         Données et artefacts du pipeline
```
