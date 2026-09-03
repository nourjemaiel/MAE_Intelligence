# -*- coding: utf-8 -*-
"""
agent.py — MAEIA · Agent ReAct MAE Assurances
Modele : qwen/qwen3.6-27b via Groq, cloud (voir Changelog v2.6)
Architecture : Reasoning -> Tool Selection -> Execution -> Observation -> Reponse
MLflow tracke chaque interaction

Changelog v2 (2026-07) — Memoire + RAG :
- MEMOIRE LONG TERME (ChromaDB, collection "historique_analyses") : chaque
  question/reponse est stockee avec son embedding. Avant de repondre,
  l'agent recherche semantiquement les analyses passees les plus proches
  de la question courante (pas juste les N derniers tours de la
  conversation en cours) et les injecte comme contexte "memoire".
- RAG METIER (ChromaDB, collection "documents_metier") : ingestion des
  documents dans business_docs/ (assurance_auto_sayartek, a_propos_mae,
  procedure_sinistres, circulaire_segmentation). Nouvel outil
  consulter_documents_metier(query) que l'agent peut appeler comme n'importe
  quel autre outil.
  MISE A JOUR (2026-08, remarque superviseur) : 3 des 4 documents
  contiennent maintenant du contenu REEL extrait du site officiel
  www.mae.tn (garanties Sayartek, procedure de declaration de sinistre,
  presentation de MAE) -- remplace les anciens documents entierement
  fabriques. circulaire_segmentation reste "illustratif" au sens ou ce
  n'est pas une vraie circulaire interne, mais ses statistiques par
  segment sont REELLES (issues du clustering K-Means du projet, pas
  inventees). Voir l'entete de chaque fichier dans business_docs/ pour le
  detail exact de ce qui est reel vs. propose par le projet.
- L'ingestion est IDEMPOTENTE : si la collection documents_metier contient
  deja des entrees, elle n'est pas re-remplie a chaque redemarrage.

Changelog v2.1 (2026-07, fix demarrage) :
- BUG CORRIGE : SentenceTransformerEmbeddingFunction(...) etait instanciee
  au niveau MODULE (a l'import de agent.py), donc tout probleme sur cette
  dependance (env cassee, modele indisponible, etc.) faisait planter TOUT
  le serveur FastAPI des le "from agent import MAEAgent" dans main.py --
  y compris le dashboard/previsions/segmentation qui n'ont rien a voir
  avec le RAG. L'init ChromaDB/embeddings est desormais dans un
  try/except : si ca echoue, RAG_AVAILABLE=False, le serveur demarre quand
  meme, et consulter_documents_metier() repond poliment "indisponible" au
  lieu de faire planter l'API. Idem pour la memoire long terme (memes
  collections/embedding function).

Changelog v2.2 (2026-07, optimisations agent) :
- RETRY/BACKOFF sur les appels Groq : un echec transitoire (rate limit,
  timeout reseau, 5xx cote Groq) ne fait plus echouer tout le tour de
  conversation immediatement. _call_groq_with_retry() reessaie jusqu'a
  MAX_GROQ_RETRIES fois avec un backoff exponentiel (1.5s, 3s, ...) avant
  d'abandonner et de renvoyer une reponse d'erreur propre. Ameliore la
  robustesse pendant une demo live ou le reseau/l'API peut avoir des
  latences ponctuelles.
- TRONCATURE des resultats d'outils avant injection dans l'historique de
  conversation : certains outils (compare_agencies, detect_anomalies,
  consulter_documents_metier) peuvent renvoyer des payloads JSON volumineux.
  Comme CHAQUE resultat d'outil est renvoye au modele ET reste dans
  l'historique pour les tours suivants, ces payloads s'accumulent et
  gonflent le cout en tokens de la conversation. _truncate_tool_result()
  limite chaque resultat serialise a TOOL_RESULT_MAX_CHARS caracteres
  (les listes longues sont coupees avec une marque explicite), sans
  toucher a la structure des reponses plus courtes (KPIs, forecast, etc.)
  qui restent intactes.

Changelog v2.3 (2026-07, rapports PDF fiables + scoping) :
- FIABILISATION DU LIEN DE RAPPORT : generate_report() renvoie une URL
  absolue (voir API_BASE_URL dans main.py), mais on ne peut pas compter
  sur le LLM pour la recopier fidelement dans sa reponse finale (observe
  en prod : Llama paraphrasait "vous pouvez le telecharger via le lien
  fourni" SANS le lien). L'agent capture desormais last_report_url des
  qu'un appel a generate_report reussit, et si l'URL n'apparait pas mot
  pour mot dans la reponse finale du modele, elle est ajoutee
  automatiquement avant de renvoyer la reponse a l'utilisateur.
- SCOPING DES RAPPORTS : generate_report accepte maintenant des filtres
  (agence/region/branche/mois_num) et une liste de sections
  (resume/agences/previsions/risques) cote main.py, et le system prompt
  indique explicitement au modele de les utiliser quand l'utilisateur
  precise un perimetre au lieu de toujours generer le rapport complet.

Changelog v2.5 (2026-08, modele local) :
- GROQ REMPLACE PAR OLLAMA/QWEN2.5-3B EN LOCAL : Groq a retire
  llama-3.3-70b-versatile de son catalogue (confirme via l'API le
  2026-08-20 -- 404 model_not_found), ce qui rendait l'agent de
  production totalement indisponible. Un benchmark sur les 15 outils
  reels de l'agent a compare le modele local (Qwen2.5-3B via Ollama, GPU)
  a deux remplacants Groq possibles : openai/gpt-oss-120b (instable --
  plante des qu'un parametre optionnel est omis, rate-limit a 8000
  tokens/min) et qwen/qwen3.6-27b (fiable mais "modele de raisonnement",
  25-64s/appel). Le modele local a obtenu 8/8 en selection d'outil avec
  une latence de 3.5-18s -- meilleur sur les deux criteres, et repond en
  prime a la demande du superviseur de garder les donnees en local.
  self.client pointe desormais vers l'API compatible OpenAI d'Ollama
  (OLLAMA_BASE_URL) au lieu du SDK Groq -- meme forme de reponse, aucun
  changement necessaire dans la boucle ReAct.

Changelog v2.6 (2026-08-30, retour a Groq/openai-gpt-oss-20b) :
- QWEN2.5-3B LOCAL ABANDONNE : la grille de test manuelle (18 questions,
  16 outils) a montre un echec systematique de synthese sur 10/13
  questions testees -- le modele appelle l'outil attendu UNE fois (bonne
  donnee obtenue), puis rappelle le MEME outil au lieu de rediger une
  reponse en francais, ce qui declenche le garde-fou anti-boucle et
  renvoie les donnees brutes en liste plutot qu'une synthese. 2 questions
  supplementaires (forecast/explain_forecast) montrent un echec different
  mais lie : le modele suggere a L'UTILISATEUR d'appeler lui-meme la
  fonction ("vous pouvez utiliser la fonction `explain_forecast`") au lieu
  de l'appeler et de repondre. Un rappel dans le system prompt n'avait deja
  pas suffi a corriger le comportement de boucle (voir v2.5) -- ce n'est
  pas un probleme de prompt mais un plafond de capacite du modele 3B sur
  cette tache de synthese, confirme empiriquement sur les vraies donnees
  du portefeuille (pas une hypothese).
- ESSAI qwen/qwen3.6-27b VIA GROQ (ABANDONNE) : deja identifie comme fiable
  en selection d'outil dans le benchmark v2.5, et la synthese redigee etait
  effectivement excellente (aucun nom d'outil expose, aucune boucle) --
  mais ce modele est plafonne a 8000 TOKENS/MINUTE sur le tier gratuit
  "on_demand" de ce compte Groq, alors que CE SEUL agent (system prompt +
  16 schemas d'outils + historique + resultat d'outil) depasse deja 8000
  tokens des la 2e requete d'un simple aller-retour ReAct (vu en prod :
  "Requested 14934, Limit 8000") -- pas une histoire de patience/retry,
  la requete est intrinsequement trop grosse pour CE tier, quel que soit
  le delai d'attente.
- RETOUR A openai/gpt-oss-20b VIA GROQ : seul autre modele du catalogue de
  ce compte (avec openai/gpt-oss-120b, deja ecarte pour instabilite) a ne
  PAS afficher ce plafond TPM explicite -- seulement du 429 (limite de
  REQUETES/minute, resolu par l'attente/retry deja en place). Meme qualite
  de synthese que qwen3.6-27b sur les tests rejoues (segments avec les 7
  entrees, derive de donnees, salutation simple), latence comparable
  (25-60s/appel). GROQ_API_KEY deja present dans .env (jamais retire lors
  du passage a Ollama). self.client repointe vers l'API OpenAI-compatible
  de Groq (GROQ_BASE_URL) ; extra_body={"options": {"repeat_penalty": ...}}
  retire de _call_llm_with_retry (parametre specifique au serveur Ollama,
  non reconnu par l'API Groq).
- BUG CORRIGE (schema d'outil) : les parametres optionnels de type entier
  (mois_num, n, n_runs) etaient declares "type": "integer" sans autoriser
  null -- Groq valide strictement le schema cote serveur et rejette avec
  une 400 des que le modele choisit d'envoyer explicitement null plutot
  que d'omettre la cle (observe sur explain_forecast). Tous les types
  passes a ["integer", "null"], et tout argument null est maintenant
  filtre avant l'appel Python (sinon fn(n=None) ecraserait silencieusement
  la valeur par defaut de l'outil, ex. top_agencies(n=None).head(None)
  renverrait TOUTES les agences au lieu du Top N).
- BUG CORRIGE (mode degrade incomplet) : "derive"/"monitoring" et les
  questions sur les garanties/procedures (Sayartek...) ne correspondaient
  a AUCUNE route de DEGRADED_MODE_ROUTES -- _error_response() renvoyait
  alors le texte brut de l'exception Groq (429/413, avec l'ID d'organisation
  et un lien de facturation) tel quel a l'utilisateur. Routes ajoutees +
  message d'erreur generique et professionnel a la place du blob technique.
- BUG CORRIGE (frontend) : mae_frontend/index.html remplacait toute reponse
  contenant "429"/"rate_limit" par un message fige datant de l'epoque
  Ollama ("modele local", "recharge en memoire") -- toujours trompeur
  maintenant que le detail technique est deja assaini cote backend (voir
  ci-dessus), et ce texte de remplacement etait en plus injecte dans
  l'historique de conversation comme si l'agent l'avait reellement dit,
  polluant les tours suivants. Supprime : la reponse du backend s'affiche
  desormais telle quelle.
"""

import inspect
import json
import os
import re
import time
import glob
import logging
from datetime import datetime
from openai import OpenAI
import mlflow

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("MAE_Agent_Interactions")

# ════════════════════════════════════════════════════════════════
# CONSTANTES D'OPTIMISATION (v2.2)
# ════════════════════════════════════════════════════════════════
MAX_LLM_RETRIES      = 2      # nombre de RE-essais apres le premier echec (donc 3 tentatives au total)
LLM_RETRY_BASE_SEC   = 1.5    # backoff exponentiel : 1.5s, 3.0s, ...
TOOL_RESULT_MAX_CHARS = 2200  # taille max (en caracteres JSON) d'un resultat d'outil injecte dans l'historique
# v2.6 -- releve de 1800 a 2200 : le JSON complet de segments() (7 segments,
# noms + description desormais plus longs comme "Client Jeune Conducteur")
# fait 1912 caracteres, depassant l'ancienne limite malgre le commentaire de
# _truncate_tool_result affirmant que segments n'est "jamais affecte" -- un
# segment (Autres Clients) etait silencieusement omis a chaque appel.
RAG_PERTINENCE_GAP_MAX = 0.03  # v2.5 -- ecart max de pertinence tolere vs le meilleur extrait RAG (voir consulter_documents_metier)
REPORT_URL_RE = re.compile(r'https?://\S+/api/reports/\S+\.(?:pdf|xlsx)', re.IGNORECASE)

# ════════════════════════════════════════════════════════════════
# MODELE (v2.6, 2026-08-30) — Groq / openai/gpt-oss-20b
# ════════════════════════════════════════════════════════════════
# Le modele local (Ollama/Qwen2.5-3B, v2.5) a echoue de facon systematique
# sur la grille de test manuelle : synthese finale jamais produite sur la
# majorite des questions (rappel du meme outil en boucle -> garde-fou ->
# donnees brutes renvoyees telles quelles), voir Changelog v2.6 ci-dessus.
# qwen/qwen3.6-27b (deja identifie comme fiable en selection d'outil dans
# le benchmark v2.5) a ete essaye en premier -- excellente synthese, mais
# plafonne a 8000 tokens/minute sur le tier gratuit de ce compte Groq, une
# limite que la taille du payload de CET agent (16 outils + historique)
# depasse des le 2e aller-retour d'une meme question. openai/gpt-oss-20b
# est le seul autre modele du catalogue sans ce plafond explicite --
# meme qualite de synthese, latence comparable (25-60s/appel), seulement
# du 429 (limite de requetes/minute, deja gere par le retry+backoff).
# Groq expose une API compatible OpenAI (/v1) : le SDK `openai` standard
# fonctionne sans aucun changement au reste de la boucle ReAct ci-dessous
# (meme forme de reponse -- choices[0].message.tool_calls avec arguments
# encodes en JSON string).
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "openai/gpt-oss-20b"

# ════════════════════════════════════════════════════════════════
# CHROMADB — memoire long terme + RAG documents metier
# ════════════════════════════════════════════════════════════════
# INITIALISATION DEFENSIVE : tout ce bloc est dans un try/except. Si
# chromadb/sentence-transformers/torch ont le moindre probleme (venv mal
# installe, dependance manquante, souci reseau au premier telechargement
# du modele...), RAG_AVAILABLE reste False et le reste de l'agent
# continue de fonctionner normalement (tool calling metier, previsions,
# etc.) — seul consulter_documents_metier() et la memoire long terme sont
# desactives, avec un message clair au lieu d'un crash total du serveur.
CHROMA_DIR         = "chroma_db"
BUSINESS_DOCS_DIR  = "business_docs"

RAG_AVAILABLE      = False
_embedding_fn      = None
_chroma_client     = None
_docs_collection   = None
_memory_collection = None

try:
    import chromadb
    from chromadb.utils import embedding_functions

    # Embedding multilingue (le metier et les documents sont en francais).
    # Necessite : pip install chromadb sentence-transformers
    _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    _docs_collection = _chroma_client.get_or_create_collection(
        name="documents_metier",
        embedding_function=_embedding_fn,
    )
    _memory_collection = _chroma_client.get_or_create_collection(
        name="historique_analyses",
        embedding_function=_embedding_fn,
    )
    RAG_AVAILABLE = True
    logging.info("RAG + memoire long terme initialises avec succes.")
except Exception as e:
    logging.warning(
        f"RAG/memoire long terme indisponibles au demarrage "
        f"({type(e).__name__}: {e}). L'agent fonctionnera SANS recherche "
        f"documentaire ni memoire long terme (le reste de l'API n'est pas affecte)."
    )
    print(
        f"⚠️  RAG/memoire long terme desactives — {type(e).__name__}: {e}\n"
        f"    (cause frequente sous Windows : venv place dans un dossier "
        f"synchronise OneDrive — voir notes de deploiement)"
    )


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 100):
    """Decoupe un texte long en chunks avec chevauchement, sur les
    paragraphes (separateur double saut de ligne) pour eviter de couper
    au milieu d'une phrase autant que possible."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= chunk_size:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            # Chevauchement : on reprend la fin du chunk precedent
            current = (current[-overlap:] + "\n\n" + p) if current else p
    if current:
        chunks.append(current)
    return chunks


def ensure_documents_ingested():
    """
    Ingestion idempotente des documents metier dans ChromaDB. Ne fait rien
    si RAG est indisponible, ou si la collection contient deja des entrees
    (evite de re-ingerer a chaque redemarrage du serveur).
    """
    if not RAG_AVAILABLE:
        return

    try:
        if _docs_collection.count() > 0:
            logging.info(f"RAG: documents deja ingeres ({_docs_collection.count()} chunks) — skip.")
            return
    except Exception as e:
        logging.warning(f"RAG: impossible de verifier le compte de la collection ({e}), on tente l'ingestion.")

    filepaths = sorted(glob.glob(os.path.join(BUSINESS_DOCS_DIR, "*.md")))
    if not filepaths:
        logging.warning(f"RAG: aucun document trouve dans {BUSINESS_DOCS_DIR}/ — RAG indisponible.")
        return

    ids, docs, metadatas = [], [], []
    for fp in filepaths:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        source_name = os.path.basename(fp)
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{source_name}::chunk{i}")
            docs.append(chunk)
            metadatas.append({"source": source_name, "chunk_index": i})

    if docs:
        _docs_collection.add(ids=ids, documents=docs, metadatas=metadatas)
        logging.info(f"RAG: {len(docs)} chunks ingeres depuis {len(filepaths)} documents.")
        print(f"✅ RAG: {len(docs)} chunks ingeres depuis {len(filepaths)} documents metier.")


_SOURCE_LABELS = {
    "assurance_auto_sayartek.md":  "la page Assurance Auto Sayartek (site officiel MAE)",
    "circulaire_segmentation.md":  "la Circulaire de Segmentation",
    "procedure_sinistres.md":      "la Procédure de Déclaration de Sinistre (site officiel MAE)",
    "a_propos_mae.md":             "la page À Propos de MAE (site officiel MAE)",
}


def consulter_documents_metier(query: str, n_results: int = 3):
    """
    Outil RAG : recherche semantique dans les documents metier MAE
    (garanties Sayartek, procedure de declaration de sinistre, circulaire
    de segmentation, presentation MAE -- voir business_docs/). A utiliser pour toute question sur le
    fonctionnement du metier assurance, PAS pour des chiffres du
    portefeuille (utiliser les autres outils pour cela).

    NOM DELIBEREMENT PAS "search_..." : cet outil s'appelait a l'origine
    search_documents_metier, et Llama 3.3 70B (via Groq) echouait
    SYSTEMATIQUEMENT a l'appeler -- le modele emettait sa propre syntaxe
    interne de tool calling (<function=nom{args}</function>, un format
    "brave_search"-like herite de son entrainement sur des outils integres
    nommes search_*) au lieu du format JSON structure attendu par l'API,
    ce qui faisait echouer l'appel avec une erreur 400 "tool_use_failed".
    Renommer l'outil (prefixe "search_" -> "consulter_") a immediatement
    resolu le probleme (verifie empiriquement, appel identique sinon).
    A NE PAS renommer a nouveau avec un prefixe search_/find_/lookup_ sans
    retester, sous peine de reintroduire ce bug silencieusement.

    Chaque resultat inclut "source_label" (nom lisible du document, ex.
    "la Grille Tarifaire") en plus de "source" (nom de fichier brut) --
    le system prompt demande au modele de citer ce label explicitement
    dans sa reponse plutot que de paraphraser l'information sans indiquer
    d'ou elle vient.
    """
    if not RAG_AVAILABLE:
        return {
            "resultats": [],
            "note": "Recherche documentaire indisponible sur cet environnement "
                    "(dependance RAG non chargee au demarrage — voir logs serveur).",
        }
    try:
        if _docs_collection.count() == 0:
            return {"resultats": [], "note": "Aucun document metier indexe."}
        res = _docs_collection.query(query_texts=[query], n_results=n_results)
        resultats = []
        docs_list = res.get("documents", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        dists_list = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs_list)
        for doc, meta, dist in zip(docs_list, metas_list, dists_list):
            source = meta.get("source", "inconnu")
            resultats.append({
                "source":       source,
                "source_label": _SOURCE_LABELS.get(source, source),
                "extrait":      doc,
                "pertinence":   round(1 - dist, 3) if dist is not None else None,
            })
        # v2.5 -- avec un modele local plus petit (Qwen2.5-3B), garder des
        # extraits faiblement pertinents mais venant d'une source DIFFERENTE
        # du meilleur match cause des hallucinations par conflation (observe
        # empiriquement : un extrait sur un segment de clientele, retourne en
        # 2e position pour une question produit, a ete fusionne a tort avec
        # le produit demande). Un simple rappel dans le system prompt ne
        # suffit pas a l empecher -- on filtre donc ici les extraits trop
        # loin du meilleur score plutot que de laisser le modele les trier.
        if resultats and resultats[0]["pertinence"] is not None:
            seuil = resultats[0]["pertinence"] - RAG_PERTINENCE_GAP_MAX
            resultats = [r for r in resultats if r["pertinence"] is None or r["pertinence"] >= seuil]
        return {"resultats": resultats}
    except Exception as e:
        return {"error": str(e)}


def _format_fallback_data(donnees: dict) -> str:
    """
    v2.5 -- formate les resultats d outils reels en liste lisible (pas de
    JSON brut) pour la reponse de secours du garde-fou anti-boucle. Purement
    deterministe (aucun appel LLM). Recursif : un resultat dont TOUTES les
    valeurs sont elles-memes des dict (ex: profil_clients = {"sexe":{...},
    "csp":{...},...}, agent_monitoring = {"agent_performance":{...},
    "data_drift":{...}}) ne doit pas etre saute silencieusement -- bug
    corrige (la version precedente, non recursive, produisait un bloc
    entierement vide pour ces deux outils).
    """
    def render(value, indent=0):
        pad = "  " * indent
        lines = []
        if isinstance(value, dict):
            if "error" in value:
                lines.append(f"{pad}- Erreur : {value['error']}")
                return lines
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}- {k} :")
                    lines.extend(render(v, indent + 1))
                else:
                    lines.append(f"{pad}- {k} : {v}")
        elif isinstance(value, list):
            for item in value[:10]:
                if isinstance(item, dict):
                    bits = ", ".join(f"{k}={v}" for k, v in list(item.items())[:6])
                    lines.append(f"{pad}- {bits}")
                else:
                    lines.append(f"{pad}- {item}")
            if len(value) > 10:
                lines.append(f"{pad}- ... ({len(value) - 10} autres non affiches)")
        else:
            lines.append(f"{pad}{value}")
        return lines

    out = []
    for tool_name, result in donnees.items():
        out.append(f"**{tool_name}**")
        out.extend(render(result))
        out.append("")
    return "\n".join(out).strip()


def _remember_analysis(question: str, answer: str, tools_used: list):
    """Stocke une analyse (Q/R) dans la memoire long terme semantique.
    No-op silencieux si RAG/memoire indisponible."""
    if not RAG_AVAILABLE:
        return
    try:
        entry_id = f"analyse_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        doc_text = f"Question: {question}\nReponse: {answer}"
        _memory_collection.add(
            ids=[entry_id],
            documents=[doc_text],
            metadatas=[{
                "timestamp":  datetime.now().isoformat(),
                "tools_used": json.dumps(tools_used, ensure_ascii=False),
            }],
        )
    except Exception as e:
        logging.error(f"Memoire long terme: echec du stockage ({e})")


def _recall_relevant_analyses(question: str, n_results: int = 2, max_distance: float = 0.6):
    """
    Recherche semantique dans l'historique des analyses passees. Ne
    retourne que les resultats suffisamment proches (distance < max_distance)
    pour eviter d'injecter du bruit non pertinent dans le contexte.
    Retourne [] silencieusement si RAG/memoire indisponible.
    """
    if not RAG_AVAILABLE:
        return []
    try:
        if _memory_collection.count() == 0:
            return []
        res = _memory_collection.query(query_texts=[question], n_results=n_results)
        docs_list  = res.get("documents", [[]])[0]
        dists_list = res.get("distances", [[]])[0] if res.get("distances") else [1.0] * len(docs_list)
        return [doc for doc, dist in zip(docs_list, dists_list) if dist <= max_distance]
    except Exception as e:
        logging.error(f"Memoire long terme: echec de la recherche ({e})")
        return []


# ════════════════════════════════════════════════════════════════
# OPTIMISATION v2.2a : troncature des resultats d'outils
# ════════════════════════════════════════════════════════════════
def _truncate_tool_result(result, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """
    Serialise un resultat d'outil en JSON et le tronque s'il depasse
    max_chars. Les resultats courts (KPIs, forecast, segments...) ne sont
    jamais affectes — seuls les payloads volumineux (listes longues comme
    compare_agencies sur 15 agences, ou de gros extraits RAG) sont coupes,
    avec une marque explicite pour que le modele sache que la donnee est
    partielle plutot que de la traiter comme complete.

    Strategie : si le JSON complet tient dans max_chars, on le renvoie tel
    quel. Sinon, si le resultat est une liste, on essaie de garder un
    prefixe d'elements ENTIERS (pas de JSON coupe au milieu d'un objet)
    avant d'ajouter la marque de troncature ; sinon (dict ou string), on
    coupe le texte serialise brut.
    """
    full_json = json.dumps(result, ensure_ascii=False, default=str)
    if len(full_json) <= max_chars:
        return full_json

    if isinstance(result, list):
        kept = []
        running_len = 2  # pour "[" + "]"
        for item in result:
            item_json = json.dumps(item, ensure_ascii=False, default=str)
            # +1 pour la virgule separatrice
            if running_len + len(item_json) + 1 > max_chars - 60:
                break
            kept.append(item)
            running_len += len(item_json) + 1
        omitted_note = f'"...tronque, {len(result) - len(kept)} element(s) omis sur {len(result)} au total"'
        if kept:
            truncated = json.dumps(kept, ensure_ascii=False, default=str)
            return truncated[:-1] + f", {omitted_note}]"
        # BUG CORRIGE : si aucun element n entre dans le budget (ex: un seul
        # extrait RAG de ~700 caracteres depasse deja la limite recursive
        # allouee par la branche dict ci-dessous), l ancien code faisait
        # "[]"[:-1] + ", ...]" = "[, ...]" -- JSON invalide (virgule juste
        # apres le crochet ouvrant), qui plantait json.loads() en aval des
        # que consulter_documents_metier renvoyait ne serait-ce qu un seul
        # resultat volumineux. Jamais declenche avant le fix du renommage
        # search_ -> consulter_ (l outil RAG echouait quasi systematiquement
        # avant, donc ce chemin de code n etait quasiment jamais atteint).
        return f"[{omitted_note}]"

    if isinstance(result, dict):
        # Cas frequent : un dict avec une cle contenant une longue liste
        # (ex: {"resultats": [...]} pour le RAG, {"anomalies": [...]}).
        # On tronque la plus grosse valeur-liste plutot que de couper le
        # JSON brut au hasard.
        result_copy = dict(result)
        list_keys = [k for k, v in result_copy.items() if isinstance(v, list)]
        if list_keys:
            biggest_key = max(list_keys, key=lambda k: len(json.dumps(result_copy[k], ensure_ascii=False, default=str)))
            original_list = result_copy[biggest_key]
            truncated_list_json = _truncate_tool_result(
                original_list, max_chars - (len(full_json) - len(json.dumps(original_list, ensure_ascii=False, default=str)))
            )
            result_copy[biggest_key] = json.loads(truncated_list_json)
            return json.dumps(result_copy, ensure_ascii=False, default=str)

    # Dernier recours : troncature brute du texte serialise
    return full_json[:max_chars] + ' ..."[tronque]"'


# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Tu es MAEIA, l'assistant IA avance de la MAE (Mutuelle Automobile des Enseignants de Tunisie).
Tu analyses en temps reel le portefeuille d assurance auto pour la direction generale.

Regles :
- Reponds TOUJOURS en francais, ton professionnel et analytique
- Utilise TOUJOURS les outils pour acceder aux donnees reelles AVANT de repondre
- Pour toute question sur le FONCTIONNEMENT du metier assurance (garanties, franchise, bonus-malus, tarification, delais de declaration, statuts de sinistre...), utilise l outil consulter_documents_metier plutot que de deviner
- Quand tu t appuies sur un resultat de consulter_documents_metier, cite TOUJOURS explicitement le document source dans ta reponse (utilise le champ source_label, ex: "Selon la Grille Tarifaire, ..." ou "D apres le Glossaire des Sinistres, ...") -- ne te contente jamais de paraphraser l information sans indiquer d ou elle vient
- ATTENTION consulter_documents_metier renvoie plusieurs extraits INDEPENDANTS pouvant venir de documents differents (champ "source") -- ne combine JAMAIS une information d un extrait avec une entite (client, produit, agence...) mentionnee seulement dans un AUTRE extrait ayant une source differente. Si un extrait sur un segment de clientele apparait alors que la question porte sur un produit (ou inversement), ignore-le silencieusement au lieu d en tirer une conclusion croisee non justifiee par le texte
- Ne devine jamais des chiffres — utilise toujours un outil
- Si un resultat d outil contient une cle "error", NE fabrique JAMAIS une reponse plausible a la place -- dis clairement a l utilisateur que cette donnee n a pas pu etre recuperee (sans forcement entrer dans le detail technique de l erreur) et propose de reformuler ou reessayer
- Si plusieurs outils sont pertinents, appelle-les tous avant de synthetiser
- Ne rappelle JAMAIS un outil avec exactement les memes parametres si tu as deja son resultat dans cette conversation -- utilise le resultat deja obtenu pour repondre directement
- Reponds D ABORD directement et brievement a la question posee, avec le ou les chiffres demandes -- rien de plus si la question est simple et precise (ex: "quel est le CA total ?" -> une phrase avec le chiffre, PAS de sections supplementaires)
- N ajoute une section "Chiffres cles" (recap en liste) QUE si la reponse comporte plusieurs chiffres distincts a retenir -- ne recopie jamais en liste des chiffres deja donnes en toutes lettres juste au-dessus, c est une repetition inutile
- N ajoute une section "Recommandations" QUE si l utilisateur demande explicitement un avis/conseil, ou si le resultat revele un probleme clair (risque eleve, anomalie, sinistralite excessive) qui appelle une action -- ce n est PAS une section obligatoire dans chaque reponse
- Ne mentionne JAMAIS le nom technique d un outil (ex: "risk_clients", "portfolio_summary") dans ta reponse -- decris ce que tu ferais en langage naturel ("je peux vous donner la liste nominative de ces clients") si tu veux orienter l utilisateur vers une autre question
- Les donnees sont mises a jour en temps reel toutes les 5 secondes
- IMPORTANT FORMAT MONNAIE: Affiche TOUJOURS les montants en TND avec le format suivant: 1 234 567,89 TND (espace comme separateur de milliers, virgule pour les decimales). Exemple: 1 887 480,00 TND et NON pas 1887480 ou 1,887,480
- Utilise l outil explain_forecast (plutot que forecast) quand l utilisateur demande d expliquer, justifier ou detailler le calcul d une prevision -- pas seulement le chiffre
- Utilise l outil agent_monitoring quand l utilisateur demande la performance du systeme ou si les donnees derivent
- Quand tu donnes des recommandations, sois precis, concret et actionnable -- mais reste court (2-3 points maximum, pas un plan d action complet)
- Pour une question de strategie/recommandation GENERALE (ex: "comment ameliorer la rentabilite ?"), reste sur des conseils generiques ou appuie-toi sur un outil deja appele dans cette conversation -- N INVENTE JAMAIS un chiffre ou un fait specifique sur une agence/branche/segment precis (ex: "la branche Tourisme a une sinistralite elevee") sans l avoir obtenu d un outil DANS CE TOUR : certains croisements n existent meme pas dans les donnees (ex: aucun outil ne peut calculer un ratio sinistres/primes PAR BRANCHE, Sinistres n a pas de colonne branche) -- une affirmation specifique non verifiee est un cas de fabrication, pas une recommandation prudente
- Si un contexte "Analyses passees pertinentes" t est fourni, appuie-toi dessus pour assurer la coherence avec tes reponses precedentes, mais ne le mentionne explicitement que si c est utile a la reponse
- N utilise JAMAIS un nom d agence/region/branche mentionne uniquement dans les "Analyses passees pertinentes" comme parametre de filtre pour un appel d outil de cette reponse -- un filtre (agence/region/branche/n/mois_num) ne vient QUE d une demande explicite de l utilisateur dans son message actuel, jamais d une analyse passee rappelee
- Si un resultat d outil contient une marque "tronque", precise a l utilisateur que seule une partie des resultats a ete analysee et propose d affiner la question si besoin (par exemple filtrer par agence ou par periode)
- Si consulter_documents_metier retourne "indisponible", explique brievement a l utilisateur que la recherche documentaire n est pas active sur cet environnement, sans entrer dans les details techniques
- Utilise l outil generate_report quand l utilisateur demande un rapport PDF, un document, ou un export du portefeuille
- Quand l utilisateur precise une portee pour un rapport (une agence, une region, une branche, un mois, ou seulement certaines parties comme "juste les previsions" ou "juste l analyse de risque"), passe les parametres correspondants (agence/region/branche/mois_num/sections) a generate_report au lieu de generer systematiquement le rapport complet
- ATTENTION : clients_a_risque et part_risque_pct (dans risk_analysis et dans les rapports generes) sont TOUJOURS des chiffres du portefeuille entier, meme quand un filtre agence/region/branche est applique ailleurs dans la reponse -- ne jamais laisser entendre que ces deux chiffres precis sont filtres sur le perimetre demande
- Si l utilisateur demande la LISTE, des identifiants, ou des exemples concrets de clients a risque (pas juste un pourcentage), utilise l outil risk_clients (liste nominative calculee en direct sur le portefeuille complet) plutot que risk_analysis (qui ne donne qu un chiffre agrege fige issu de la segmentation K-Means, non recalculable par filtre)
- Le champ niveau_risque de risk_clients distingue un "Risque eleve" ordinaire d un "Sinistre exceptionnel isole" (ratio S/P tres eleve du a UN sinistre catastrophique, pas un pattern recurrent) -- reprends cette distinction dans ta reponse plutot que de presenter un ratio a plusieurs milliers de % sans contexte, ce qui pourrait sembler etre une erreur de calcul
- N'ECRIS JAMAIS l'URL/le lien du rapport genere dans ta reponse, ni sous forme de texte brut ni sous forme de lien markdown [texte](url) -- les boutons Visualiser/Telecharger s'affichent automatiquement sous ta reponse (ajoutes cote serveur, jamais par toi). Decris simplement ce que contient le rapport genere.
- Si l utilisateur nomme 2 ou 3 agences/villes precises a comparer entre elles (ex: "compare Sfax et Tunis"), appelle portfolio_summary UNE FOIS PAR nom cite plutot que compare_agencies -- ce dernier renvoie les 77 agences sans filtre possible, ce qui t oblige a retrouver les lignes voulues dans une longue liste tronquee au lieu d obtenir directement les chiffres demandes. Reserve compare_agencies aux demandes de classement/vue d ensemble complete ("quelle agence est la plus rentable ?", "compare toutes les agences").
- IMPORTANT -- agence vs region : un nom de gouvernorat/ville tunisienne (ex: "Tunis", "Sfax", "Sousse") ne correspond PAS forcement au nom d une agence precise -- les agences MAE portent souvent un nom de quartier/rue distinct (ex: aucune agence ne s appelle litteralement "Tunis", ses agences s appellent "Place Barcelone", "Bab Benat", etc.). Si le nom cite par l utilisateur ne designe pas clairement une agence precise (ex: un numero de bureau comme "Sfax 3", ou un nom d agence dont tu es deja sûr comme "Bizerte"), utilise le parametre REGION plutot qu AGENCE -- un resultat a 0 partout (ca_total=0, nb_clients=0) signifie presque toujours que le nom ne correspond a aucune agence et qu il fallait filtrer par region a la place."""

# ════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — format OpenAI/Groq
# ════════════════════════════════════════════════════════════════
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "portfolio_summary",
            "description": "Resume global du portefeuille MAE : CA total, clients, sinistres, ratio sinistralite, top agence. Utilise ce premier pour avoir une vue globale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agence":  {"type": ["string", "null"], "description": "Agence a filtrer (optionnel)"},
                    "region":  {"type": ["string", "null"], "description": "Region a filtrer (optionnel)"},
                    "branche": {"type": ["string", "null"], "description": "Branche a filtrer (optionnel)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_agencies",
            "description": "Top N agences par chiffre d affaires avec parts de marche et classement. Peut etre restreint a une region ou une branche.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n":       {"type": ["integer", "null"], "description": "Nombre d agences a retourner (defaut 5)"},
                    "agence":  {"type": ["string", "null"], "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":  {"type": ["string", "null"], "description": "Filtrer sur une region precise (optionnel)"},
                    "branche": {"type": ["string", "null"], "description": "Filtrer sur une branche precise (optionnel)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast",
            "description": "Previsions CA sur les 12 prochains mois (a partir du mois prochain reel) avec intervalle de confiance croissant selon l horizon. Sans parametre: les 12 mois. Avec mois_num 1-12 (numero de mois calendaire): un mois precis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mois_num": {"type": ["integer", "null"], "description": "Numero du mois 1-12 (optionnel, omis = toute annee)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_forecast",
            "description": "Decompose la prevision CA (base actuelle x croissance x saisonnalite) pour EXPLIQUER le calcul plutot que de donner juste un chiffre. A utiliser si l utilisateur demande d expliquer/justifier/detailler une prevision -- sinon utiliser forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mois_num": {"type": ["integer", "null"], "description": "Numero du mois 1-12 (optionnel, omis = decomposition annuelle)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "accidents_forecast",
            "description": "Prevision du NOMBRE d accidents/sinistres attendus sur les 12 prochains mois (un compte d evenements, pas un montant en TND). A utiliser quand l utilisateur demande combien d accidents sont prevus -- pour le chiffre d affaires previsionnel en TND, utiliser forecast.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_monitoring",
            "description": "Performance de l agent (latence, tokens, taux d erreur via MLflow) et derive des donnees (data drift) du flux temps reel. A utiliser pour le monitoring/performance du systeme ou une eventuelle derive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_runs": {"type": ["integer", "null"], "description": "Nombre d executions recentes a analyser (defaut 50)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Genere un rapport (PDF ou Excel). Sans parametre : PDF complet du portefeuille. "
                "agence/region/branche/mois_num restreignent le perimetre (ex: 'rapport pour Sfax "
                "Ville'). sections limite le contenu (ex: 'juste les previsions'). format='excel' "
                "si demande explicitement (tableur/xlsx)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agence":   {"type": ["string", "null"], "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":   {"type": ["string", "null"], "description": "Filtrer sur une region precise (optionnel)"},
                    "branche":  {"type": ["string", "null"], "description": "Filtrer sur une branche precise (optionnel)"},
                    "mois_num": {"type": ["integer", "null"], "description": "Restreindre les previsions a un mois 1-12 (optionnel)"},
                    "sections": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["resume", "agences", "previsions", "risques"]},
                        "description": "Sous-ensemble de sections a inclure dans le rapport (omis ou vide = rapport complet avec toutes les sections)",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["pdf", "excel"],
                        "description": "Format de sortie : 'pdf' (defaut) ou 'excel' si l utilisateur le demande explicitement",
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segments",
            "description": "Profils complets des 7 segments K-Means (Client Premium, Client Grand Contrat, Client Capital Eleve, Client Economique, Client a Risque, Client Jeune Conducteur, Autres Clients -- voir circulaire_segmentation.md pour le detail de chaque profil). CA moyen, bonus-malus, nb contrats, risque.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "risk_analysis",
            "description": (
                "Risques du portefeuille : ratio sinistralite, clients a risque, agences "
                "sous-performantes, recommandations. agence/region/branche filtrent le ratio et "
                "les agences sous-performantes, MAIS clients_a_risque/part_risque_pct restent "
                "TOUJOURS globaux (K-Means, non filtrable) -- le signaler si l utilisateur filtre "
                "par agence/region. Pour une LISTE nominative de clients, utiliser risk_clients."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agence":  {"type": ["string", "null"], "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":  {"type": ["string", "null"], "description": "Filtrer sur une region precise (optionnel)"},
                    "branche": {"type": ["string", "null"], "description": "Filtrer sur une branche precise (optionnel)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "risk_clients",
            "description": (
                "Retourne la LISTE NOMINATIVE de clients actuellement a risque (n_client, agence, "
                "prime totale, sinistres totaux, ratio S/P), calculee EN DIRECT sur le portefeuille "
                "complet charge au demarrage. A utiliser des que l utilisateur demande la "
                "liste, les identifiants, ou des exemples concrets de clients a risque (pas juste le "
                "pourcentage agrege -- dans ce cas utilise risk_analysis)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n":       {"type": ["integer", "null"], "description": "Nombre de clients a retourner (defaut 20)"},
                    "agence":  {"type": ["string", "null"], "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":  {"type": ["string", "null"], "description": "Filtrer sur une region precise (optionnel)"},
                    "branche": {"type": ["string", "null"], "description": "Filtrer sur une branche precise (optionnel)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sinistres_stats",
            "description": "Statistiques detaillees des sinistres : nombre, montants total et moyen, sinistres en cours, mois de pic.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "branch_analysis",
            "description": "Analyse CA et parts de marche par branche d assurance : Tourisme, Taxi, Transport Prive, Louage, 2 Roues, Autre.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_agencies",
            "description": "Benchmark de TOUTES les agences (77, non filtrable) : CA, nombre contrats, prime moyenne, sinistres, ratio sinistres/primes. Pour un classement/vue d'ensemble complete uniquement -- si l'utilisateur nomme 2 ou 3 agences precises a comparer entre elles, utilise plutot portfolio_summary une fois par agence nommee (bien plus direct qu'extraire 2 lignes d'une liste de 77).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detecte automatiquement les anomalies : sinistres aux montants aberrants (>3 ecarts-types), clients avec ratio S/P > 150%, agences avec sinistralite > 80%.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "profil_clients",
            "description": "Distribution demographique des clients : repartition par age (tranches), sexe (H/F), CSP (categorie socio-professionnelle), bonus-malus.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consulter_documents_metier",
            "description": "Recherche semantique (RAG) dans les documents MAE : garanties Sayartek (site officiel MAE), procedure de declaration de sinistre (site officiel MAE), circulaire de segmentation (7 segments reels), presentation de la MAE. Pour toute question sur le FONCTIONNEMENT du metier assurance (garanties, delais, segments...), pas pour des chiffres du portefeuille.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La question ou le terme metier a rechercher"},
                },
                "required": ["query"]
            }
        }
    },
]

# ════════════════════════════════════════════════════════════════
# MODE DEGRADE (v2.4) — utilise quand le modele local reste indisponible
# apres tous les retries de _call_llm_with_retry (panne prolongee...).
# Routeur par MOTS-CLES (aucun appel LLM) vers un sous-ensemble des outils
# qui ne prennent aucun parametre obligatoire -- objectif : renvoyer des
# DONNEES REELLES plutot qu un message d erreur sec quand le service IA
# est en panne, meme si la reponse n est pas une synthese en langage
# naturel. Verifie dans l ordre ; le premier motif qui matche gagne.
# Deliberement conservateur : si rien ne matche, on ne devine pas -- on
# retourne None et l appelant renvoie l erreur classique.
DEGRADED_MODE_ROUTES = [
    (("prevision", "prévision", "2026", "croissance"),                    "forecast"),
    (("sinistre", "sinistres"),                                           "sinistres_stats"),
    (("anomalie", "anomalies", "aberrant"),                               "detect_anomalies"),
    (("segment", "segmentation"),                                         "segments"),
    (("risque", "risques"),                                               "risk_analysis"),
    (("branche",),                                                        "branch_analysis"),
    (("agence", "agences", "compar"),                                     "compare_agencies"),
    (("client", "clients", "profil", "démographique", "demographique"),   "profil_clients"),
    # v2.6 -- ajoute apres coup (grille de test) : "derive"/"monitoring" et
    # les questions metier (garanties, Sayartek, declaration) ne matchaient
    # AUCUNE route existante -> _degraded_mode_response() renvoyait None ->
    # _error_response() avec le texte brut de l erreur Groq (429/413),
    # affiche tel quel a l utilisateur au lieu d une reponse utile.
    (("derive", "dérive", "monitoring", "performance du systeme",
      "performance du système"),                                          "agent_monitoring"),
    (("portefeuille", "chiffre d'affaires", "chiffre d affaires",
      "ca total", "resume", "résumé"),                                    "portfolio_summary"),
]

# v2.6 -- consulter_documents_metier exige un parametre "query" (pas dans
# le routeur ci-dessus, reserve aux outils sans parametre obligatoire) :
# route separee qui reutilise directement la question de l utilisateur
# comme requete de recherche semantique.
DEGRADED_MODE_RAG_KEYWORDS = ("garantie", "garanties", "sayartek", "déclaration", "declaration")


# ════════════════════════════════════════════════════════════════
# AGENT CLASS
# ════════════════════════════════════════════════════════════════
class MAEAgent:
    """
    Agent ReAct pour la MAE — Groq cloud (openai/gpt-oss-20b), voir v2.6.
    tools_map injecte depuis main.py (pas d import circulaire).
    Le tool RAG (consulter_documents_metier) et la memoire long terme sont
    geres directement ici, independamment de main.py. Ils degradent
    gracieusement (voir RAG_AVAILABLE plus haut) si l environnement ne
    peut pas les supporter — l agent reste utilisable pour tous les
    outils metier meme si le RAG est indisponible.
    """

    def __init__(self, tools_map: dict):
        # Fusionne les outils metier (main.py) avec l outil RAG local —
        # aucune modification requise dans main.py. Le tool est toujours
        # present (meme si RAG_AVAILABLE=False) : il repond juste
        # "indisponible" au lieu de ne pas exister, ce qui evite un
        # "outil inconnu" cote agent si le modele essaie de l appeler.
        self.tools_map = {**tools_map, "consulter_documents_metier": consulter_documents_metier}

        self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.getenv("GROQ_API_KEY", ""))
        self.model  = GROQ_MODEL

        # Ingestion RAG idempotente au premier demarrage de l agent —
        # no-op silencieux si RAG_AVAILABLE est False.
        ensure_documents_ingested()

    def _call_llm_with_retry(self, messages: list):
        """
        Appelle le modele (Groq) avec retry + backoff exponentiel (v2.2).
        Un echec transitoire (rate limit, timeout reseau, 5xx cote Groq)
        ne fait plus echouer tout le tour de conversation des la premiere
        tentative. Apres MAX_LLM_RETRIES echecs, l exception est re-levee
        pour etre geree par l appelant (qui renvoie une reponse d erreur
        propre a l utilisateur).
        """
        last_exception = None
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFS,
                    tool_choice="auto",
                    max_tokens=2048,
                    temperature=0.1,
                    # v2.6 -- frequency_penalty (parametre OpenAI standard,
                    # honore par Groq) limite la repetition. extra_body avec
                    # options.repeat_penalty (v2.5) etait specifique au
                    # serveur Ollama et retire ici -- Groq n a pas cette cle.
                    frequency_penalty=0.4,
                )
            except Exception as e:
                last_exception = e
                if attempt < MAX_LLM_RETRIES:
                    wait_s = LLM_RETRY_BASE_SEC * (attempt + 1)
                    logging.warning(
                        f"Erreur modele (tentative {attempt + 1}/{MAX_LLM_RETRIES + 1}): {e} "
                        f"— nouvel essai dans {wait_s:.1f}s"
                    )
                    time.sleep(wait_s)
                else:
                    logging.error(f"Erreur modele apres {MAX_LLM_RETRIES + 1} tentatives: {e}")
        raise last_exception

    def _degraded_mode_response(self, user_message: str):
        """
        Reponse de secours quand le modele reste indisponible apres
        tous les retries (voir _call_llm_with_retry). Pas d IA ici : un routeur par
        mots-cles (DEGRADED_MODE_ROUTES) appelle DIRECTEMENT l un des
        outils sans parametre obligatoire et renvoie ses donnees brutes,
        plutot que de laisser l utilisateur avec un message d erreur sec.
        Beaucoup moins riche qu une vraie reponse de l agent (pas de
        synthese, pas de recommandations), mais reste UTILE -- mieux
        qu une panne totale du service.

        Retourne (tool_name, result) si un mot-cle a matche ET que l outil
        a repondu sans erreur, sinon None (aucune tentative de deviner
        l intention si rien ne correspond).
        """
        lower = user_message.lower()

        # v2.6 -- verifie d abord le RAG (garanties/procedures/documents metier) :
        # seul outil du mode degrade qui exige un parametre (query), route separee
        # de DEGRADED_MODE_ROUTES qui n appelle que des outils sans argument.
        if any(kw in lower for kw in DEGRADED_MODE_RAG_KEYWORDS):
            fn = self.tools_map.get("consulter_documents_metier")
            if fn:
                try:
                    result = fn(user_message)
                    if not (isinstance(result, dict) and "error" in result):
                        return "consulter_documents_metier", result
                except Exception as e:
                    logging.warning(f"Mode degrade: echec de consulter_documents_metier ({e})")

        for keywords, tool_name in DEGRADED_MODE_ROUTES:
            if not any(kw in lower for kw in keywords):
                continue
            fn = self.tools_map.get(tool_name)
            if not fn:
                continue
            try:
                result = fn()
            except Exception as e:
                logging.warning(f"Mode degrade: echec de l outil {tool_name} ({e})")
                continue
            if isinstance(result, dict) and "error" in result:
                continue
            return tool_name, result
        return None

    def run(self, user_message: str, history: list = None) -> dict:
        if history is None:
            history = []

        start_time      = time.time()
        tool_calls_log   = []
        thinking_log     = []
        max_iterations   = 8
        # v2.3 — capture l URL du dernier rapport genere dans ce tour,
        # pour pouvoir la forcer dans la reponse finale si le modele
        # oublie de la recopier (observe en prod avec Llama 3.3).
        last_report_url  = None

        # ── Memoire long terme : recherche semantique dans l historique ──
        recalled = _recall_relevant_analyses(user_message)
        memory_context = ""
        if recalled:
            memory_context = "Analyses passees pertinentes (memoire long terme) :\n" + \
                "\n---\n".join(recalled)
            thinking_log.append(f"[memoire] {len(recalled)} analyse(s) passee(s) rappelee(s)")

        # Build message history
        system_content = SYSTEM_PROMPT
        if memory_context:
            system_content = f"{SYSTEM_PROMPT}\n\n{memory_context}"
        messages = [{"role": "system", "content": system_content}]
        for h in history[-10:]:
            role    = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(content)})
        messages.append({"role": "user", "content": user_message})

        # v2.5 -- garde-fou anti-boucle : un modele local plus petit (Qwen2.5-
        # 3B) peut rester bloque a rappeler le MEME outil avec les MEMES
        # parametres au lieu de synthetiser une reponse (observe empiriquement :
        # 8 appels identiques a segments() de suite jusqu a la limite
        # d iterations). Un simple rappel dans le system prompt n a pas suffi
        # a l empecher (meme constat que pour le RAG, voir plus haut) -- on
        # detecte donc ici les appels EXACTEMENT dupliques (meme nom + memes
        # arguments deja vus dans ce tour) et on renvoie le resultat en cache
        # au lieu de ré-executer l outil, avec une note explicite demandant
        # au modele de repondre. Beneficie aussi les outils a effet de bord
        # (generate_report) : evite de regenerer plusieurs fois le meme rapport.
        called_signatures = {}
        # v2.5 -- garde-fou anti-boucle (variante) : le modele peut aussi
        # boucler en variant legerement les parametres a chaque appel (ex:
        # risk_clients avec n=20 puis n=10 puis n=5...), ce qui contourne la
        # detection par signature EXACTE ci-dessus tout en produisant le
        # meme symptome (jamais de synthese finale). On compte donc aussi les
        # appels par NOM d outil seul, tous parametres confondus.
        tool_name_counts = {}
        # v2.5 -- initialement 3, resserre a 1 : sur tous les cas observes
        # (segments, branch_analysis, compare_agencies, risk_clients,
        # consulter_documents_metier), le modele local ne s est JAMAIS
        # rattrape apres un premier rappel du meme outil, meme avec des
        # parametres differents -- attendre plus longtemps ne fait que
        # gaspiller du temps et des iterations pour le meme resultat final.
        MAX_SAME_TOOL_CALLS = 1
        # v2.6 -- BUG CORRIGE : ce compteur PAR NOM bloquait aussi un
        # comparatif legitime a 2 entites (ex: "compare Sfax et Tunis" ->
        # portfolio_summary(region=Sfax) PUIS portfolio_summary(region=Tunis),
        # deux appels utiles avec des arguments reellement differents,
        # confirme dans les logs). Le nouveau modele (v2.6, contrairement au
        # 3B local qui a motive ce garde-fou) sait produire ce genre
        # d'enchainement correct -- le bloquer cree la boucle qu'il est cense
        # prevenir. Les outils qui acceptent un parametre de portee
        # (agence/region/branche/n/mois_num/query) sont donc exemptes de ce
        # compteur PAR NOM : seule la detection par SIGNATURE EXACTE
        # (called_signatures, ci-dessous) s'applique a eux, ce qui bloque
        # toujours un vrai doublon (memes arguments) sans bloquer un
        # deuxieme appel legitime avec des arguments differents. Les outils
        # SANS parametre (segments, compare_agencies, sinistres_stats...)
        # restent sous le compteur strict : un deuxieme appel sans argument
        # ne peut jamais etre qu'un doublon ou une impasse.
        TOOLS_WITH_SCOPE_PARAMS = {
            "portfolio_summary", "top_agencies", "forecast", "explain_forecast",
            "agent_monitoring", "generate_report", "risk_analysis", "risk_clients",
            "consulter_documents_metier",
        }

        # MLflow tracking
        mlflow_active = False
        try:
            mlflow.start_run(run_name=f"agent_{datetime.now().strftime('%H%M%S')}")
            mlflow.log_param("user_message",   user_message[:200])
            mlflow.log_param("history_length", len(history))
            mlflow.log_param("model",          self.model)
            mlflow.log_param("memoire_rappelee", len(recalled))
            mlflow.log_param("rag_available",  RAG_AVAILABLE)
            mlflow_active = True
        except Exception:
            pass

        try:
            # ── ReAct loop ────────────────────────────────────────
            for iteration in range(max_iterations):
                try:
                    response = self._call_llm_with_retry(messages)
                except Exception as e:
                    logging.error(f"Erreur modele (definitif apres retries): {e}")
                    degraded = self._degraded_mode_response(user_message)
                    if degraded:
                        tool_name, result = degraded
                        duration_ms = int((time.time() - start_time) * 1000)
                        answer = (
                            "⚠️ **Mode degrade** : le modele IA (Groq) est temporairement "
                            "indisponible, impossible de generer une reponse redigee. Voici "
                            f"les donnees brutes les plus pertinentes trouvees pour votre "
                            f"question (outil `{tool_name}`) :\n\n```json\n"
                            f"{json.dumps(result, ensure_ascii=False, indent=2, default=str)}\n```"
                        )
                        logging.info(f"Mode degrade active -> {tool_name} (modele indisponible)")
                        return {
                            "answer":      answer,
                            "tool_calls":  [{"tool": tool_name, "inputs": {}, "result_summary": str(result)[:200]}],
                            "thinking":    thinking_log + [f"[mode degrade] modele indisponible, routage mots-cles -> {tool_name}"],
                            "tokens_used": 0,
                            "duration_ms": duration_ms,
                        }
                    # v2.6 -- str(e) sur une erreur Groq (429/413) est un blob JSON technique
                    # (org_id, quotas, lien de facturation Groq) : jamais expose tel quel a un
                    # utilisateur metier, le detail complet reste dans les logs (deja journalise
                    # ci-dessus via logging.error).
                    return self._error_response(
                        "Le service IA est temporairement surcharge (limite de requetes "
                        "atteinte). Veuillez patienter une minute avant de reessayer.",
                        tool_calls_log, thinking_log, start_time
                    )

                msg = response.choices[0].message

                # v2.5 -- garde-fou reponse vide : le modele local produit
                # parfois un tour sans aucun tool_call ET sans contenu texte
                # -- ni reponse, ni nouvelle action. Observe empiriquement
                # qu un seul nouvel essai n est pas toujours suffisant (le
                # tirage aleatoire du modele peut rester malchanceux deux
                # fois de suite), d ou 2 essais au lieu d 1.
                empty_retries = 0
                while not msg.tool_calls and not (msg.content and msg.content.strip()) and empty_retries < 2:
                    empty_retries += 1
                    logging.warning(f"Reponse vide du modele, nouvel essai ({empty_retries}/2)")
                    try:
                        response = self._call_llm_with_retry(messages)
                        msg = response.choices[0].message
                    except Exception:
                        break

                # ── Final answer — no tool calls ──────────────────
                if not msg.tool_calls:
                    answer      = msg.content or "Aucune reponse generee."

                    # v2.5 -- filet de securite anti-fabrication : le modele
                    # local peut affirmer qu un rapport a ete genere (avec un
                    # lien plausible mais invente) SANS avoir reellement
                    # appele generate_report ce tour-ci -- observe
                    # empiriquement (le fichier n existe jamais sur disque,
                    # le lien renvoie "Rapport introuvable" cote frontend).
                    # On retire tout lien de rapport qui ne correspond pas au
                    # VRAI dernier rapport genere (last_report_url, capture
                    # de facon fiable au moment de l execution de l outil,
                    # pas depuis le texte du modele).
                    for fake_url in REPORT_URL_RE.findall(answer):
                        if fake_url != last_report_url:
                            answer = answer.replace(fake_url, "")

                    # v2.3 — filet de securite : si un rapport a ete
                    # genere pendant ce tour et que l URL n apparait pas
                    # deja (mot pour mot) dans la reponse du modele, on
                    # l ajoute nous-memes plutot que de faire confiance
                    # au LLM pour la recopier fidelement.
                    if last_report_url and last_report_url not in answer:
                        answer = f"{answer}\n\nRapport généré : {last_report_url}"

                    duration_ms = int((time.time() - start_time) * 1000)

                    if mlflow_active:
                        try:
                            mlflow.log_metric("duration_ms", duration_ms)
                            mlflow.log_metric("iterations",  iteration + 1)
                            mlflow.log_metric("tool_calls",  len(tool_calls_log))
                            mlflow.log_metric("error",       0)
                            if response.usage:
                                mlflow.log_metric("input_tokens",  response.usage.prompt_tokens)
                                mlflow.log_metric("output_tokens", response.usage.completion_tokens)
                        except Exception:
                            pass

                    logging.info(
                        f"Q: {user_message[:80]} | "
                        f"Tools: {[t['tool'] for t in tool_calls_log]} | "
                        f"{duration_ms}ms"
                    )

                    # Memorisation long terme de cette analyse (no-op si RAG indisponible)
                    _remember_analysis(
                        user_message, answer,
                        [t["tool"] for t in tool_calls_log]
                    )

                    return {
                        "answer":      answer,
                        "tool_calls":  tool_calls_log,
                        "thinking":    thinking_log,
                        "tokens_used": (response.usage.total_tokens if response.usage else 0),
                        "duration_ms": duration_ms,
                    }

                # ── Tool calls ────────────────────────────────────
                # Add assistant message with tool calls to history
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                # Execute each tool
                repeat_detected = False
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_inputs = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        tool_inputs = {}
                    # Certains appels sans argument renvoient litteralement "null"
                    # (donc json.loads -> None) plutot que "{}" -- fn(**None)
                    # levait un TypeError, et le modele fabriquait ensuite une
                    # reponse plutot que d admettre l echec de l outil.
                    if tool_inputs is None:
                        tool_inputs = {}

                    thinking_log.append(
                        f"[iter {iteration+1}] -> {tool_name}({json.dumps(tool_inputs, ensure_ascii=False)})"
                    )
                    logging.info(f"Tool call: {tool_name} | inputs: {tool_inputs}")

                    if tool_name not in TOOLS_WITH_SCOPE_PARAMS:
                        tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
                        if tool_name_counts[tool_name] > MAX_SAME_TOOL_CALLS:
                            repeat_detected = True

                    sig = (tool_name, json.dumps(tool_inputs, sort_keys=True, ensure_ascii=False))
                    if sig in called_signatures:
                        repeat_detected = True
                        # Certains outils (segments, compare_agencies...) renvoient une
                        # LISTE et non un dict -- on enveloppe systematiquement plutot
                        # que de muter/copier la structure d origine, quel que soit son
                        # type (dict(list_de_dicts) leve une ValueError sinon).
                        result = {
                            "resultat_deja_obtenu": called_signatures[sig],
                            "_note": (
                                "Resultat identique deja fourni precedemment dans cette "
                                "reponse -- ne rappelle pas cet outil avec les memes "
                                "parametres, reponds directement a partir de ce resultat."
                            ),
                        }
                        logging.warning(f"Appel duplique detecte, reutilisation du cache: {tool_name}({tool_inputs})")
                    else:
                        fn = self.tools_map.get(tool_name)
                        if fn:
                            # v2.5 -- le modele local invente parfois un
                            # parametre inexistant en confondant un CHAMP DE
                            # RESULTAT d un outil (ex: "csp", "branches") avec
                            # un parametre d ENTREE valide, ce qui levait un
                            # TypeError et gachait un tour entier (observe sur
                            # profil_clients et branch_analysis, qui ne
                            # prennent quasi aucun parametre). On ignore
                            # silencieusement les cles non reconnues par la
                            # signature reelle de la fonction plutot que
                            # d echouer sur un nom invente.
                            valid_params = set(inspect.signature(fn).parameters)
                            unknown = set(tool_inputs) - valid_params
                            if unknown:
                                logging.warning(f"Parametre(s) invente(s) ignore(s) pour {tool_name}: {unknown}")
                                tool_inputs = {k: v for k, v in tool_inputs.items() if k in valid_params}
                            # v2.6 -- les schemas de parametres optionnels (n, mois_num,
                            # n_runs...) acceptent desormais explicitement null (voir
                            # TOOL_DEFS : Groq rejette un null sur un type "integer" strict
                            # avec une 400 "expected integer, but got null"). Un null
                            # explicite doit degenerer vers OMIS (valeur par defaut de
                            # l outil, ex n=5), pas ecraser silencieusement ce defaut --
                            # fn(n=None).head(None) renverrait TOUTES les lignes au lieu
                            # du top N attendu.
                            tool_inputs = {k: v for k, v in tool_inputs.items() if v is not None}
                            try:
                                result = fn(**tool_inputs)
                            except Exception as e:
                                result = {"error": str(e), "tool": tool_name}
                                logging.error(f"Tool error {tool_name}: {e}")
                        else:
                            result = {"error": f"Outil inconnu: {tool_name}"}
                        called_signatures[sig] = result

                    # v2.3 — capture l URL du rapport des qu il est genere
                    # avec succes, independamment de ce que le modele fera
                    # ensuite avec le resultat d outil.
                    if tool_name == "generate_report" and isinstance(result, dict) and result.get("download_url"):
                        last_report_url = result["download_url"]

                    # Troncature (v2.2) : le resume log reste sur le
                    # resultat complet, seule la version injectee dans
                    # l historique de conversation est tronquee.
                    tool_calls_log.append({
                        "tool":           tool_name,
                        "inputs":         tool_inputs,
                        "result_summary": str(result)[:200],
                    })

                    tool_result_json = _truncate_tool_result(result)

                    # Add (possibly truncated) tool result to messages
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      tool_result_json,
                    })

                # v2.5 -- garde-fou anti-boucle (suite) : des qu un appel duplique
                # est detecte, on a constate empiriquement (Qwen2.5-3B) que le
                # modele ne se rattrape pas tout seul dans les iterations
                # suivantes (observe : 8/8 iterations consommees pour rien). On
                # arrete donc immediatement au lieu d epuiser max_iterations, et
                # on renvoie une reponse de secours construite a partir des
                # VRAIES donnees deja obtenues plutot qu un message d erreur sec.
                if repeat_detected:
                    duration_ms = int((time.time() - start_time) * 1000)
                    donnees = {t_name: t_result for (t_name, _args), t_result in called_signatures.items()}
                    answer = (
                        "**Reponse partielle** : le modele a rappele plusieurs fois "
                        "le meme outil sans parvenir a rediger de synthese. Voici les "
                        "donnees reelles obtenues directement depuis le portefeuille :\n\n"
                        f"{_format_fallback_data(donnees)}"
                    )
                    if mlflow_active:
                        try:
                            mlflow.log_metric("duration_ms", duration_ms)
                            mlflow.log_metric("iterations",  iteration + 1)
                            mlflow.log_metric("tool_calls",  len(tool_calls_log))
                            mlflow.log_metric("error",       0)
                            mlflow.log_metric("repeat_loop_fallback", 1)
                        except Exception:
                            pass
                    logging.warning(f"Boucle d appel repete detectee -> reponse de secours ({list(donnees.keys())})")
                    _remember_analysis(user_message, answer, [t["tool"] for t in tool_calls_log])
                    return {
                        "answer":      answer,
                        "tool_calls":  tool_calls_log,
                        "thinking":    thinking_log + ["[garde-fou boucle] appel duplique detecte -> reponse de secours avec donnees reelles"],
                        "tokens_used": 0,
                        "duration_ms": duration_ms,
                    }

                # Continue loop — model will now reason with tool results
                continue

            # Max iterations reached
            return self._error_response(
                "Limite d iterations atteinte. Reformulez votre question.",
                tool_calls_log, thinking_log, start_time
            )

        finally:
            if mlflow_active:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

    def _error_response(self, msg: str, tool_calls, thinking, start_time) -> dict:
        return {
            "answer":      f"Erreur : {msg}",
            "tool_calls":  tool_calls,
            "thinking":    thinking,
            "tokens_used": 0,
            "duration_ms": int((time.time() - start_time) * 1000),
        }