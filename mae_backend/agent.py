# -*- coding: utf-8 -*-
"""
agent.py — MAEIA · Agent ReAct MAE Assurances
Modele : Llama 3.3 70B via Groq (gratuit, rapide, tool calling)
Architecture : Reasoning -> Tool Selection -> Execution -> Observation -> Reponse
MLflow tracke chaque interaction

Changelog v2 (2026-07) — Memoire + RAG :
- MEMOIRE LONG TERME (ChromaDB, collection "historique_analyses") : chaque
  question/reponse est stockee avec son embedding. Avant de repondre,
  l'agent recherche semantiquement les analyses passees les plus proches
  de la question courante (pas juste les N derniers tours de la
  conversation en cours) et les injecte comme contexte "memoire".
- RAG METIER (ChromaDB, collection "documents_metier") : ingestion des
  documents dans business_docs/ (conditions generales, grille tarifaire,
  circulaire segmentation, glossaire sinistres). Nouvel outil
  consulter_documents_metier(query) que l'agent peut appeler comme n'importe
  quel autre outil.
  IMPORTANT : ces documents sont ILLUSTRATIFS, rediges pour ce PFE car
  aucun document officiel MAE n'a ete fourni — voir l'entete de chaque
  fichier dans business_docs/. A remplacer par les vrais documents MAE
  si/quand ils deviennent disponibles (il suffira de les deposer dans
  business_docs/ et de relancer l'ingestion, aucun autre changement requis).
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
"""

import json
import os
import time
import glob
import logging
from datetime import datetime
from groq import Groq
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
MAX_GROQ_RETRIES     = 2      # nombre de RE-essais apres le premier echec (donc 3 tentatives au total)
GROQ_RETRY_BASE_SEC  = 1.5    # backoff exponentiel : 1.5s, 3.0s, ...
TOOL_RESULT_MAX_CHARS = 1800  # taille max (en caracteres JSON) d'un resultat d'outil injecte dans l'historique

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
    "conditions_generales_extrait.md": "les Conditions Générales",
    "grille_tarifaire.md":             "la Grille Tarifaire",
    "circulaire_segmentation.md":      "la Circulaire de Segmentation",
    "glossaire_sinistres.md":          "le Glossaire des Sinistres",
}


def consulter_documents_metier(query: str, n_results: int = 3):
    """
    Outil RAG : recherche semantique dans les documents metier MAE
    (conditions generales, grille tarifaire, circulaire segmentation,
    glossaire sinistres). A utiliser pour toute question sur le
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
        return {"resultats": resultats}
    except Exception as e:
        return {"error": str(e)}


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
- Ne devine jamais des chiffres — utilise toujours un outil
- Si un resultat d outil contient une cle "error", NE fabrique JAMAIS une reponse plausible a la place -- dis clairement a l utilisateur que cette donnee n a pas pu etre recuperee (sans forcement entrer dans le detail technique de l erreur) et propose de reformuler ou reessayer
- Si plusieurs outils sont pertinents, appelle-les tous avant de synthetiser
- Structure ta reponse : Analyse -> Chiffres cles -> Recommandations
- Les donnees sont mises a jour en temps reel toutes les 5 secondes
- IMPORTANT FORMAT MONNAIE: Affiche TOUJOURS les montants en TND avec le format suivant: 1 234 567,89 TND (espace comme separateur de milliers, virgule pour les decimales). Exemple: 1 887 480,00 TND et NON pas 1887480 ou 1,887,480
- Utilise l outil explain_forecast (plutot que forecast) quand l utilisateur demande d expliquer, justifier ou detailler le calcul d une prevision -- pas seulement le chiffre
- Utilise l outil agent_monitoring quand l utilisateur demande la performance du systeme ou si les donnees derivent
- Sois precis, concret et actionnable dans tes recommandations
- Si un contexte "Analyses passees pertinentes" t est fourni, appuie-toi dessus pour assurer la coherence avec tes reponses precedentes, mais ne le mentionne explicitement que si c est utile a la reponse
- Si un resultat d outil contient une marque "tronque", precise a l utilisateur que seule une partie des resultats a ete analysee et propose d affiner la question si besoin (par exemple filtrer par agence ou par periode)
- Si consulter_documents_metier retourne "indisponible", explique brievement a l utilisateur que la recherche documentaire n est pas active sur cet environnement, sans entrer dans les details techniques
- Utilise l outil generate_report quand l utilisateur demande un rapport PDF, un document, ou un export du portefeuille
- Quand l utilisateur precise une portee pour un rapport (une agence, une region, une branche, un mois, ou seulement certaines parties comme "juste les previsions" ou "juste l analyse de risque"), passe les parametres correspondants (agence/region/branche/mois_num/sections) a generate_report au lieu de generer systematiquement le rapport complet
- ATTENTION : clients_a_risque et part_risque_pct (dans risk_analysis et dans les rapports generes) sont TOUJOURS des chiffres du portefeuille entier, meme quand un filtre agence/region/branche est applique ailleurs dans la reponse -- ne jamais laisser entendre que ces deux chiffres precis sont filtres sur le perimetre demande
- Si l utilisateur demande la LISTE, des identifiants, ou des exemples concrets de clients a risque (pas juste un pourcentage), utilise l outil risk_clients (liste nominative calculee en direct sur le portefeuille complet) plutot que risk_analysis (qui ne donne qu un chiffre agrege fige issu de la segmentation K-Means, non recalculable par filtre)
- Le champ niveau_risque de risk_clients distingue un "Risque eleve" ordinaire d un "Sinistre exceptionnel isole" (ratio S/P tres eleve du a UN sinistre catastrophique, pas un pattern recurrent) -- reprends cette distinction dans ta reponse plutot que de presenter un ratio a plusieurs milliers de % sans contexte, ce qui pourrait sembler etre une erreur de calcul
- Indique toujours l'URL directe du rapport généré sous forme de texte brut cliquable que l'utilisateur peut copier."""

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
                    "agence":  {"type": "string", "description": "Agence a filtrer (optionnel)"},
                    "region":  {"type": "string", "description": "Region a filtrer (optionnel)"},
                    "branche": {"type": "string", "description": "Branche a filtrer (optionnel)"},
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
                    "n":       {"type": "integer", "description": "Nombre d agences a retourner (defaut 5)"},
                    "agence":  {"type": "string",  "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":  {"type": "string",  "description": "Filtrer sur une region precise (optionnel)"},
                    "branche": {"type": "string",  "description": "Filtrer sur une branche precise (optionnel)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast",
            "description": "Previsions CA 2026 avec intervalle de confiance 95%. Sans parametre: toute l annee. Avec mois_num 1-12: un mois precis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mois_num": {"type": "integer", "description": "Numero du mois 1-12 (optionnel, omis = toute annee)"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_forecast",
            "description": "Decompose la prevision CA (base 2025 x croissance x saisonnalite) pour EXPLIQUER le calcul plutot que de donner juste un chiffre. A utiliser si l utilisateur demande d expliquer/justifier/detailler une prevision -- sinon utiliser forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mois_num": {"type": "integer", "description": "Numero du mois 1-12 (optionnel, omis = decomposition annuelle)"},
                },
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
                    "n_runs": {"type": "integer", "description": "Nombre d executions recentes a analyser (defaut 50)"},
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
                    "agence":   {"type": "string",  "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":   {"type": "string",  "description": "Filtrer sur une region precise (optionnel)"},
                    "branche":  {"type": "string",  "description": "Filtrer sur une branche precise (optionnel)"},
                    "mois_num": {"type": "integer", "description": "Restreindre les previsions a un mois 1-12 (optionnel)"},
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
            "description": "Profils complets des 4 segments K-Means : Premium, Standard, Occasionnel, A Risque. CA moyen, bonus-malus, nb contrats, risque.",
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
                    "agence":  {"type": "string", "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":  {"type": "string", "description": "Filtrer sur une region precise (optionnel)"},
                    "branche": {"type": "string", "description": "Filtrer sur une branche precise (optionnel)"},
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
                    "n":       {"type": "integer", "description": "Nombre de clients a retourner (defaut 20)"},
                    "agence":  {"type": "string",  "description": "Filtrer sur une agence precise (optionnel)"},
                    "region":  {"type": "string",  "description": "Filtrer sur une region precise (optionnel)"},
                    "branche": {"type": "string",  "description": "Filtrer sur une branche precise (optionnel)"},
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
            "description": "Benchmark detaille entre toutes les agences : CA, nombre contrats, prime moyenne, sinistres, ratio sinistres/primes. Ideal pour comparer les performances.",
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
            "description": "Recherche semantique (RAG) dans les documents MAE : conditions generales, grille tarifaire, circulaire de segmentation, glossaire des sinistres. Pour toute question sur le FONCTIONNEMENT du metier assurance (garanties, franchise, bonus-malus, delais...), pas pour des chiffres du portefeuille.",
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
# MODE DEGRADE (v2.4) — utilise quand Groq reste indisponible apres tous
# les retries de _call_groq_with_retry (panne prolongee, quota epuise...).
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
    (("portefeuille", "chiffre d'affaires", "chiffre d affaires",
      "ca total", "resume", "résumé"),                                    "portfolio_summary"),
]


# ════════════════════════════════════════════════════════════════
# AGENT CLASS
# ════════════════════════════════════════════════════════════════
class MAEAgent:
    """
    Agent ReAct pour la MAE — Groq + Llama 3.3 70B.
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

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY non configuree. "
                "Creez un compte gratuit sur console.groq.com et ajoutez la cle dans .env"
            )
        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.3-70b-versatile"

        # Ingestion RAG idempotente au premier demarrage de l agent —
        # no-op silencieux si RAG_AVAILABLE est False.
        ensure_documents_ingested()

    def _call_groq_with_retry(self, messages: list):
        """
        Appelle l API Groq avec retry + backoff exponentiel (v2.2). Un
        echec transitoire (rate limit 429, timeout reseau, erreur 5xx
        cote Groq) ne fait plus echouer tout le tour de conversation des
        la premiere tentative — utile en demo live ou l API peut avoir
        des latences ponctuelles. Apres MAX_GROQ_RETRIES echecs, l
        exception est re-levee pour etre geree par l appelant (qui
        renvoie une reponse d erreur propre a l utilisateur).
        """
        last_exception = None
        for attempt in range(MAX_GROQ_RETRIES + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFS,
                    tool_choice="auto",
                    max_tokens=2048,
                    temperature=0.1,
                )
            except Exception as e:
                last_exception = e
                if attempt < MAX_GROQ_RETRIES:
                    wait_s = GROQ_RETRY_BASE_SEC * (attempt + 1)
                    logging.warning(
                        f"Groq API error (tentative {attempt + 1}/{MAX_GROQ_RETRIES + 1}): {e} "
                        f"— nouvel essai dans {wait_s:.1f}s"
                    )
                    time.sleep(wait_s)
                else:
                    logging.error(f"Groq API error apres {MAX_GROQ_RETRIES + 1} tentatives: {e}")
        raise last_exception

    def _degraded_mode_response(self, user_message: str):
        """
        Reponse de secours quand Groq reste indisponible apres tous les
        retries (voir _call_groq_with_retry). Pas d IA ici : un routeur par
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
                    response = self._call_groq_with_retry(messages)
                except Exception as e:
                    logging.error(f"Groq API error (definitif apres retries): {e}")
                    degraded = self._degraded_mode_response(user_message)
                    if degraded:
                        tool_name, result = degraded
                        duration_ms = int((time.time() - start_time) * 1000)
                        answer = (
                            "⚠️ **Mode degrade** : le service IA (Groq) est temporairement "
                            "indisponible, impossible de generer une reponse redigee. Voici "
                            f"les donnees brutes les plus pertinentes trouvees pour votre "
                            f"question (outil `{tool_name}`) :\n\n```json\n"
                            f"{json.dumps(result, ensure_ascii=False, indent=2, default=str)}\n```"
                        )
                        logging.info(f"Mode degrade active -> {tool_name} (Groq indisponible)")
                        return {
                            "answer":      answer,
                            "tool_calls":  [{"tool": tool_name, "inputs": {}, "result_summary": str(result)[:200]}],
                            "thinking":    thinking_log + [f"[mode degrade] Groq indisponible, routage mots-cles -> {tool_name}"],
                            "tokens_used": 0,
                            "duration_ms": duration_ms,
                        }
                    return self._error_response(str(e), tool_calls_log, thinking_log, start_time)

                msg = response.choices[0].message

                # ── Final answer — no tool calls ──────────────────
                if not msg.tool_calls:
                    answer      = msg.content or "Aucune reponse generee."

                    # v2.3 — filet de securite : si un rapport a ete
                    # genere pendant ce tour et que l URL n apparait pas
                    # deja (mot pour mot) dans la reponse du modele, on
                    # l ajoute nous-memes plutot que de faire confiance
                    # au LLM pour la recopier fidelement.
                    if last_report_url and last_report_url not in answer:
                        answer = f"{answer}\n\n📄 Lien direct du rapport : {last_report_url}"

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

                    fn = self.tools_map.get(tool_name)
                    if fn:
                        try:
                            result = fn(**tool_inputs)
                        except Exception as e:
                            result = {"error": str(e), "tool": tool_name}
                            logging.error(f"Tool error {tool_name}: {e}")
                    else:
                        result = {"error": f"Outil inconnu: {tool_name}"}

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