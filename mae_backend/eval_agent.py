# -*- coding: utf-8 -*-
"""
eval_agent.py — Harnais d'evaluation pour MAEIA.

Avant ce script, la qualite de l'agent etait evaluee en le testant a la
main pendant une conversation ("demo et on voit") -- sans mesure objective,
reproductible, ni chiffree pour le rapport. Ce script repond a ca : un jeu
de ~30 questions couvrant les 15 outils, executees automatiquement contre
l'agent REEL (vrais appels Groq), avec deux mesures objectives :

1. PRECISION DU CHOIX D'OUTIL : pour chaque question, on sait a l'avance
   quel(s) outil(s) l'agent DEVRAIT appeler. On compare aux outils
   reellement appeles.
2. TAUX DE HALLUCINATION (proxy) : on extrait tous les nombres presents
   dans la reponse finale de l'agent, et on verifie que chacun est
   retrouvable (a une tolerance pres) dans au moins un des resultats
   d'outils reellement recus. Un nombre qui n'apparait dans AUCUN resultat
   d'outil est un signal fort de fabrication -- ce n'est pas une preuve
   absolue (l'agent peut legitimement calculer une somme/moyenne a partir
   de plusieurs nombres reels), mais c'est un proxy objectif et reproductible.

COUT EN TOKENS : chaque question declenche un vrai appel Groq (system
prompt + 15 schemas d'outils + question, soit environ 2000-4000 tokens
d'entree par appel selon le nombre d'iterations ReAct). Sur le quota
gratuit Groq (100 000 tokens/JOUR pour llama-3.3-70b-versatile), les 30
questions completes peuvent representer une fraction significative du
quota quotidien -- prevoir de lancer avec --limit N pour un sous-ensemble
si le quota est deja entame par d'autres tests.

Usage :
    cd mae_backend
    python eval_agent.py                 # les 30 questions
    python eval_agent.py --limit 5       # seulement les 5 premieres (test rapide)
    python eval_agent.py --category rag  # seulement une categorie
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import main as main_module
from agent import MAEAgent


# ════════════════════════════════════════════════════════════════
# JEU DE QUESTIONS — 30 cas couvrant les 15 outils + le mode degrade non
# teste ici (voir agent.py DEGRADED_MODE_ROUTES pour ce dernier).
# expected_tools : au moins UN de ces outils doit apparaitre dans les
# tool_calls reels pour que le cas soit compte comme correct (une question
# peut legitimement declencher plusieurs outils pertinents).
# ════════════════════════════════════════════════════════════════
EVAL_QUESTIONS = [
    # --- Portfolio / KPIs generaux ---
    {"id": "kpi_01", "category": "kpi", "question": "Resume-moi le portefeuille MAE en quelques chiffres.", "expected_tools": ["portfolio_summary"]},
    {"id": "kpi_02", "category": "kpi", "question": "Quel est le chiffre d'affaires total actuellement ?", "expected_tools": ["portfolio_summary"]},
    {"id": "kpi_03", "category": "kpi", "question": "Quel est le ratio de sinistralite du portefeuille ?", "expected_tools": ["portfolio_summary", "risk_analysis"]},

    # --- Agences ---
    {"id": "agence_01", "category": "agences", "question": "Quelles sont les 5 meilleures agences par chiffre d'affaires ?", "expected_tools": ["top_agencies"]},
    {"id": "agence_02", "category": "agences", "question": "Compare les performances de toutes les agences entre elles.", "expected_tools": ["compare_agencies"]},
    {"id": "agence_03", "category": "agences", "question": "Quelle est la part de marche de l'agence Sfax Ville ?", "expected_tools": ["top_agencies", "compare_agencies"]},

    # --- Prevision ---
    {"id": "forecast_01", "category": "forecast", "question": "Quelle est la prevision de chiffre d'affaires pour 2026 ?", "expected_tools": ["forecast"]},
    {"id": "forecast_02", "category": "forecast", "question": "Quel CA est prevu pour le mois de juin 2026 ?", "expected_tools": ["forecast"]},
    {"id": "forecast_03", "category": "forecast", "question": "Explique-moi en detail comment est calculee la prevision de decembre.", "expected_tools": ["explain_forecast"]},
    {"id": "forecast_04", "category": "forecast", "question": "Pourquoi la croissance prevue est-elle de 4.7% ?", "expected_tools": ["explain_forecast"]},

    # --- Segmentation ---
    {"id": "segment_01", "category": "segmentation", "question": "Quels sont les segments de clients identifies ?", "expected_tools": ["segments"]},
    {"id": "segment_02", "category": "segmentation", "question": "Decris-moi le profil du segment Premium.", "expected_tools": ["segments"]},

    # --- Risque ---
    {"id": "risque_01", "category": "risque", "question": "Analyse les risques du portefeuille.", "expected_tools": ["risk_analysis"]},
    {"id": "risque_02", "category": "risque", "question": "Combien de clients sont identifies comme a risque ?", "expected_tools": ["risk_analysis"]},
    {"id": "risque_03", "category": "risque", "question": "Donne-moi la liste des clients les plus a risque.", "expected_tools": ["risk_clients"]},
    {"id": "risque_04", "category": "risque", "question": "Quels sont les clients a risque dans l'agence Sfax Ville ?", "expected_tools": ["risk_clients"]},

    # --- Sinistres ---
    {"id": "sinistre_01", "category": "sinistres", "question": "Quelles sont les statistiques sur les sinistres ?", "expected_tools": ["sinistres_stats"]},
    {"id": "sinistre_02", "category": "sinistres", "question": "Combien de sinistres sont actuellement en cours de traitement ?", "expected_tools": ["sinistres_stats"]},

    # --- Branche ---
    {"id": "branche_01", "category": "branche", "question": "Quelle est la repartition du chiffre d'affaires par branche d'assurance ?", "expected_tools": ["branch_analysis"]},
    {"id": "branche_02", "category": "branche", "question": "Quelle branche genere le plus de revenus ?", "expected_tools": ["branch_analysis"]},

    # --- Anomalies ---
    {"id": "anomalie_01", "category": "anomalies", "question": "Y a-t-il des anomalies dans les donnees actuellement ?", "expected_tools": ["detect_anomalies"]},
    {"id": "anomalie_02", "category": "anomalies", "question": "Detecte les sinistres avec des montants aberrants.", "expected_tools": ["detect_anomalies"]},

    # --- Profil clients ---
    {"id": "profil_01", "category": "profil_clients", "question": "Quel est le profil demographique des clients (age, sexe, CSP) ?", "expected_tools": ["profil_clients"]},
    {"id": "profil_02", "category": "profil_clients", "question": "Quelle est la repartition par tranche d'age des assures ?", "expected_tools": ["profil_clients"]},

    # --- RAG documents metier ---
    {"id": "rag_01", "category": "rag", "question": "Quel est le delai pour declarer un sinistre ?", "expected_tools": ["consulter_documents_metier"]},
    {"id": "rag_02", "category": "rag", "question": "Comment fonctionne le systeme de bonus-malus ?", "expected_tools": ["consulter_documents_metier"]},
    {"id": "rag_03", "category": "rag", "question": "Selon la grille tarifaire, comment est calculee la prime pour un taxi ?", "expected_tools": ["consulter_documents_metier"]},
    {"id": "rag_04", "category": "rag", "question": "Que dit le glossaire sur le statut 'EN COURS' d'un sinistre ?", "expected_tools": ["consulter_documents_metier"]},

    # --- Rapports ---
    {"id": "rapport_01", "category": "rapport", "question": "Genere-moi un rapport PDF complet du portefeuille.", "expected_tools": ["generate_report"]},
    {"id": "rapport_02", "category": "rapport", "question": "J'ai besoin d'un export Excel avec juste les previsions.", "expected_tools": ["generate_report"]},

    # --- Monitoring ---
    {"id": "monitoring_01", "category": "monitoring", "question": "Comment se comporte l'agent en ce moment (latence, erreurs) ?", "expected_tools": ["agent_monitoring"]},
    {"id": "monitoring_02", "category": "monitoring", "question": "Est-ce qu'il y a une derive dans les donnees en ce moment ?", "expected_tools": ["agent_monitoring"]},
]


# ════════════════════════════════════════════════════════════════
# EXTRACTION DE NOMBRES — pour le proxy de hallucination
# ════════════════════════════════════════════════════════════════
_ISO_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")
_NUMBER_RE = re.compile(r"-?\d[\d\s]*(?:[.,]\d+)?")


def _extract_numbers(text: str):
    """Extrait les nombres d'un texte (formats FR '1 234,56' et EN '1234.56'),
    normalises en float. Deux filtres anti-bruit :
    - les dates/horodatages ISO (ex: "2026-08-10T14:32:01") sont retires
      AVANT l'extraction, sinon "2026" ou "10" seraient lus comme des
      valeurs metier fabriquees ;
    - les nombres < 100 sont ignores (mois, pourcentages, ids courts --
      trop de bruit pour etre un signal de hallucination utile).
    """
    text = _ISO_DATETIME_RE.sub(" ", text or "")
    numbers = []
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace(" ", "").replace(",", ".")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if abs(val) >= 100:
            numbers.append(val)
    return numbers


def _collect_numbers_from_tool_results(tool_calls_log, tools_map):
    """Rassemble tous les nombres presents dans les resultats REELS et
    COMPLETS des outils appeles pendant ce tour.

    IMPORTANT : tool_calls_log ne contient que result_summary, tronque a
    200 caracteres cote agent.py (necessaire pour ne pas gonfler le cout
    en tokens de la conversation reelle, mais inutilisable tel quel ici --
    un resultat tronque ferait passer des chiffres reels pour "non traces"
    par simple decoupage en plein milieu). On ré-invoque donc chaque outil
    avec les memes arguments pour recuperer son resultat COMPLET, sans
    aucun appel LLM (rejouer un outil est gratuit et deterministe, a la
    difference de rejouer l agent).
    """
    numbers = []
    for tc in tool_calls_log:
        fn = tools_map.get(tc.get("tool"))
        if not fn:
            continue
        try:
            full_result = fn(**(tc.get("inputs") or {}))
        except Exception:
            full_result = tc.get("result_summary", "")
        numbers.extend(_extract_numbers(json.dumps(full_result, ensure_ascii=False, default=str)))
    return numbers


def _check_hallucination(answer: str, tool_calls_log, tools_map):
    """Retourne (nb_nombres_dans_reponse, nb_non_traces, liste_non_traces).
    Tolerance relative de 1% pour absorber les arrondis d'affichage."""
    answer_numbers = _extract_numbers(answer)
    tool_numbers = _collect_numbers_from_tool_results(tool_calls_log, tools_map)

    untraced = []
    for n in answer_numbers:
        found = any(abs(n - t) <= max(abs(t), 1) * 0.01 for t in tool_numbers)
        if not found:
            untraced.append(n)
    return len(answer_numbers), len(untraced), untraced


# ════════════════════════════════════════════════════════════════
# EXECUTION
# ════════════════════════════════════════════════════════════════
def run_eval(questions, verbose=True):
    main_module.seed_data()
    agent = MAEAgent(tools_map=main_module.TOOLS_MAP)

    results = []
    for i, case in enumerate(questions):
        if verbose:
            print(f"[{i+1}/{len(questions)}] {case['id']} — {case['question']}")

        t0 = time.time()
        try:
            r = agent.run(case["question"])
        except Exception as e:
            r = {"answer": f"[EXCEPTION] {e}", "tool_calls": [], "tokens_used": 0, "duration_ms": 0}
        elapsed = time.time() - t0

        tools_used = [tc["tool"] for tc in r.get("tool_calls", [])]
        tool_ok = any(t in tools_used for t in case["expected_tools"])
        n_nums, n_untraced, untraced_vals = _check_hallucination(
            r.get("answer", ""), r.get("tool_calls", []), main_module.TOOLS_MAP
        )

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_tools": case["expected_tools"],
            "actual_tools": tools_used,
            "tool_call_correct": tool_ok,
            "answer_numbers_total": n_nums,
            "answer_numbers_untraced": n_untraced,
            "untraced_values": untraced_vals,
            "possible_hallucination": n_untraced > 0,
            "tokens_used": r.get("tokens_used", 0),
            "duration_ms": r.get("duration_ms", 0),
            "wall_time_s": round(elapsed, 2),
            "answer_preview": (r.get("answer") or "")[:200],
        }
        results.append(result)

        if verbose:
            status = "OK" if tool_ok else "ECHEC (mauvais outil)"
            halluc = f", {n_untraced} nombre(s) non traces" if n_untraced else ""
            print(f"    -> outils attendus={case['expected_tools']} obtenus={tools_used} [{status}]{halluc}")

    return results


def summarize(results):
    n = len(results)
    tool_acc = sum(r["tool_call_correct"] for r in results) / n * 100
    halluc_rate = sum(r["possible_hallucination"] for r in results) / n * 100
    avg_tokens = sum(r["tokens_used"] for r in results) / n
    avg_latency = sum(r["duration_ms"] for r in results) / n

    by_cat = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += r["tool_call_correct"]

    print("\n" + "=" * 60)
    print("RESUME DE L'EVALUATION")
    print("=" * 60)
    print(f"Questions evaluees        : {n}")
    print(f"Precision choix d'outil   : {tool_acc:.1f}%")
    print(f"Taux de hallucination*    : {halluc_rate:.1f}%  (*proxy, voir docstring)")
    print(f"Tokens moyens / question  : {avg_tokens:.0f}")
    print(f"Latence moyenne           : {avg_latency:.0f} ms")
    print("\nPrecision par categorie :")
    for cat, c in sorted(by_cat.items()):
        print(f"  {cat:15s} {c['ok']}/{c['n']}  ({c['ok']/c['n']*100:.0f}%)")

    failures = [r for r in results if not r["tool_call_correct"]]
    if failures:
        print("\nEchecs de choix d'outil :")
        for r in failures:
            print(f"  [{r['id']}] \"{r['question']}\" -> attendu {r['expected_tools']}, obtenu {r['actual_tools']}")

    halluc = [r for r in results if r["possible_hallucination"]]
    if halluc:
        print("\nReponses avec nombre(s) non traces (hallucination potentielle) :")
        for r in halluc:
            print(f"  [{r['id']}] valeurs non tracees : {r['untraced_values']}")

    return {
        "n_questions": n,
        "tool_call_accuracy_pct": round(tool_acc, 1),
        "hallucination_rate_pct": round(halluc_rate, 1),
        "avg_tokens": round(avg_tokens, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "by_category": {c: {"correct": v["ok"], "total": v["n"]} for c, v in by_cat.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalue MAEIA sur un jeu de questions types.")
    parser.add_argument("--limit", type=int, default=None, help="N'evaluer que les N premieres questions")
    parser.add_argument("--category", type=str, default=None, help="N'evaluer qu'une categorie precise")
    args = parser.parse_args()

    questions = EVAL_QUESTIONS
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[:args.limit]

    if not questions:
        print("Aucune question ne correspond aux filtres fournis.")
        raise SystemExit(1)

    print(f"Evaluation de {len(questions)} question(s)...\n")
    results = run_eval(questions)
    summary = summarize(results)

    os.makedirs("eval_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("eval_results", f"eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResultats detailles sauvegardes dans {out_path}")
