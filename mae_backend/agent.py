# -*- coding: utf-8 -*-
"""
agent.py — MAEIA · Agent ReAct MAE Assurances
Modele : Llama 3.3 70B via Groq (gratuit, rapide, tool calling)
Architecture : Reasoning -> Tool Selection -> Execution -> Observation -> Reponse
MLflow tracke chaque interaction
"""

import json
import os
import time
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
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Tu es MAEIA, l'assistant IA avance de la MAE (Mutuelle Automobile des Enseignants de Tunisie).
Tu analyses en temps reel le portefeuille d assurance auto pour la direction generale.

Regles :
- Reponds TOUJOURS en francais, ton professionnel et analytique
- Utilise TOUJOURS les outils pour acceder aux donnees reelles AVANT de repondre
- Ne devine jamais des chiffres — utilise toujours un outil
- Si plusieurs outils sont pertinents, appelle-les tous avant de synthetiser
- Structure ta reponse : Analyse -> Chiffres cles -> Recommandations
- Les donnees sont mises a jour en temps reel toutes les 5 secondes
- IMPORTANT FORMAT MONNAIE: Affiche TOUJOURS les montants en TND avec le format suivant: 1 234 567,89 TND (espace comme separateur de milliers, virgule pour les decimales). Exemple: 1 887 480,00 TND et NON pas 1887480 ou 1,887,480
- Tu peux detecter des anomalies, comparer des agences, predire le CA, analyser les risques
- Sois precis, concret et actionnable dans tes recommandations"""

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
            "description": "Top N agences par chiffre d affaires avec parts de marche et classement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Nombre d agences a retourner (defaut 5)"},
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
            "description": "Analyse complete des risques du portefeuille : ratio sinistralite, clients a risque, agences sous-performantes, recommandations prioritaires.",
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
]

# ════════════════════════════════════════════════════════════════
# AGENT CLASS
# ════════════════════════════════════════════════════════════════
class MAEAgent:
    """
    Agent ReAct pour la MAE — Groq + Llama 3.3 70B.
    tools_map injecte depuis main.py (pas d import circulaire).
    """

    def __init__(self, tools_map: dict):
        self.tools_map = tools_map
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY non configuree. "
                "Creez un compte gratuit sur console.groq.com et ajoutez la cle dans .env"
            )
        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.3-70b-versatile"

    def run(self, user_message: str, history: list = None) -> dict:
        if history is None:
            history = []

        start_time     = time.time()
        tool_calls_log = []
        thinking_log   = []
        max_iterations = 8

        # Build message history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
            mlflow_active = True
        except Exception:
            pass

        try:
            # ── ReAct loop ────────────────────────────────────────
            for iteration in range(max_iterations):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_DEFS,
                        tool_choice="auto",
                        max_tokens=2048,
                        temperature=0.1,
                    )
                except Exception as e:
                    logging.error(f"Groq API error: {e}")
                    return self._error_response(str(e), tool_calls_log, thinking_log, start_time)

                msg = response.choices[0].message

                # ── Final answer — no tool calls ──────────────────
                if not msg.tool_calls:
                    answer      = msg.content or "Aucune reponse generee."
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

                    tool_calls_log.append({
                        "tool":           tool_name,
                        "inputs":         tool_inputs,
                        "result_summary": str(result)[:200],
                    })

                    # Add tool result to messages
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      json.dumps(result, ensure_ascii=False, default=str),
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