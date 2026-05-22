# -*- coding: utf-8 -*-
"""
main.py — MAE Assurances · FastAPI Backend v3.0
Real-time simulator + Analytics endpoints + Agent ReAct (via agent.py)
Modele agent : Llama 3.3 70B via Groq (gratuit)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import threading
import time
import random
from datetime import datetime, timedelta
import mlflow

from agent import MAEAgent

from dotenv import load_dotenv
load_dotenv()

# ════════════════════════════════════════════════════════════════
# APP INIT
# ════════════════════════════════════════════════════════════════
app = FastAPI(title="MAE Intelligence API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("mlruns")

# ════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════
AGENCES  = ["Sfax Ville","Tunis Centre","Sousse","Nabeul","Bizerte",
            "Gabes","Sfax Nord","Monastir","Ariana","Gafsa",
            "Kairouan","Medenine","Jendouba","Beja","Tozeur"]
BRANCHES = ["Tourisme","Taxi","Transport_Prive","Louage","2_Roues","Autre"]
REGIONS  = ["Grand Tunis","Nord","Centre","Sud","Sahel"]
CSP_LIST = ["Salarie","Fonctionnaire","Commercant","Artisan","Profession liberale","Retraite"]
MOIS_LABELS = ["Jan","Fev","Mar","Avr","Mai","Jun","Jul","Aou","Sep","Oct","Nov","Dec"]

_bw = [0.857,0.06,0.04,0.02,0.015,0.008]; _bs = sum(_bw)
BRANCH_WEIGHTS = [x/_bs for x in _bw]
_aw = [0.12,0.11,0.09,0.08,0.07,0.07,0.06,0.06,0.06,0.05,0.05,0.05,0.05,0.05,0.04]; _as = sum(_aw)
AGENCE_WEIGHTS = [x/_as for x in _aw]

# ════════════════════════════════════════════════════════════════
# ETAT GLOBAL
# ════════════════════════════════════════════════════════════════
state = {
    "contrats":        [],
    "sinistres":       [],
    "last_update":     datetime.now().isoformat(),
    "total_generated": 0,
    "lock":            threading.Lock(),
}

# ════════════════════════════════════════════════════════════════
# SEED DATA
# ════════════════════════════════════════════════════════════════
def seed_data():
    prod_path = "../processed_data/Production_Cleaned.csv"
    sin_path  = "../processed_data/Sinistres_Cleaned.csv"
    np.random.seed(42)

    if os.path.exists(prod_path):
        try:
            df = pd.read_csv(prod_path, nrows=5000)
            records = df.to_dict("records")
            for r in records:
                r["AGENCE"]      = str(r.get("AGENCE", random.choice(AGENCES)))
                r["BRANCHE"]     = str(r.get("BRANCHE", np.random.choice(BRANCHES, p=BRANCH_WEIGHTS)))
                r["PRIME_NETTE"] = abs(float(r.get("PRIME_NETTE", 500) or 500))
                r["MOIS"]        = pd.to_datetime(r.get("DEBUT_PERI"), errors="coerce").month \
                                   if r.get("DEBUT_PERI") else random.randint(1,12)
                r["AGE"]         = r.get("AGE", random.randint(22,72))
                r["SEXE"]        = str(r.get("SEXE", random.choice(["M","F"])))
                r["Region"]      = str(r.get("Region", random.choice(REGIONS)))
                r["CSP"]         = str(r.get("CSP", random.choice(CSP_LIST)))
                r["BONUS_MALUS"] = int(float(r.get("BONUS_MALUS", 0) or 0))
                r["timestamp"]   = (datetime.now() - timedelta(days=random.randint(0,364))).isoformat()
            state["contrats"] = records
            print(f"✅ {len(records)} contrats reels charges")
        except Exception as e:
            print(f"⚠️  CSV prod ({e}) -> donnees synthetiques")
            _gen_synthetic()
    else:
        _gen_synthetic()

    if os.path.exists(sin_path):
        try:
            df = pd.read_csv(sin_path, nrows=2000)
            records = df.to_dict("records")
            for r in records:
                r["AGENCE"]      = str(r.get("AGENCE", random.choice(AGENCES)))
                r["REGLEMENTS"]  = abs(float(r.get("REGLEMENTS", 1000) or 1000))
                r["TRANSACTION"] = str(r.get("TRANSACTION", random.choice(["REGLEMENT","EN COURS","EXPERTISE"])))
                r["MOIS"]        = pd.to_datetime(r.get("DATE_ACCIDENT"), errors="coerce").month \
                                   if r.get("DATE_ACCIDENT") else random.randint(1,12)
                r["timestamp"]   = (datetime.now() - timedelta(days=random.randint(0,364))).isoformat()
            state["sinistres"] = records
            print(f"✅ {len(records)} sinistres reels charges")
        except Exception as e:
            print(f"⚠️  CSV sin ({e}) -> synthetiques")
            _gen_synthetic_sin()
    else:
        _gen_synthetic_sin()

    state["total_generated"] = len(state["contrats"])


def _gen_synthetic(n=5000):
    records = []
    for i in range(n):
        mois = random.randint(1,12)
        records.append({
            "N_CLIENT":    i+1,
            "N_POLICE":    random.randint(100000,999999),
            "AGENCE":      np.random.choice(AGENCES, p=AGENCE_WEIGHTS),
            "BRANCHE":     np.random.choice(BRANCHES, p=BRANCH_WEIGHTS),
            "PRIME_NETTE": float(np.clip(np.random.lognormal(7.5,1.2), 200, 80000)),
            "BONUS_MALUS": int(np.random.choice(range(14), p=[.05,.08,.10,.12,.13,.13,.12,.10,.08,.06,.05,.03,.03,.02])),
            "SEXE":        np.random.choice(["M","F"], p=[0.75,0.25]),
            "CSP":         np.random.choice(CSP_LIST, p=[0.32,0.28,0.18,0.10,0.08,0.04]),
            "Region":      np.random.choice(REGIONS, p=[0.35,0.15,0.20,0.15,0.15]),
            "AGE":         int(np.clip(np.random.normal(47,11), 18, 80)),
            "MOIS":        mois,
            "TRIMESTRE":   (mois-1)//3+1,
            "timestamp":   (datetime.now()-timedelta(days=random.randint(0,364))).isoformat(),
        })
    state["contrats"] = records


def _gen_synthetic_sin(n=1000):
    records = []
    for i in range(n):
        records.append({
            "N_CLIENT":    random.randint(1,5000),
            "AGENCE":      np.random.choice(AGENCES, p=AGENCE_WEIGHTS),
            "REGLEMENTS":  float(np.clip(np.random.lognormal(7.8,1.3), 500, 200000)),
            "TRANSACTION": np.random.choice(["REGLEMENT","EN COURS","EXPERTISE"], p=[0.55,0.30,0.15]),
            "MOIS":        random.randint(1,12),
            "timestamp":   (datetime.now()-timedelta(days=random.randint(0,364))).isoformat(),
        })
    state["sinistres"] = records


# ════════════════════════════════════════════════════════════════
# SIMULATEUR TEMPS REEL
# ════════════════════════════════════════════════════════════════
def simulator(interval_seconds: int = 5):
    while True:
        time.sleep(interval_seconds)
        with state["lock"]:
            mois = datetime.now().month
            new_c = {
                "N_CLIENT":    state["total_generated"] + 1,
                "N_POLICE":    random.randint(100000,999999),
                "AGENCE":      np.random.choice(AGENCES, p=AGENCE_WEIGHTS),
                "BRANCHE":     np.random.choice(BRANCHES, p=BRANCH_WEIGHTS),
                "PRIME_NETTE": float(np.clip(np.random.lognormal(7.5,1.2), 200, 80000)),
                "BONUS_MALUS": int(np.random.choice(range(14))),
                "SEXE":        np.random.choice(["M","F"], p=[0.75,0.25]),
                "CSP":         np.random.choice(CSP_LIST),
                "Region":      np.random.choice(REGIONS),
                "AGE":         int(np.clip(np.random.normal(47,11), 18, 80)),
                "MOIS":        mois,
                "TRIMESTRE":   (mois-1)//3+1,
                "timestamp":   datetime.now().isoformat(),
            }
            state["contrats"].append(new_c)
            state["total_generated"] += 1
            if random.random() < 0.30:
                state["sinistres"].append({
                    "N_CLIENT":    new_c["N_CLIENT"],
                    "AGENCE":      new_c["AGENCE"],
                    "REGLEMENTS":  float(np.clip(np.random.lognormal(7.8,1.3), 500, 200000)),
                    "TRANSACTION": np.random.choice(["REGLEMENT","EN COURS","EXPERTISE"], p=[0.55,0.30,0.15]),
                    "MOIS":        mois,
                    "timestamp":   datetime.now().isoformat(),
                })
            state["last_update"] = datetime.now().isoformat()


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def get_df():
    with state["lock"]:
        return pd.DataFrame(list(state["contrats"])), pd.DataFrame(list(state["sinistres"]))


def apply_filters(df, agence=None, region=None, branche=None, mois=None):
    if agence  and agence  != "all" and "AGENCE"  in df.columns: df = df[df["AGENCE"]  == agence]
    if region  and region  != "all" and "Region"  in df.columns: df = df[df["Region"]  == region]
    if branche and branche != "all" and "BRANCHE" in df.columns: df = df[df["BRANCHE"] == branche]
    if mois    and mois    != 0     and "MOIS"    in df.columns: df = df[df["MOIS"]    == mois]
    return df


# ════════════════════════════════════════════════════════════════
# TOOLS METIER
# ════════════════════════════════════════════════════════════════
def tool_portfolio_summary(agence=None, region=None, branche=None):
    prod, sin = get_df()
    prod = apply_filters(prod, agence, region, branche)
    ca   = prod["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod.columns else 0
    st   = sin["REGLEMENTS"].sum()   if "REGLEMENTS"  in sin.columns  else 0
    top_ag = prod.groupby("AGENCE")["PRIME_NETTE"].sum().idxmax() if len(prod) > 0 else "N/A"
    return {
        "ca_total":           round(ca, 2),
        "nb_clients":         int(prod["N_CLIENT"].nunique()) if "N_CLIENT" in prod.columns else len(prod),
        "nb_contrats":        len(prod),
        "sinistres_total":    round(st, 2),
        "ratio_sinistralite": round(st / ca * 100, 2) if ca > 0 else 0,
        "top_agence":         str(top_ag),
        "derniere_maj":       state["last_update"],
        "total_generes":      state["total_generated"],
    }


def tool_top_agencies(n=5):
    prod, _ = get_df()
    top      = prod.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values(ascending=False).head(n)
    ca_total = prod["PRIME_NETTE"].sum()
    return [{"agence": str(a), "ca": round(v,2), "part_pct": round(v/ca_total*100,1)}
            for a,v in top.items()]


def tool_forecast(mois_num=None):
    base     = 1_680_000
    trend    = np.linspace(0, base * 0.047, 12)
    seasonal = [0.92,0.88,1.02,1.05,1.08,1.10,1.07,1.05,1.03,1.00,0.97,1.12]
    months   = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                "Juillet","Aout","Septembre","Octobre","Novembre","Decembre"]
    if mois_num and 1 <= mois_num <= 12:
        i  = mois_num - 1
        ca = (base + trend[i]) * seasonal[i]
        return {"mois": months[i], "ca_prev": round(ca,2),
                "ci_low": round(ca*0.93,2), "ci_high": round(ca*1.07,2)}
    all_ca = [(base+trend[i])*seasonal[i] for i in range(12)]
    return {
        "total_2026":     round(sum(all_ca), 2),
        "croissance_pct": 4.7,
        "mois_pic":       months[int(np.argmax(all_ca))],
        "detail":         [{"mois": months[i], "ca": round(all_ca[i],2)} for i in range(12)],
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


def tool_risk_analysis():
    prod, sin = get_df()
    ca = prod["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod.columns else 1
    st = sin["REGLEMENTS"].sum()   if "REGLEMENTS"  in sin.columns  else 0
    sous_perf = prod.groupby("AGENCE")["PRIME_NETTE"].sum().sort_values().head(3).index.tolist() \
                if "AGENCE" in prod.columns else []
    return {
        "ratio_sinistralite":  round(st / ca * 100, 2),
        "clients_a_risque":    33819,
        "part_risque_pct":     46.5,
        "agences_sous_perf":   [str(a) for a in sous_perf],
        "recommandations": [
            "Reviser la politique tarifaire pour le segment A Risque",
            "Renforcer les controles sur la branche Taxi",
            "Campagne de fidelisation pour les Occasionnels",
            "Programme VIP pour les clients Premium",
        ],
    }


def tool_sinistres_stats():
    _, sin = get_df()
    return {
        "total_sinistres": len(sin),
        "montant_total":   round(sin["REGLEMENTS"].sum(), 2) if "REGLEMENTS" in sin.columns else 0,
        "montant_moyen":   round(sin["REGLEMENTS"].mean(), 2) if len(sin) > 0 and "REGLEMENTS" in sin.columns else 0,
        "en_cours":        int(len(sin[sin["TRANSACTION"] == "EN COURS"])) if "TRANSACTION" in sin.columns else 0,
        "pic_mois":        int(sin.groupby("MOIS")["REGLEMENTS"].sum().idxmax()) if "MOIS" in sin.columns and len(sin) > 0 else 0,
    }


def tool_branch_analysis():
    prod, _ = get_df()
    ca_total = prod["PRIME_NETTE"].sum()
    by_br    = prod.groupby("BRANCHE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    return [{"branche": str(b), "ca": round(v,2), "part_pct": round(v/ca_total*100,1)}
            for b,v in by_br.items()]


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
        ca_cl   = prod.groupby("N_CLIENT")["PRIME_NETTE"].sum()
        st_cl   = sin.groupby("N_CLIENT")["REGLEMENTS"].sum()
        ratio_df = pd.DataFrame({"CA": ca_cl, "ST": st_cl}).dropna()
        ratio_df["ratio"] = ratio_df["ST"] / ratio_df["CA"] * 100
        haut    = ratio_df[ratio_df["ratio"] > 150]
        anomalies.append({
            "type":        "Clients ratio S/P > 150%",
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
    sexe = prod["SEXE"].value_counts().to_dict()        if "SEXE"        in prod.columns else {}
    csp  = prod["CSP"].value_counts().head(6).to_dict() if "CSP"         in prod.columns else {}
    bm   = prod["BONUS_MALUS"].value_counts().sort_index().to_dict() if "BONUS_MALUS" in prod.columns else {}
    ages = prod["AGE"].dropna().tolist()                if "AGE"         in prod.columns else []
    age_bins = {"18-30": 0, "31-45": 0, "46-60": 0, "61-80": 0}
    for a in ages:
        if   a <= 30: age_bins["18-30"] += 1
        elif a <= 45: age_bins["31-45"] += 1
        elif a <= 60: age_bins["46-60"] += 1
        else:         age_bins["61-80"] += 1
    return {"sexe": sexe, "csp": csp, "bonus_malus": bm, "age_bins": age_bins}


TOOLS_MAP = {
    "portfolio_summary": tool_portfolio_summary,
    "top_agencies":      tool_top_agencies,
    "forecast":          tool_forecast,
    "segments":          tool_segments,
    "risk_analysis":     tool_risk_analysis,
    "sinistres_stats":   tool_sinistres_stats,
    "branch_analysis":   tool_branch_analysis,
    "compare_agencies":  tool_compare_agencies,
    "detect_anomalies":  tool_detect_anomalies,
    "profil_clients":    tool_profil_clients,
}

_agent: MAEAgent | None = None


# ════════════════════════════════════════════════════════════════
# ENDPOINTS REST
# ════════════════════════════════════════════════════════════════
@app.get("/api/status")
def status():
    return {
        "status":          "online",
        "last_update":     state["last_update"],
        "total_contrats":  len(state["contrats"]),
        "total_sinistres": len(state["sinistres"]),
        "total_generated": state["total_generated"],
    }


@app.get("/api/kpis")
def kpis(agence: str = "all", region: str = "all", branche: str = "all", mois: int = 0):
    prod, sin = get_df()
    prod  = apply_filters(prod, agence, region, branche, mois)
    sin_f = apply_filters(sin,  agence, None,   None,    mois)
    ca    = prod["PRIME_NETTE"].sum() if "PRIME_NETTE" in prod.columns else 0
    st    = sin_f["REGLEMENTS"].sum() if "REGLEMENTS"  in sin_f.columns else 0
    top_ag = prod.groupby("AGENCE")["PRIME_NETTE"].sum().idxmax() if len(prod) > 0 else "N/A"
    top_ca = prod.groupby("AGENCE")["PRIME_NETTE"].sum().max()    if len(prod) > 0 else 0
    return {
        "ca_total":        round(ca, 2),
        "nb_clients":      int(prod["N_CLIENT"].nunique()) if "N_CLIENT" in prod.columns else len(prod),
        "nb_contrats":     len(prod),
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
    return [{"agence": str(a), "ca": round(v,2), "part": round(v/total*100,1) if total>0 else 0}
            for a,v in data.items()]


@app.get("/api/ca-by-branche")
def ca_by_branche(agence: str = "all", region: str = "all", mois: int = 0):
    prod, _ = get_df()
    prod  = apply_filters(prod, agence, region, None, mois)
    data  = prod.groupby("BRANCHE")["PRIME_NETTE"].sum().sort_values(ascending=False)
    total = data.sum()
    return [{"branche": str(b), "ca": round(v,2), "part": round(v/total*100,1) if total>0 else 0}
            for b,v in data.items()]


@app.get("/api/ca-by-month")
def ca_by_month(agence: str = "all", region: str = "all", branche: str = "all"):
    prod, sin = get_df()
    prod    = apply_filters(prod, agence, region, branche)
    prime_m = prod.groupby("MOIS")["PRIME_NETTE"].sum() if "MOIS" in prod.columns else pd.Series()
    sin_m   = sin.groupby("MOIS")["REGLEMENTS"].sum()   if "MOIS" in sin.columns  else pd.Series()
    return [{"mois": MOIS_LABELS[m-1], "mois_num": m,
             "primes":    round(float(prime_m.get(m, 0)), 2),
             "sinistres": round(float(sin_m.get(m, 0)),   2)}
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
    base     = 1_680_000
    trend    = np.linspace(0, base * 0.047, 12)
    seasonal = [0.92,0.88,1.02,1.05,1.08,1.10,1.07,1.05,1.03,1.00,0.97,1.12]
    prod, _  = get_df()
    hist_m   = prod.groupby("MOIS")["PRIME_NETTE"].sum() if "MOIS" in prod.columns else pd.Series()
    return [{
        "mois":     MOIS_LABELS[i],
        "mois_num": i+1,
        "ca_prev":  round((base+trend[i])*seasonal[i], 2),
        "ci_low":   round((base+trend[i])*seasonal[i]*0.93, 2),
        "ci_high":  round((base+trend[i])*seasonal[i]*1.07, 2),
        "ca_reel":  round(float(hist_m.get(i+1, 0)), 2),
    } for i in range(12)]


@app.get("/api/segments")
def segments_endpoint():
    return [
        {"segment":"Premium",    "count":7751,  "ca_moyen":729000, "color":"#00e5b4",
         "bm_moyen":2.1,"nb_contrats_moyen":12.3,"risque":"Faible"},
        {"segment":"Standard",   "count":11771, "ca_moyen":368000, "color":"#3b82f6",
         "bm_moyen":7.0,"nb_contrats_moyen":5.8, "risque":"Modere"},
        {"segment":"Occasionnel","count":19333, "ca_moyen":190000, "color":"#f59e0b",
         "bm_moyen":5.2,"nb_contrats_moyen":2.4, "risque":"Modere"},
        {"segment":"A Risque",   "count":33819, "ca_moyen":172000, "color":"#f43f5e",
         "bm_moyen":10.8,"nb_contrats_moyen":1.9,"risque":"Eleve"},
    ]


@app.get("/api/filters")
def filters_endpoint():
    prod, _ = get_df()
    return {
        "agences":  ["all"] + sorted(prod["AGENCE"].dropna().unique().tolist()) if "AGENCE" in prod.columns else ["all"],
        "regions":  ["all"] + sorted(prod["Region"].dropna().unique().tolist()) if "Region" in prod.columns else ["all"],
        "branches": ["all"] + sorted(prod["BRANCHE"].dropna().unique().tolist()) if "BRANCHE" in prod.columns else ["all"],
    }


@app.get("/api/live-feed")
def live_feed(limit: int = 10):
    with state["lock"]:
        recent = sorted(state["contrats"][-50:], key=lambda x: x.get("timestamp",""), reverse=True)[:limit]
    return recent


@app.get("/api/anomalies")
def anomalies_endpoint():
    return tool_detect_anomalies()


@app.get("/api/compare-agencies")
def compare_agencies_endpoint():
    return tool_compare_agencies()


# ════════════════════════════════════════════════════════════════
# AGENT ENDPOINT
# ════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str
    history: list = []


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


# ════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════
@app.on_event("startup")
def startup():
    seed_data()
    threading.Thread(target=simulator, args=(5,), daemon=True).start()
    print("✅ MAE Intelligence API v3.0 — simulateur actif toutes les 5s")