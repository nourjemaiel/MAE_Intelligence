# -*- coding: utf-8 -*-
"""
test_main.py — Tests unitaires cibles pour main.py (MAE Intelligence API).

Portee volontairement restreinte a quelques fonctions a fort risque de
regression SILENCIEUSE plutot qu'une couverture exhaustive :
- normalize_agence()/normalize_branche() sont appelees a l'ingestion et
  affichees telles quelles dans les filtres/rapports -- une regression n'y
  leve aucune exception, elle degrade juste l'affichage sans avertissement.
- tool_forecast() encode l'invariant central de la prevision CA (base *
  croissance) -- un refactoring qui le casse doit etre detecte immediatement,
  pas decouvert a l'oral devant le jury.

Lancer depuis mae_backend/ : pytest test_main.py -v
"""
import math

import pandas as pd

import main


# ════════════════════════════════════════════════════════════════
# tool_forecast() — invariant central : total_2026 == base * 1.047
# ════════════════════════════════════════════════════════════════
def test_tool_forecast_total_matches_growth_formula():
    main.state["ca_total_reel"] = 1_000_000.0
    result = main.tool_forecast()
    expected = 1_000_000.0 * 1.047
    # Tolerance liee aux arrondis a 2 decimales appliques sur chaque mois
    # avant sommation (round(ca, 2) x12), pas un simple round() global.
    assert math.isclose(result["total_2026"], expected, rel_tol=1e-4)


def test_tool_forecast_single_month_within_annual_total():
    main.state["ca_total_reel"] = 1_000_000.0
    annuel  = main.tool_forecast()
    mensuel = main.tool_forecast(mois_num=6)
    assert 0 < mensuel["ca_prev"] < annuel["total_2026"]


def test_tool_forecast_confidence_interval_widens_with_horizon():
    # Verifie que l'intervalle de confiance dynamique (voir ci_width_for_month
    # dans main.py) s'elargit bien avec l'horizon, et n'est pas revenu a un
    # pourcentage fixe par accident lors d'un futur refactoring.
    main.state["ca_total_reel"] = 1_000_000.0
    janvier  = main.tool_forecast(mois_num=1)
    decembre = main.tool_forecast(mois_num=12)
    assert janvier["ci_width_pct"] < decembre["ci_width_pct"]


# ════════════════════════════════════════════════════════════════
# normalize_branche() — mapping code MAE -> libelle metier
# ════════════════════════════════════════════════════════════════
def test_normalize_branche_maps_known_codes():
    assert main.normalize_branche("11") == "Tourisme"
    assert main.normalize_branche("41") == "Taxi"


def test_normalize_branche_strips_parasitic_dot_zero():
    # Artefact frequent de lecture pandas (codes numeriques lus comme float
    # puis convertis en string -> "11.0" au lieu de "11")
    assert main.normalize_branche("11.0") == main.normalize_branche("11") == "Tourisme"


def test_normalize_branche_passthrough_for_unknown_code():
    assert main.normalize_branche("999") == "999"


# ════════════════════════════════════════════════════════════════
# normalize_agence() — deduplication accents/casse
# ════════════════════════════════════════════════════════════════
def test_normalize_agence_dedupes_accents_and_case():
    canonical = main.normalize_agence("Beja")
    assert main.normalize_agence("Béja") == canonical
    assert main.normalize_agence("BEJA") == canonical
    assert main.normalize_agence("béja") == canonical


def test_normalize_agence_applies_known_overrides():
    assert main.normalize_agence("sfax")  == "Sfax Ville"
    assert main.normalize_agence("tunis") == "Tunis Centre"


def test_normalize_agence_passthrough_for_unknown_agence():
    assert main.normalize_agence("Nouvelle Agence XYZ") == "Nouvelle Agence XYZ"


# ════════════════════════════════════════════════════════════════
# tool_risk_clients() — liste nominative sur le portefeuille COMPLET
# (get_full_df(), pas get_df()/l'echantillon temps reel -- voir le
# commentaire dans main.py sur le bug de ratios absurdes que ça evitait)
# ════════════════════════════════════════════════════════════════
def _set_full_portfolio(prod_rows, sin_rows):
    main.state["prod_full_df"] = pd.DataFrame(prod_rows)
    main.state["sin_full_df"]  = pd.DataFrame(sin_rows)


def test_risk_clients_flags_ratio_above_threshold_only():
    _set_full_portfolio(
        prod_rows=[
            {"N_CLIENT": "1", "AGENCE": "Sfax Ville", "PRIME_NETTE": 1000.0},
            {"N_CLIENT": "2", "AGENCE": "Sfax Ville", "PRIME_NETTE": 1000.0},
        ],
        sin_rows=[
            {"N_CLIENT": "1", "AGENCE": "Sfax Ville", "REGLEMENTS": 500.0},   # ratio 50% -- pas a risque
            {"N_CLIENT": "2", "AGENCE": "Sfax Ville", "REGLEMENTS": 2000.0},  # ratio 200% -- a risque
        ],
    )
    result = main.tool_risk_clients(n=10)
    assert {c["n_client"] for c in result["clients"]} == {"2"}


def test_risk_clients_excludes_tiny_premium_denominator():
    # PRIME_NETTE=50 < MIN_CA_FOR_RISK_TND (200) -- meme avec un ratio enorme
    # (2000%), ce client doit etre exclu (denominateur non representatif).
    _set_full_portfolio(
        prod_rows=[{"N_CLIENT": "3", "AGENCE": "Sfax Ville", "PRIME_NETTE": 50.0}],
        sin_rows=[{"N_CLIENT": "3", "AGENCE": "Sfax Ville", "REGLEMENTS": 1000.0}],
    )
    result = main.tool_risk_clients(n=10)
    assert result["clients"] == []


def test_risk_clients_labels_exceptional_claims_separately():
    _set_full_portfolio(
        prod_rows=[
            {"N_CLIENT": "4", "AGENCE": "Sfax Ville", "PRIME_NETTE": 1000.0},  # ratio 250%
            {"N_CLIENT": "5", "AGENCE": "Sfax Ville", "PRIME_NETTE": 1000.0},  # ratio 2500%
        ],
        sin_rows=[
            {"N_CLIENT": "4", "AGENCE": "Sfax Ville", "REGLEMENTS": 2500.0},
            {"N_CLIENT": "5", "AGENCE": "Sfax Ville", "REGLEMENTS": 25000.0},
        ],
    )
    by_id = {c["n_client"]: c for c in main.tool_risk_clients(n=10)["clients"]}
    assert by_id["4"]["niveau_risque"] == "Risque eleve"
    assert by_id["5"]["niveau_risque"] == "Sinistre exceptionnel isole (a distinguer d'un risque recurrent)"


def test_risk_clients_respects_agence_filter():
    _set_full_portfolio(
        prod_rows=[
            {"N_CLIENT": "6", "AGENCE": "Sfax Ville", "PRIME_NETTE": 1000.0},
            {"N_CLIENT": "7", "AGENCE": "Sousse",     "PRIME_NETTE": 1000.0},
        ],
        sin_rows=[
            {"N_CLIENT": "6", "AGENCE": "Sfax Ville", "REGLEMENTS": 2000.0},
            {"N_CLIENT": "7", "AGENCE": "Sousse",     "REGLEMENTS": 2000.0},
        ],
    )
    result = main.tool_risk_clients(n=10, agence="Sousse")
    assert {c["n_client"] for c in result["clients"]} == {"7"}


def test_risk_clients_degrades_gracefully_without_full_portfolio():
    main.state.pop("prod_full_df", None)
    main.state.pop("sin_full_df", None)
    result = main.tool_risk_clients(n=10)
    assert result["clients"] == []
    assert "note" in result


# ════════════════════════════════════════════════════════════════
# compute_data_drift() — utilise en continu par simulator() (voir main.py)
# ════════════════════════════════════════════════════════════════
def test_compute_data_drift_insufficient_data():
    main.state["contrats"] = []
    main.state["nb_contrats_reel"] = 0
    result = main.compute_data_drift()
    assert "note" in result
    assert "alerte_derive" not in result


def test_compute_data_drift_flags_deviation_above_threshold():
    main.state["ca_total_reel"]    = 100_000.0
    main.state["nb_contrats_reel"] = 100  # reference moyenne = 1000
    main.state["contrats"] = [{"PRIME_NETTE": 2000.0} for _ in range(50)]  # +100%
    result = main.compute_data_drift()
    assert result["alerte_derive"] is True
    assert result["ecart_pct"] > main.DRIFT_SEUIL_ALERTE_PCT


def test_compute_data_drift_no_alert_within_threshold():
    main.state["ca_total_reel"]    = 100_000.0
    main.state["nb_contrats_reel"] = 100  # reference moyenne = 1000
    main.state["contrats"] = [{"PRIME_NETTE": 1050.0} for _ in range(50)]  # +5%
    result = main.compute_data_drift()
    assert result["alerte_derive"] is False
