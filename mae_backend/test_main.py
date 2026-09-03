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
# tool_forecast() — invariant central : total_12_mois == base * (1 + FORECAST_GROWTH_RATE)
# ════════════════════════════════════════════════════════════════
def test_tool_forecast_total_matches_growth_formula():
    main.state["ca_total_reel"] = 1_000_000.0
    result = main.tool_forecast()
    expected = 1_000_000.0 * (1 + main.FORECAST_GROWTH_RATE)
    # Tolerance liee aux arrondis a 2 decimales appliques sur chaque mois
    # avant sommation (round(ca, 2) x12), pas un simple round() global.
    assert math.isclose(result["total_12_mois"], expected, rel_tol=1e-4)


def test_tool_forecast_single_month_within_annual_total():
    main.state["ca_total_reel"] = 1_000_000.0
    annuel  = main.tool_forecast()
    mensuel = main.tool_forecast(mois_num=6)
    assert 0 < mensuel["ca_prev"] < annuel["total_12_mois"]


def test_tool_forecast_confidence_interval_widens_with_horizon():
    # Verifie que l'intervalle de confiance dynamique (voir ci_width_for_month
    # dans main.py) s'elargit bien avec l'horizon, et n'est pas revenu a un
    # pourcentage fixe par accident lors d'un futur refactoring. Teste
    # directement les indices d'horizon (0=premier mois prevu, 11=dernier),
    # PAS des numeros de mois calendaires fixes -- la prevision demarre
    # desormais au mois prochain reel (voir forecast_calendar), donc
    # "Janvier"/"Decembre" ne correspondent plus forcement au 1er/12e mois.
    assert main.ci_width_for_month(0) < main.ci_width_for_month(11)


def test_forecast_calendar_starts_next_real_month():
    from datetime import datetime
    today = datetime(2026, 8, 14)
    calendar = main.forecast_calendar(today=today)
    assert (calendar[0]["mois_num"], calendar[0]["annee"]) == (9, 2026)
    assert (calendar[-1]["mois_num"], calendar[-1]["annee"]) == (8, 2027)


def test_forecast_calendar_wraps_year_at_december():
    from datetime import datetime
    today = datetime(2026, 12, 5)
    calendar = main.forecast_calendar(today=today)
    assert (calendar[0]["mois_num"], calendar[0]["annee"]) == (1, 2027)


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
    # Depuis le passage aux 77 vraies agences MAE (plusieurs par ville,
    # ex. "Sfax 1".."Sfax 5"), il n'y a plus UNE seule agence par ville --
    # normalize_agence fait donc une correspondance par prefixe generique
    # sur la liste reelle plutot qu'une table d'alias figee ville->agence.
    assert main.normalize_agence("sfax").lower().startswith("sfax")
    assert main.normalize_agence("tunis").lower().startswith("tunis")


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
    # AGENCE stocke le nom deja normalise ("Sousse 3", pas "Sousse") --
    # exactement comme dans le vrai portefeuille (voir seed_data()) --
    # tandis que le filtre passe le nom informel tel qu'un utilisateur/agent
    # l'ecrirait, pour verifier tout le pipeline normalize_agence() +
    # apply_filters() ensemble, pas seulement une comparaison exacte.
    _set_full_portfolio(
        prod_rows=[
            {"N_CLIENT": "6", "AGENCE": "Sfax Ville", "PRIME_NETTE": 1000.0},
            {"N_CLIENT": "7", "AGENCE": "Sousse 3",   "PRIME_NETTE": 1000.0},
        ],
        sin_rows=[
            {"N_CLIENT": "6", "AGENCE": "Sfax Ville", "REGLEMENTS": 2000.0},
            {"N_CLIENT": "7", "AGENCE": "Sousse 3",   "REGLEMENTS": 2000.0},
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
