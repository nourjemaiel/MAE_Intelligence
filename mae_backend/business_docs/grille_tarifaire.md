# Grille Tarifaire Indicative par Branche (document illustratif)

> ⚠️ Document ILLUSTRATIF créé pour la démonstration RAG — coefficients
> indicatifs, pas la grille tarifaire officielle de MAE.

## Principe de tarification

La prime nette d'un contrat dépend de trois facteurs principaux :
1. **La branche** (nature du véhicule / usage — Tourisme, Taxi, Transport,
   2 Roues, etc.)
2. **Le coefficient bonus-malus** de l'assuré (0 à 13)
3. **La puissance fiscale et l'ancienneté du véhicule**

## Coefficients indicatifs par branche (base 1.0 = Tourisme)

| Branche                          | Coefficient indicatif | Profil de risque |
|-----------------------------------|:---:|---|
| Tourisme                          | 1.00 | Référence — usage particulier standard |
| Taxi                               | 2.10 | Usage intensif, kilométrage élevé |
| Taxi_Collectif                     | 2.30 | Usage intensif, transport de passagers |
| Taxi_Plus_4_places                 | 2.25 | Usage intensif, capacité accrue |
| Louage                             | 2.00 | Transport interurbain régulier |
| Transport_Prive_Inf_3.5T           | 1.35 | Usage professionnel léger |
| Transport_Prive_Sup_3.5T           | 1.70 | Usage professionnel lourd |
| Transport_Public_Inf_3.5T          | 1.90 | Transport public, exposition élevée |
| Transport_Public_Sup_3.5T          | 2.20 | Transport public lourd |
| Transport_Personnel                | 1.15 | Usage mixte particulier/pro |
| Transport_Hotel_Agence             | 1.40 | Usage touristique professionnel |
| Transport_Agricole_Inf_3.5T        | 1.20 | Usage agricole léger |
| Transport_Agricole_Sup_3.5T        | 1.45 | Usage agricole lourd |
| Auto_Ecole_Tourisme                | 1.60 | Conducteurs multiples, apprentissage |
| Auto_Ecole_Utilitaire              | 1.75 | Conducteurs multiples, véhicule utilitaire |
| Engin_Agricole_Prive               | 0.90 | Usage limité, faible kilométrage |
| Engin_Agricole_Etablissement       | 1.05 | Usage professionnel agricole |
| Engin_de_Chantier                  | 1.10 | Usage limité au chantier |
| Transport_Rural                    | 1.25 | Usage rural mixte |
| Ambulance                          | 1.55 | Usage prioritaire, exposition variable |
| 2_Roues                            | 0.75 | Faible capital assuré, sinistralité corporelle plus fréquente |

## Application du bonus-malus

La prime finale = Prime de base (selon coefficient branche) × Coefficient
bonus-malus de l'assuré, où le coefficient bonus-malus suit une échelle
approximative de 0.50 (bonus maximal, BM=0) à 2.00 (malus maximal, BM=13),
avec 1.00 au coefficient de référence (BM=7).

## Remarque sur le segment "Tourisme"

La branche Tourisme représente la grande majorité du portefeuille de
contrats (environ 85% selon l'analyse exploratoire du portefeuille), ce qui
en fait la branche structurante pour le chiffre d'affaires global.
