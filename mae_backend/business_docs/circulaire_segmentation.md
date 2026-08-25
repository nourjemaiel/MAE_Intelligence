# Circulaire Interne — Segmentation Clients et Actions Commerciales

> ⚠️ Document ILLUSTRATIF proposé dans le cadre de ce PFE, PAS une
> circulaire officielle MAE — mais les 7 segments et leurs statistiques
> ci-dessous sont RÉELS, directement issus du clustering K-Means du
> portefeuille (05_clustering.py, outputs/segments_clients.csv), pas
> inventés. À mettre à jour si le clustering est ré-exécuté sur de
> nouvelles données (le nombre de segments et leurs noms peuvent changer).

## Objet

Cette circulaire précise les actions commerciales recommandées pour chacun
des 7 segments identifiés par l'analyse de segmentation du portefeuille
(clustering K-Means + PCA, k=7 retenu par score de silhouette).

## Client Premium (9,3% du portefeuille, 6 795 clients)

CA moyen le plus élevé (612 TND), bonus-malus bas (2,9), 10,8 contrats en
moyenne — clients fidèles à forte valeur. Ratio sinistres/primes de 43,3%,
le plus bas devant Client Grand Contrat.

**Actions recommandées** : programme de fidélisation VIP, gestionnaire de
compte dédié, offres multi-contrats exclusives, priorité sur le traitement
des sinistres.

## Client Grand Contrat (4,8%, 3 503 clients)

Peu de contrats (1,7 en moyenne) mais à forte prime unitaire (CA moyen 406
TND). Ratio sinistres/primes le plus bas du portefeuille (37,1%).

**Actions recommandées** : suivi individualisé plutôt que volume,
renouvellement prioritaire, point de contact dédié pour sécuriser ce
contrat à forte valeur.

## Client Fidèle (12,9%, 9 396 clients)

Durée de contrat la plus longue du portefeuille, bonus-malus le plus élevé
des 7 segments (6,7) — base de clientèle stable malgré ce bonus-malus.
Ratio sinistres/primes modéré (50,3%).

**Actions recommandées** : programme de reconnaissance de fidélité,
renouvellement facilité.

## Client Capital Élevé (3,2%, 2 317 clients)

Capital assuré nettement supérieur à la moyenne. Ratio sinistres/primes de
48,7%.

**Actions recommandées** : revue de couverture, vérification que la
tarification reflète bien l'exposition réelle.

## Client à Risque (19,1%, 13 911 clients)

Ratio sinistres/primes le plus élevé du portefeuille (62,4%) — critère de
nommage de ce segment (pas le bonus-malus, qui est au contraire l'un des
plus bas, 2,0 : la sinistralité de ce segment n'est pas liée à un
historique de conduite dégradé mais à un ratio coût/prime défavorable).

**Actions recommandées** : révision de la politique tarifaire pour ce
segment, renforcement des contrôles à la souscription, programme de
prévention des sinistres, suivi rapproché du ratio sinistres/primes par
agence.

## Client Économique (34,2%, 24 867 clients — le plus grand segment)

Prime et durée de contrat les plus faibles, fort volume. Ratio
sinistres/primes de 52,5%.

**Actions recommandées** : digitalisation du parcours pour réduire le coût
d'acquisition et de gestion, offres packagées.

## Clientèle Féminine (16,4%, 11 886 clients)

Segment démographiquement distinct (écart le plus marqué vers la clientèle
féminine parmi les 7 segments). Ratio sinistres/primes de 52,7%.

**Actions recommandées** : analyser les garanties les plus souscrites par
ce segment pour adapter la communication et l'offre.

## Principe général

Toute action commerciale ciblée sur un segment doit être mesurée sur au
moins un trimestre avant d'être généralisée, en suivant l'évolution du
ratio de sinistralité et du taux de rétention du segment concerné.
