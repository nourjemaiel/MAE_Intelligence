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

CA moyen le plus élevé (612 TND), bonus-malus bas (3), 10,8 contrats en
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

## Client Jeune Conducteur (12,9%, 9 396 clients)

Le plus jeune des 7 segments de loin (46 ans en moyenne, contre 54 ans sur
l'ensemble du portefeuille — l'écart le plus marqué de tous les segments,
bien plus que son propre bonus-malus). Cohérent avec les deux autres traits
du segment : le bonus-malus le plus élevé des 7 segments (7) et la durée
moyenne de contrat la plus courte — des conducteurs jeunes et récemment
assurés démarrent plus haut sur l'échelle bonus-malus et n'ont pas encore
eu le temps de la faire baisser. Chiffre d'affaires moyen correct malgré
tout (260 TND, 2e derrière Premium et Grand Contrat). Ratio
sinistres/primes modéré (50,3%).

**Actions recommandées** : offres d'accompagnement/fidélisation ciblant les
jeunes conducteurs (le bonus-malus élevé baissera avec l'ancienneté si le
client est retenu), pédagogie sur le fonctionnement du bonus-malus dès la
souscription.

## Client Capital Élevé (3,2%, 2 317 clients)

Capital assuré nettement supérieur à la moyenne. Ratio sinistres/primes de
48,7%.

**Actions recommandées** : revue de couverture, vérification que la
tarification reflète bien l'exposition réelle.

## Client à Risque (19,1%, 13 911 clients)

Ratio sinistres/primes le plus élevé du portefeuille (62,4%) — critère de
nommage de ce segment (pas le bonus-malus, qui est au contraire l'un des
plus bas, 2 : la sinistralité de ce segment n'est pas liée à un
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

## Autres Clients (16,4%, 11 886 clients)

Sur le plan financier, ce segment est presque indiscernable du segment
Client Économique : prime moyenne 50,8 TND, 3,3 contrats en moyenne,
bonus-malus 2, ratio sinistres/primes 52,7% (contre 52,5% pour Client
Économique). Si on ne regardait que le chiffre d'affaires ou le risque,
les deux segments seraient à peine distinguables.

Ce qui les sépare réellement : ce segment est composé à 94,3% de clientes
femmes, contre seulement 3,7% pour Client Économique et 24,5% en moyenne
sur l'ensemble du portefeuille (aucun autre segment ne dépasse 30,3%).
L'algorithme de clustering a donc détecté une différence de comportement
réelle, corrélée au genre, qui n'apparaît pas dans les indicateurs
financiers agrégés ci-dessus. Le nom volontairement neutre ("Autres
Clients") évite de nommer ce segment directement sur cette base
démographique : le genre est la variable la plus discriminante trouvée
par l'algorithme pour CE groupe, mais ce n'est pas parce qu'une variable
démographique sépare bien deux groupes que les autres segments en sont
exclusifs pour autant — les autres segments ont eux aussi des clientes,
simplement pas majoritairement.

**Actions recommandées** : analyser spécifiquement les garanties et
options les plus souscrites par ce segment (plutôt que son chiffre
d'affaires, similaire à Client Économique) pour comprendre ce qui motive
réellement cette séparation, et adapter la communication en conséquence.

## Principe général

Toute action commerciale ciblée sur un segment doit être mesurée sur au
moins un trimestre avant d'être généralisée, en suivant l'évolution du
ratio de sinistralité et du taux de rétention du segment concerné.
