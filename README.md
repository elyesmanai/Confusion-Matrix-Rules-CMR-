#==============================================================#
||   Projet XAI basé sur CMR, RIPPER et BRCG pour la         ||
||   cybersécurité et l’explicabilité des modèles IA         ||
#==============================================================#

# Présentation

Ce projet implémente un pipeline complet d’intelligence artificielle explicable (XAI)
basé sur des règles afin d’expliquer un modèle de machine learning de type boîte noire.

Le modèle principal utilisé comme référence est un modèle entraîné avec l’algorithme XGBoost.

Les approches explicables étudiées sont :

- CMR (Confusion Matrix Rules)
- RIPPER (Repeated Incremental Pruning to Produce Error Reduction)
- BRCG (Boolean Rule Column Generation)

L’objectif du projet est de :

- entraîner un modèle de classification ;
- extraire des règles explicables ;
- reproduire le comportement du modèle boîte noire ;
- mesurer la fidélité des règles ;
- analyser la couverture des règles ;
- comparer les performances des approches explicables.

Le projet est orienté vers des datasets de cybersécurité et de détection d’intrusion réseau.

------------------------------------------------------

# Pipeline expérimental

Le pipeline global suit les étapes suivantes :

Raw data (X, y) -->
                    XGBoost
                    (modèle boîte noire) -->     
                                             predictions_model -->
                                                                   CMR / RIPPER / BRCG -->
                                                                                           rules -->
                                                                                                     Application des règles sur X_test --> 
                                                                                                                                           predictions_rules -->
                                                                                                                                                                 Évaluation : 
                                                                                                                                                                 F1, Coverage, Fidelity, Temps, RAM

------------------------------------------------------

# Structure du projet

## Modèle boîte noire

### `xgb_blackbox.py`

Entraîne XGBoost et génère :
* predictions_model_train
* predictions_model_test

Permet également l’évaluation du modèle de référence.

------------------------------------------------------

## Pipeline CMR hybride

### `cmr_xgb_pipeline.py`

Implémente un pipeline hybride :
X --> XGBoost --> predictions_model --> CMR --> règles

Le modèle XGBoost est utilisé comme fallback
pour les instances non couvertes par les règles.

------------------------------------------------------

## Pipeline CMR pur

### `cmr_pure.py`

Implémente un CMR basé uniquement sur les règles :
X --> règles --> prédictions

Les cas non couverts utilisent la classe majoritaire.

------------------------------------------------------

## Extraction des règles

### `extract_rules.py`

Extrait les règles CMR, RIPPER et BRCG
à partir des prédictions du modèle XGBoost.

Pipeline :
predictions_model --> XAI --> rules

------------------------------------------------------

## Génération des prédictions des règles

### `predictions_rules.py`

Applique les règles générées sur les données de test
et produit :
* predictions_CMR
* predictions_RIPPER
* predictions_BRCG

------------------------------------------------------

## Algorithme RIPPER

### `ripper_algorithm.py`

Implémentation complète de RIPPER :

* apprentissage ;
* extraction des règles ;
* calcul des métriques ;
* calcul de complexité ;
* évaluation du modèle.

------------------------------------------------------

## Algorithme BRCG

### `brcg_algorithm.py`

Implémentation complète de BRCG :

* binarisation des données ;
* apprentissage du modèle logique ;
* extraction des règles ;
* calcul des métriques ;
* calcul de complexité.

------------------------------------------------------

## Fonctions CMR

### `explainer.py`

Contient les fonctions principales du CMR :

* construction du CMC ;
* extraction des règles exclusives ;
* calcul de coverage ;
* application des règles ;
* gestion des conflits.

------------------------------------------------------

## Fonctions utilitaires

### `utils.py`

Regroupe les fonctions générales du projet :

* chargement des datasets ;
* préparation des données ;
* nettoyage des données ;
* alignement des variables ;
* calcul des métriques ;
* calcul de fidélité ;
* calcul de coverage ;
* calcul de Fidelity_covered ;
* mesure du temps d’exécution ;
* mesure de la RAM ;
* création des dossiers de résultats.

Le fichier centralise les fonctions réutilisables utilisées dans les expériences.


------------------------------------------------------

## Script principal

### `run_all_datasets_fidelity.py`

Script principal des expériences.

Permet :

* d’exécuter les expériences sur tous les datasets ;
* d’évaluer CMR, RIPPER et BRCG ;
* de calculer la fidélité ;
* de mesurer Coverage, Temps et RAM ;
* de sauvegarder automatiquement les résultats.

------------------------------------------------------

## Pipeline XGBoost + règles

### `xgboost_rule_extraction.py`

Pipeline complet :


XGBoost → règles → fidélité

Ce script :

* entraîne XGBoost ;
* construit le CMC ;
* extrait les règles ;
* applique les règles sur le test ;
* mesure les performances.

------------------------------------------------------

## Dataset Adult

### `adultdata_preparation_evaluat.py`

Préparation et encodage du dataset Adult :

* nettoyage ;
* encodage catégoriel ;
* séparation train/test ;
* évaluation de règles manuelles.

------------------------------------------------------

## Expériences Adult

### `run_adult.py`

Script de test rapide sur le dataset Adult
pour comparer :

* XGBoost ;
* RIPPER ;
* BRCG.

------------------------------------------------------
## Notebooks

Les notebooks du projet sont regroupés dans le dossier : notebooks/



### `1.data_prep.ipynb`

Préparation et nettoyage des données.

### `2.modeling.ipynb`

Entraînement des modèles.

### `3.eval_CMR.ipynb`

Évaluation des règles CMR.

### `4.personalize_rules.ipynb`

Personnalisation et analyse des règles générées.

### `5.tableau_comparative_google_Colab.ipynb`

visualisation et analyse des résultats expérimentaux.

------------------------------------------------------

# Datasets utilisés

```python
DATASETS = {
    "KDD99": "Datasets/KDD99/",
    "BIG15": "Datasets/BIG15/",
    "UNSW-NB15": "Datasets/UNSW-NB15/processed/",
    "DoH20": "Datasets/DoH20/"
}
```

Datasets utilisés pour :

* détection d’intrusion ;
* classification réseau ;
* cybersécurité ;
* trafic malveillant.

------------------------------------------------------

# Métriques utilisées

## Classification

* F1_train
* F1_test
* Accuracy

## Coverage

* Train_Coverage
* Test_Coverage
* Coverage_total

## Fidelity

* Fidelity
* Fidelity_covered

## Interprétabilité

* Complexité
* Nombre de règles
* Nombre de conditions

## Ressources système

* Temps d’exécution
* RAM utilisée
* TE / TG / TT
* RAM_TE_MB / RAM_TG_MB / RAM_TT_MB

------------------------------------------------------

# Interprétation des méthodes

## CMR

* génère des règles exclusives ;
* mesure un coverage réel ;
* calcule la fidélité sur les cas couverts.

## RIPPER

* algorithme de règles supervisées ;
* produit un ensemble de règles interprétables ;
* coverage considéré comme complet.

## BRCG

* modèle logique basé sur optimisation ;
* recherche des règles compactes ;
* privilégie des règles simples et interprétables.

------------------------------------------------------

# Exécution avec Docker

## Construction de l’image

sur terminal : 
docker build -t cmr-project .


------------------------------------------------------

## Lancement du projet

sur terminal : 
docker run cmr-project


------------------------------------------------------

## Utilisation des datasets locaux

Les datasets ne sont pas inclus dans le dépôt GitHub.

Ils doivent être placés dans le dossier :

Datasets/


Structure attendue :


Datasets/
        KDD99/
        BIG15/
        UNSW-NB15/
                processed/
        DoH20/


Exécution avec montage local :

sur terminal : 
docker run -v ${PWD}/Datasets:/app/Datasets cmr-project


------------------------------------------------------

# Résultats

Les résultats sont sauvegardés dans :
                                        results/

Fichier principal :
                results/progress_iterations.csv

Tableau comparatif final :

                results/tableau_comparaison_moyenne.csv

tableau_comparative_google_Colab.ipynb : permet de visualiser et analyser les résultats.

------------------------------------------------------

## Tableaux générés

Les tableaux finaux sont enregistrés dans :

results/tableaux_results/


Scripts associés :

* `tableaux/generate_table7.py` : génère le tableau global des performances.
* `tableaux/generate_table8.py` : analyse l’impact du paramètre `s_min` sur CMR.
* `tableaux/generate_table9.py` : calcule les moyennes et écarts-types sur les itérations.
* `tableaux/generate_table10.py` : compare CMR aux méthodes de référence avec le test de Wilcoxon.

Fichiers générés :

results/tableaux_results/table7.csv
results/tableaux_results/table8.csv
results/tableaux_results/table9.csv
results/tableaux_results/table10.csv

Description rapide :

* Table 7 : comparaison globale des performances, temps, mémoire et complexité.
* Table 8 : impact du paramètre `s_min` sur la couverture, les conflits, le F1 et la complexité.
* Table 9 : stabilité des méthodes avec moyenne et écart-type `(μ ± σ)`.
* Table 10 : comparaison statistique entre CMR, RIPPER et BRCG.

------------------------------------------------------

## Figures et distributions

Les figures générées sont enregistrées dans :

results/distributions/

Ce dossier contient les graphiques de distribution des métriques, par exemple :

F1_test.png
Temps.png
RAM_MB.png
TE.png
TG.png
TT.png
RAM_TE_MB.png
RAM_TG_MB.png
RAM_TT_MB.png
Coverage_total.png
Fidelity_covered.png


Ces figures permettent de visualiser les performances, le temps d’exécution, la mémoire utilisée, la couverture et la fidélité selon les méthodes et les datasets.

------------------------------------------------------

# Bibliothèques principales

* XGBoost
* AIX360
* CVXPY
* scikit-learn
* pandas
* NumPy
* matplotlib
* seaborn
* wittgenstein

------------------------------------------------------

# Environnement expérimental

Les expériences ont été réalisées avec :

* Python 3.9
* Windows 10
* scikit-learn 1.6.1
* NumPy 1.23.5
* pandas 1.5.3

## Configuration de la machine

Les expériences ont été réalisées dans l’environnement technique suivant :

* Système d’exploitation : Windows 10
* Processeur : Intel64 Family 6 Model 78 (GenuineIntel)
* Mémoire RAM : 15.41 Go
* Langage de programmation : Python 3.9.25 (64 bits)

------------------------------------------------------

## Auteur

Souleymane Bah  
Maîtrise en Intelligence Artificielle  
Université du Québec à Chicoutimi (UQAC)
2026

