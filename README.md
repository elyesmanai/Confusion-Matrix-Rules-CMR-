==============================================================
||Projet : Explication de modèles avec CMR, RIPPER et BRCG  ||
||Objectif du projet                                        ||
==============================================================
Ce projet vise à expliquer un modèle de machine learning (boîte noire) à l’aide de méthodes explicables basées sur des règles :
•	CMR (Confusion Matrix Rules)
•	RIPPER
•	BRCG
L’objectif est de :
•	Extraire des règles à partir des prédictions du modèle
•	Comparer ces règles au modèle initial
•	Mesurer leur fidélité et leur couverture

=============================================================================
 Pipeline global
    Le pipeline suit les étapes suivantes :
    1. Raw data (X, y)
            ↓
    2. XGBoost (modèle boîte noire)
            ↓
    3. predictions_model
            ↓
    4. XAI (CMR / RIPPER / BRCG)
            ↓
    5. rules
            ↓
    6. Application des règles sur X_test
            ↓
    7. predictions_rules
            ↓
    8. Évaluation (F1,temps test, ram test, Coverage, Fidelity)

===========================================================================
== Structure du projet==
    1. Modèle boîte noire
    xgb_blackbox.py
    •	Entraîne un modèle XGBoost
    •	Génère :
    o	predictions_model_train
    o	predictions_model_test
===========================================================================
 2. Extraction des règles
    extract_rules.py
    •	Utilise predictions_model_train pour entraîner :
    o	CMR
    o	RIPPER
    o	BRCG
    ==> Important : les méthodes explicables apprennent à imiter XGBoost.
============================================================================
 3. Prédictions avec les règles
    predictions_rules.py
    •	Applique les règles sur X_test
    •	Produit :
    o	predictions_CMR
    o	predictions_RIPPER
    o	predictions_BRCG
============================================================================
 4. Métriques de fidélité
    fidelity.py
    •	Compare :
    predictions_model vs predictions_rules
    •	Donne la fidélité globale (%)
============================================================================
 5. Coverage + Fidelity couverte
    covered_fidelity.py
    Calcule :
    •	Coverage_total = % de données couvertes par les règles
    •	Fidelity_covered = fidélité uniquement sur ces données
===========================================================================
6. run_all_datasets.py
Compare CMR, RIPPER et BRCG directement sur (X, y).

    Pas de XGBoost -> pas de fidélité
    Évaluation classique (F1, temps, RAM, etc.)

 Fichier dans "supprimer" → à remettre dans le dossier principal pour l’utiliser (les fichiers tests inclus, donc pareil).


===========================================================================
7. xgboost_rule_extraction.py

Script qui entraine un modèle XGBoost puis génère des règles CMR à partir de ses prédictions.

 Entraîne XGBoost sur (X, y)
 Génère "predictions_model"
 Extrait des règles via CMR (explainer)
 Applique les règles sur le test

Ce script implémente le pipeline XAI complet (modèle boîte noire → règles).

===========================================================================
 8. Métriques complètes
        cmr_pure.py, ripper_algorithm.py, brcg_algorithm.py
        Calculent :
            •	F1_train / F1_test
            •	Coverage
            •	Exactitude
            •	Complexité
            •	Nombre de règles
            •	Nombre de conditions
            •   etc ...

    Fonctions utilitaires : 
        utils.py : Ce fichier regroupe les fonctions générales du projet.

==========================================================================
 9. Script principal
    run_all_datasets_iterations_fidelity.py
    •	Lance les expériences sur tous les datasets
    •	Répète sur plusieurs itérations
    •	Sauvegarde les résultats dans :
    results/progress_iterations.csv
==========================================================================
 Datasets utilisés
    DATASETS = {
        "KDD99": "../Datasets/KDD99/",
        "BIG15": "../Datasets/BIG15/",
        "UNSW-NB15": "../Datasets/UNSW-NB15/processed/",
        "DoH20": "../Datasets/DoH20/"
    }
==========================================================================
 Métriques utilisées
    Classification
    •	F1_train
    •	F1_test
    Coverage
    •	Train_Coverage
    •	Test_Coverage
    •	Coverage_total
    Fidelity
    •	Fidelity (globale)
    •	Fidelity_covered (sur zone couverte)
    Autres
    •	Complexité
    •	Nombre de règles
    •	Nombre de conditions
    •	Temps (test uniquement)
    •	RAM (test uniquement)
========================================================================
 Temps et mémoire
    Important :
    Les mesures de temps et RAM correspondent uniquement à :
    Application des règles sur X_test
    Elles n’incluent pas :
    •	l’entraînement XGBoost
    •	l’extraction des règles
=======================================================================
 Interprétation des méthodes
    CMR
    •	Coverage réel des règles
    •	Fidelity calculée uniquement sur les cas couverts
    RIPPER & BRCG
    •	Coverage = 100% (modèles complets)
    •	Fidelity_covered = fidélité globale
=======================================================================
 Exécution
    1. Activer l’environnement
        conda activate cyber_rules
    2. Lancer le script
        python run_all_datasets_iterations_fidelity.py
=======================================================================
Tableaux : 
    Pour consulter les résultats des tableaux, voir le dossier results/tableaux_results.

        Table 7 : Comparaison globale des performances (Accuracy, Coverage, F1, Complexité) entre CMR, RIPPER et BRCG.
        Table 8 : Analyse de l’impact du paramètre s_min sur les règles et les performances.
        Table 9 : Évaluation de la stabilité des modèles avec moyenne et écart-type (μ ± σ).
        Table 10 : Comparaison statistique (CMR vs baselines) avec amélioration et p-value (test de Wilcoxon).
        Table temps : Mesure du temps de génération des explications globales pour chaque méthode.
=======================================================================
 Résultats
     Pour consulter les Résultats, voir le dossier results.
        Fichier principal
            results/progress_iterations.csv
            Contient :
            •	résultats par dataset
            •	résultats par méthode
            •	résultats par itération
        Fichier final
        results/tableau_comparatif_avec_nouvelles_metriques.csv
        Contient :
            •   la moyenne des résultats sur les itérations
            •   un résumé final pour chaque :
                                            •   dataset
                                            •   méthode (CMR, RIPPER, BRCG)
=======================================================================


