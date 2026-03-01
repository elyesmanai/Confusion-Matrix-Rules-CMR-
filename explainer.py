import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

#Fonction 3 Mesurer la couverture
def compute_rule_coverage(rules_df, len_rows):
    """
    Computes total support, removes duplicate IDs, and sorts rules by percentage.

    :param rules_df: DataFrame containing rule-based results with 'support' and 'ids' columns.
    :param len_rows: Total number of rows in the dataset (for percentage calculations).
    :return: Sorted DataFrame of rules.
    """
    # Compute total support / Calcule le support total (avec doublons) par exemple: somme des supports = 1200 sur 10000 lignes → 12%
    total_support = rules_df['support'].sum()
    support_pct = round(total_support / len_rows * 100, 2)
    
    # Remove duplicate IDs /supprime les doublons (une ligne peut être couverte par plusieurs règles)
    unique_ids = set(item for sublist in rules_df['ids'].values for item in sublist)
    unique_ids_pct = round(len(unique_ids) / len_rows * 100, 2)
    
    # Print results
    print(f"Total Support: {support_pct}% ({total_support})")
    print("After removing duplicates:")
    print(f"Unique Coverage: {unique_ids_pct}% ({len(unique_ids)})")
    
    # Return sorted DataFrame / Retourne les règles triées par pourcentage
    return rules_df.sort_values(by='pct', ascending=False), unique_ids

#FONCTION2
def get_exclusive_rules(cmc, analysis_features):
    """
    Finds exclusive rules where a feature value uniquely maps to one CMC category.

    :param cmc: DataFrame containing feature values and CMC labels.
    :param analysis_features: List of features to exclude.
    :return: DataFrame containing exclusive rules.
    """
    len_rows = cmc.shape[0]
    exclusivity_rules_list = []

    # Iterate over features while dropping analysis features
        # Pour chaque feature (sauf celles dans analysis_features)
    for feature in tqdm(cmc.columns.drop(analysis_features)):
        # Get unique CMC groups for each feature value
            # ÉTAPE 1: Pour chaque valeur de la feature, combien de CMC différents ?
            # Exemple: feature='protocol_type'
            # icmp → 1 (seulement TP)
            # tcp  → 2 (TP et TN) 
            # udp  → 1 (seulement TN)
        value_to_cmc = cmc.groupby(feature)['CMC'].nunique()

        # Filter values that belong to only one CMC group
            # ÉTAPE 2: Garde seulement les valeurs avec UN SEUL CMC
            # Résultat: ['icmp', 'udp']
        exclusive_values = value_to_cmc[value_to_cmc == 1].index

        # If no exclusive values, skip
        if exclusive_values.empty:
            continue

        # Get mapping of each exclusive value to its corresponding CMC category
            # ÉTAPE 3: Quel est le CMC pour chaque valeur exclusive ?
            # icmp → 'TP', udp → 'TN'
        cmc_map = cmc.groupby(feature)['CMC'].first()

        # Get IDs for all exclusive values in one step
            # ÉTAPE 4: Quels sont les IDs (indices) pour ces valeurs ?
            # icmp → [0, 5, 12, 45] juste des exemples que jai pris
            # udp  → [3, 7, 89, 123] juste des exemple prise
        id_map = cmc[cmc[feature].isin(exclusive_values)].groupby(feature).apply(lambda x: x.index.tolist())

        # Construct rules efficiently
        # cest ici q'on cree les regles
        for val in exclusive_values:
            group = cmc_map[val]
            ids = id_map[val]
            # Ne garde que les règles correctes (TP et TN)
            if group in ['TP', 'TN']:
                pred = 1 if group == 'TP' else 0   # TP → prédire 1, TN → prédire 0
                support = len(ids)
                pct = round(support / len_rows * 100, 2)

                if support > 2:  # Filter out low-support rules
                    exclusivity_rules_list.append([feature, '==', val, pred, support, pct, ids])

    # Convert list to DataFrame at the end for efficiency
    return pd.DataFrame(exclusivity_rules_list, columns=['feature', 'is', 'value', 'predict', 'support', 'pct', 'ids'])

#1er FONCTION:  Cette fonction transforme les prédictions d’un modèle brute en une analyse détaillée des erreurs (TP, TN, FP, FN) directement intégrée aux données et comprehensible.
def make_cmc(df, y_train, y_pred_train):
    """Fast CMC computation using NumPy vectorization."""
    # 1. Copie le DataFrame pour ne pas modifier l'original
    train = df.copy()
    
    # Convert target variables to NumPy arrays to avoid broadcasting issues
    # 2. Convertit en tableaux NumPy (plus rapide)
    y_train = np.array(y_train).flatten()
    y_pred_train = np.array(y_pred_train).flatten()

    # Ensure they have the same length as df
    # 3. Ici on vérifie si les longueurs correspondent
    assert len(y_train) == len(df), "y_train length does not match dataframe rows"
    assert len(y_pred_train) == len(df), "y_pred_train length does not match dataframe rows"

    # Assign labels to the DataFrame
    train['label'] = y_train
    train['predicted'] = y_pred_train

    # Compute CMC categories efficiently using NumPy           calcule des CMC
    train['CMC'] = np.where(y_pred_train == y_train, # si prediction correcte 
                            np.where(y_pred_train == 1, 'TP', 'TN'), # TP si prédit 1, sinon TN
                            np.where(y_pred_train == 1, 'FP', 'FN')) # FP si prédit 1, sinon FN

    return train

#Fonction 4 Comparer deux ensembles
def compute_rule_overlap(uids_minmax, uids_ex, len_rows):
    """
    Computes overlap and unique coverage between two rule sets.

    :param uids_minmax: Set of unique IDs from min-max rules.
    :param uids_ex: Set of unique IDs from exclusive rules.
    :param len_rows: Total number of rows in the dataset (for percentage calculations).
    :return: Dictionary with overlap statistics.
    """
    
    # Compute intersection (common IDs in both rule sets) / Trouve les IDs communs aux deux ensembles
    intersection_ids = uids_minmax.intersection(uids_ex)
    intersection_count = len(intersection_ids)
    intersection_pct = round(intersection_count / len_rows * 100, 2)
    
    # Compute total unique IDs without double counting overlap /Calcule le total sans double-compte
    total_unique_ids = len(uids_minmax) + len(uids_ex) - intersection_count
    total_unique_pct = round(total_unique_ids / len_rows * 100, 2)
    
    # Print results
    print("Overlap")
    print(f"Count: {intersection_count}")
    print(f"Percentage: {intersection_pct}%\n")
    
    print("Without Overlap")
    print(f"Total Unique Count: {total_unique_ids}")
    print(f"Percentage: {total_unique_pct}%")
    
    return {
        "overlap_count": intersection_count,
        "overlap_pct": intersection_pct,
        "unique_count": total_unique_ids,
        "unique_pct": total_unique_pct
    }

#Fonction 5 
def apply_exclusive_rules(X_test, exclusive_rules, model=None, default_label=None):
    # Initialize predictions with 0, then change skipped ones to None /INITIALISATION
    y_pred = np.full(len(X_test), 0, dtype=object)  # Start with 0 /[0, 0, 0, ...]
    conflict_mask = np.zeros(len(X_test), dtype=bool)  # Tracks conflicts / [False, False, ...]
    no_rule_mask = np.ones(len(X_test), dtype=bool)  # Tracks uncovered cases (start with all True)  /[True, True, ...]

    for _, rule in tqdm(exclusive_rules.iterrows(), total=len(exclusive_rules)):
        feature, value, label, pct = rule['feature'], rule['value'], rule['predict'], rule['pct']

        # Apply rule / trouve les lignes où cette règle s'applique
        mask = X_test[feature] == value
        matching_indices = np.where(mask)[0]

        for idx in matching_indices:
            no_rule_mask[idx] = False  # Rule covered this instance / Cette ligne est couverte

            if y_pred[idx] == 0:  # If still default, assign the first rule /Première règle pour cette ligne
                y_pred[idx] = (label, pct)  
            else:
                prev_label, prev_pct = y_pred[idx] if isinstance(y_pred[idx], tuple) else (y_pred[idx], 0)

                if prev_label == label:
                    y_pred[idx] = (prev_label, prev_pct + pct)  # Sum percentages for same prediction
                else:
                    conflict_mask[idx] = True  # Mark as conflict
                    # Compare summed percentages before deciding
                    if pct > prev_pct:
                        y_pred[idx] = (label, pct)  # Assign label with higher total pct


    # Extract final predictions, replacing default 0 with None where necessary
        # Nettoyer les prédictions
        # Transforme (label, pct) en label, et 0 en None
    y_pred = [p[0] if isinstance(p, tuple) else (None if p == 0 else p) for p in y_pred]

    # Track reasons for skipping
         # Analyser les cas non couverts
    skipped_mask = [p is None for p in y_pred]
    skipped_total = sum(skipped_mask)

    # Correct count of skipped due to conflicts and no rule
    skipped_conflicts = sum(conflict_mask & skipped_mask)
    skipped_no_rule = sum(no_rule_mask & skipped_mask)

    # Validate that elements add up correctly
    assert skipped_total == (skipped_conflicts + skipped_no_rule), \
        "Skipped total does not match the sum of conflict and no-rule skips!"

    # Convert to NumPy array for filtering
        # Calculer F1 sur les cas couverts
    y_test = X_test['label'].values
    valid_mask = ~np.array(skipped_mask)
    y_pred_valid = np.array(y_pred)[valid_mask].astype(np.int64)  # Convert to numeric
    y_test_valid = np.array(y_test)[valid_mask].astype(np.int64)

    # Compute F1-score only for valid predictions
    if len(y_pred_valid) > 0:
        f1_valid = f1_score(y_test_valid, y_pred_valid, average='weighted')
        print(f"Test set coverage: {len(y_pred_valid) / len(y_test) * 100:.2f}%")
        print(f"Test coverage F1-score: {f1_valid:.2f}")
    else:
        print("No valid predictions available for F1-score calculation.")

    # Compute F1-score using ALL instances (assign default label to skipped cases)
    if model:
        X_test_features = X_test.drop('label', axis=1).values  
        missing_idx = np.where(np.array(y_pred) == None)[0]  

        if len(missing_idx) > 0:  
            missing_preds = model.predict(X_test_features[missing_idx])  
            y_pred_full = np.array(y_pred)  
            y_pred_full[missing_idx] = missing_preds  
            y_pred_full = y_pred_full.astype(np.int64)  
        else:
            y_pred_full = np.array(y_pred, dtype=np.int64)  
    else:
        majority_label = np.bincount(y_test).argmax()
        y_pred_full = np.array([majority_label if p is None else p for p in y_pred], dtype=np.int64)

    f1_full = f1_score(y_test, y_pred_full, average='weighted')

    # Print statistics
    total_instances = len(X_test)
    print('--------------------------------------')
    print(f"Total Skipped: {skipped_total} ({round(skipped_total / total_instances * 100, 2)}%)")
    print(f" - Due to Conflicts: {skipped_conflicts} ({round(skipped_conflicts / total_instances * 100, 2)}%)")
    print(f" - Due to No Rule Coverage: {skipped_no_rule} ({round(skipped_no_rule / total_instances * 100, 2)}%)")
    print('--------------------------------------')
    print('Using model to predict non-covered cases.') if model else print('Using Majority class to predict non-covered cases.')
    print(f"F1-score (if using entire dataset): {f1_full:.4f}")
