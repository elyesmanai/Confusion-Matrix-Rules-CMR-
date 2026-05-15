# # Adult Dataset

# ## chargement et traitement de l'ensemble de données

# ### 1. Import des librairies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
from aix360.algorithms.rbm import FeatureBinarizer
import time
import warnings
import wittgenstein as lw
import explainer
import xgboost as xgb
warnings.filterwarnings("ignore")


# ### 2. Définition des types de données & chargement du dataset

# Définition des types de données
data_type = {'age': float,
             'workclass': str,
             'fnlwgt': float,
             'education': str,
             'education-num': float,
             'marital-status': str,
             'occupation': str,
             'relationship': str,
             'race': str,
             'sex': str,
             'capital-gain': float,
             'capital-loss': float,
             'native-country': str,
             'hours-per-week': float,
             'label': str}

# noms des colones
col_names = ['age', 'workclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'label']

# chargement des data
df = pd.read_csv(
    r"C:\Users\Bsmok\Desktop\uqac\adult.data",
    header=None,
    names=col_names,
    skipinitialspace=True
)

# ### 3. Nettoyage des noms de colonnes & informations dataset
# Renommage des colonnes (- en _)
df.columns = df.columns.str.replace('-', '_')
df.info()


# ### 4. Préparation des données pour l’apprentissage supervisé
# Préparation des données :
# 1. Définition de la variable cible
# 2. Analyse de la distribution des classes
# 3. Séparation des données en ensembles d'entraînement (80%) et de test (20%)
TARGET_COLUMN = 'label'
POS_VALUE = '>50K' # Setting positive value of the label for which we train
values_dist = df[TARGET_COLUMN].value_counts() # Distribution des classes

#ici on defini la separation des data de 20 et 80% 
train, test = train_test_split(df, test_size=0.2, random_state=42)
# Split the data set into 80% training and 20% test set
print('Training set:')
print(train[TARGET_COLUMN].value_counts())
print('Test set:')
print(test[TARGET_COLUMN].value_counts())

y_train = train[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
x_train = train.drop(columns=[TARGET_COLUMN])

y_test = test[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
x_test = test.drop(columns=[TARGET_COLUMN])


# ### 5. Encodage des variables catégorielles
# convertion des variables catégorielles en valeurs numériques
from sklearn.preprocessing import LabelEncoder

mapping = {}
#Label encode categorical columns
#Convertit les valeurs textuelles en nombres
le = LabelEncoder()
for col in x_train.select_dtypes(include=[object]).columns:
    x_train[col] = le.fit_transform(x_train[col])
    x_test[col] = le.transform(x_test[col])
    mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))


# ## Évaluation de règle de classification basé sur une ou plusieurs règles

# Ici on teste avec une seul regle :  rule_mask
def calculate_coverage_and_correctness(X_test, y_test, mapping):
    """
    Calculates:
    - Coverage: % of test samples covered by the rule.
    - Correctness: % of covered instances that are actually '1' in y_test.

    :param X_test: DataFrame containing test samples.
    :param y_test: Ground truth labels.
    :param mapping: Dictionary mapping categorical values to encoded integers.
    :return: Coverage percentage and correctness percentage.
    """
    # Get encoded value for 'Married-civ-spouse'
    marital_status_encoded = mapping['marital_status'].get('Married-civ-spouse', -1)

    # Get encoded values for excluded occupations
    excluded_occupations = ['Craft-repair', 'Farming-fishing', 'Handlers-cleaners', 'Other-service'] # liste des métiers à exclure dans la règle.
    excluded_occupation_codes = [mapping['occupation'].get(occ, -1) for occ in excluded_occupations] # transforme ces noms en valeurs encodées grâce au dictionnaire mapping

    # Apply rule conditions
    rule_mask = (
        (X_test['age'] > 26.0) &
        (X_test['education_num'] > 9.0) &
        (X_test['marital_status'] == marital_status_encoded) &
        (~X_test['occupation'].isin(excluded_occupation_codes))
    )

    # Compute Coverage
    covered_count = rule_mask.sum()
    total_count = len(X_test)
    coverage_pct = (covered_count / total_count) * 100 if total_count > 0 else 0

    # Extract corresponding labels
    covered_labels = y_test[rule_mask]

    # Compute Correctness (Percentage of covered instances where y_test == 1)
    correct_count = (covered_labels == 1).sum()
    correctness_pct = (correct_count / covered_count) * 100 if covered_count > 0 else 0

    # Print Results
    print(f"Covered Instances: {covered_count} / {total_count} ({coverage_pct:.2f}%)")
    print(f"Correct Instances: {correct_count} / {covered_count} ({correctness_pct:.2f}%)")

    return coverage_pct, correctness_pct


# ### Évaluation d’un modèle basé sur plusieurs règles
#ici on n’est plus sur une seule règle, mais sur un ensemble très large de règles combinées avec des OR
def calculate_coverage_and_correctness(X_test, y_test, mapping):
    """
    Calculates:
    - Coverage: % of test samples covered by the rule.
    - Correctness: % of covered instances that are actually '1' in y_test.

    :param X_test: DataFrame containing test samples.
    :param y_test: Ground truth labels.
    :param mapping: Dictionary mapping categorical values to encoded integers.
    :return: Coverage percentage and correctness percentage.
    """
    # Get encoded value for 'Married-civ-spouse'
    marital_status_encoded = mapping['marital_status'].get('Married-civ-spouse', -1)

    # Get encoded values for categorical conditions
    occupation_exec_managerial = mapping['occupation'].get('Exec-managerial', -1)
    occupation_prof_specialty = mapping['occupation'].get('Prof-specialty', -1)
    education_some_college = mapping['education'].get('Some-college', -1)
    relationship_wife = mapping['relationship'].get('Wife', -1)
    workclass_federal_gov = mapping['workclass'].get('Federal-gov', -1)
    workclass_private = mapping['workclass'].get('Private', -1)
    workclass_self_emp_inc = mapping['workclass'].get('Self-emp-inc', -1)

    # Apply rule conditions (logical OR for multiple conditions)
    rule_mask = (
    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['age'] >= 37.0) & 
     (X_test['education_num'] <= 11.0) & (X_test['hours_per_week'] >= 38.0) & 
     (X_test['workclass'] == mapping['workclass']['Federal-gov'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 14.0) & (X_test['capital_loss'] >= 1741.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['education_num'] <= 11.0) & 
     (X_test['hours_per_week'] >= 38.0) & (X_test['fnlwgt'] >= 139000.0) & 
     (X_test['fnlwgt'] <= 145098.0) & (X_test['age'] <= 54.0) & (X_test['age'] >= 51.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['education_num'] <= 10.0) &
     (X_test['hours_per_week'] >= 38.0) & (X_test['age'] <= 59.0) & 
     (X_test['hours_per_week'] <= 46.0) & (X_test['age'] >= 58.0) & 
     (X_test['occupation'] == mapping['occupation']['Adm-clerical'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['education_num'] <= 11.0) & 
     (X_test['fnlwgt'] >= 140664.0) & (X_test['hours_per_week'] >= 44.0) & 
     (X_test['hours_per_week'] <= 52.0) & (X_test['age'] >= 57.0) & 
     (X_test['fnlwgt'] <= 155256.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['education_num'] <= 10.0) & 
     (X_test['hours_per_week'] >= 40.0) & (X_test['hours_per_week'] <= 48.0) & 
     (X_test['education'] == mapping['education']['Some-college']) & 
     (X_test['age'] >= 48.0) & (X_test['fnlwgt'] >= 281540.0) & (X_test['age'] <= 52.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['occupation'] == mapping['occupation']['Prof-specialty']) & 
     (X_test['education_num'] >= 14.0) & (X_test['education'] == mapping['education']['Prof-school'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['hours_per_week'] >= 36.0) & (X_test['occupation'] == mapping['occupation']['Prof-specialty']) & 
     (X_test['education_num'] >= 14.0) & (X_test['hours_per_week'] <= 40.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 13.0) & (X_test['occupation'] == mapping['occupation']['Exec-managerial']) & 
     (X_test['age'] >= 45.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 13.0) & (X_test['hours_per_week'] >= 31.0) & 
     (X_test['capital_loss'] >= 1741.0) & (X_test['capital_loss'] <= 1977.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 10.0) & (X_test['age'] >= 36.0) & 
     (X_test['capital_gain'] >= 5060.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 13.0) & (X_test['hours_per_week'] >= 35.0) & 
     (X_test['age'] >= 29.0) & (X_test['occupation'] == mapping['occupation']['Exec-managerial'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 12.0) & (X_test['hours_per_week'] >= 41.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 11.0) & (X_test['age'] >= 48.0) & 
     (X_test['hours_per_week'] <= 40.0) & (X_test['age'] <= 61.0)) |

     ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['age'] >= 34.0) & (X_test['education_num'] >= 11.0) & 
     (X_test['occupation'] == mapping['occupation']['Exec-managerial'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 10.0) & (X_test['age'] >= 36.0) & 
     (X_test['capital_loss'] >= 1848.0) & (X_test['capital_loss'] <= 1977.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['age'] >= 36.0) & 
     (X_test['education_num'] <= 11.0) & (X_test['fnlwgt'] >= 117073.0) & 
     (X_test['occupation'] == mapping['occupation']['Tech-support'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 12.0) & (X_test['fnlwgt'] <= 306982.0) & 
     (X_test['fnlwgt'] >= 191364.0) & (X_test['hours_per_week'] >= 38.0) & 
     (X_test['age'] >= 44.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 10.0) & (X_test['age'] >= 31.0) & 
     (X_test['occupation'] == mapping['occupation']['Exec-managerial']) & 
     (X_test['workclass'] == mapping['workclass']['Self-emp-inc']) & 
     (X_test['fnlwgt'] >= 199352.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['education_num'] <= 10.0) & 
     (X_test['age'] >= 47.0) & (X_test['workclass'] == mapping['workclass']['Private']) & 
     (X_test['fnlwgt'] <= 145574.0) & (X_test['age'] <= 50.0) & 
     (X_test['fnlwgt'] >= 102821.0) & (X_test['hours_per_week'] >= 41.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 10.0) & (X_test['hours_per_week'] >= 44.0) & 
     (X_test['fnlwgt'] >= 218521.0) & (X_test['fnlwgt'] <= 255667.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['age'] >= 34.0) & (X_test['education_num'] >= 12.0) & 
     (X_test['occupation'] == mapping['occupation']['Prof-specialty']) & 
     (X_test['fnlwgt'] >= 193769.0) & (X_test['hours_per_week'] >= 39.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['capital_gain'] >= 5060.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['occupation'] == mapping['occupation']['Exec-managerial']) & 
     (X_test['workclass'] == mapping['workclass']['Private']) & 
     (X_test['age'] <= 56.0) & (X_test['age'] >= 40.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['education_num'] <= 11.0) & 
     (X_test['age'] >= 45.0) & (X_test['workclass'] == mapping['workclass']['Self-emp-inc']) & 
     (X_test['fnlwgt'] <= 194995.0) & (X_test['hours_per_week'] >= 46.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 10.0) & (X_test['age'] >= 37.0) & 
     (X_test['occupation'] == mapping['occupation']['Sales']) & 
     (X_test['age'] <= 48.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 10.0) & (X_test['age'] >= 28.0) & 
     (X_test['occupation'] == mapping['occupation']['Prof-specialty']) & 
     (X_test['relationship'] == mapping['relationship']['Wife']) & 
     (X_test['hours_per_week'] <= 35.0) & (X_test['age'] <= 42.0) & 
     (X_test['capital_gain'] <= 0.0)) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['age'] >= 34.0) & 
     (X_test['education_num'] <= 11.0) & (X_test['hours_per_week'] >= 45.0) & 
     (X_test['fnlwgt'] >= 260578.0) & 
     (X_test['education'] == mapping['education']['Some-college'])) |

    ((X_test['marital_status'] == mapping['marital_status']['Married-civ-spouse']) & 
     (X_test['education_num'] >= 9.0) & (X_test['age'] >= 36.0) & 
     (X_test['education_num'] <= 11.0) & (X_test['hours_per_week'] >= 38.0) & 
     (X_test['capital_loss'] >= 1741.0) & (X_test['capital_loss'] <= 1977.0))
)
    # Compute Coverage
    covered_count = rule_mask.sum()
    total_count = len(X_test)   
    coverage_pct = (covered_count / total_count) * 100 if total_count > 0 else 0
    # Extract corresponding labels
    covered_labels = y_test[rule_mask]

    # Compute Correctness (Percentage of covered instances where y_test == 1)
    correct_count = (covered_labels == 1).sum()
    correctness_pct = (correct_count / covered_count) * 100 if covered_count > 0 else 0

    # Print Results
    print(f"Covered Instances: {covered_count} / {total_count} ({coverage_pct:.2f}%)")
    print(f"Correct Instances: {correct_count} / {covered_count} ({correctness_pct:.2f}%)")

    return coverage_pct, correctness_pct


# ### Évaluation de la couverture et de la correction des règles
print('train')
calculate_coverage_and_correctness(x_train, y_train, mapping)
print('test')
calculate_coverage_and_correctness(x_test, y_test, mapping)


# ### Chargement et préparation du dataset Adult
def prepare_adult_dataset():
    
    return x_train, x_test, y_train, y_test, mapping

