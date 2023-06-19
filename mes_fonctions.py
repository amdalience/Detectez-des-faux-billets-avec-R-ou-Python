# Import des librairies et fonctions nécessaires aux analyses

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans

import warnings
warnings.simplefilter("ignore")

# Fonctions


def backward(data, feature_columns, target_column, alpha=0.05):
    """
    Effectue une sélection récursive des caractéristiques (backward elimination) pour une régression linéaire multiple.

    Arguments :
    - data : DataFrame contenant les données. Les colonnes doivent inclure à la fois les variables indépendantes spécifiées dans feature_columns et la variable cible spécifiée dans target_column.
    - feature_columns : Liste des noms des colonnes des variables indépendantes.
    - target_column : Nom de la colonne de la variable cible.
    - alpha : Seuil de signification pour les tests statistiques (par défaut : 0.05).

    Retour :
    - Liste des indices des paramètres significatifs dans le modèle linéaire multiple.
    """

    # 1. Préparation des données
    X = data[feature_columns]
    y = data[target_column]
    X = sm.add_constant(X)  # Ajouter une colonne constante pour l'interception

    # 2. Construction du modèle initial
    model = sm.OLS(y, X)
    results = model.fit()

    # Sélection des paramètres significatifs (backward elimination)
    significant_params = results.pvalues[1:] < alpha

    if all(significant_params):
        # Tous les paramètres sont significatifs, retourner les indices
        significant_indices = significant_params.index.tolist()
        return significant_indices
    else:
        # Supprimer les colonnes correspondant aux paramètres non significatifs
        non_significant_params = significant_params[~significant_params].index
        X = X.drop(non_significant_params, axis=1)

        # Récursivement ajuster le modèle avec les nouvelles données
        return backward(data, X.columns[1:], target_column, alpha)
    

def reg_lin(data, feature_columns, target_column, alpha=0.05):
    """
    Effectue une régression linéaire multiple sur un ensemble de données.

    Arguments :
    - data : DataFrame contenant les données. Les colonnes doivent inclure à la fois les variables indépendantes
             spécifiées dans feature_columns et la variable cible spécifiée dans target_column.
    - feature_columns : Liste des noms des colonnes des variables indépendantes.
    - target_column : Nom de la colonne de la variable cible.
    - alpha : Seuil de signification pour les tests statistiques (par défaut : 0.05).

    Retour :
    - results_dict : Dictionnaire contenant les résultats de différentes étapes de la régression linéaire.
    """
    # 1. Préparation des données
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ajouter une constante aux variables indépendantes
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # 2. Construction du modèle
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    
    # 3. Résumé du modèle
    summary = results.summary()

    # 4. Prédictions sur les données de test
    y_pred = results.predict(X_test)

    # 5. Calcul du R² pour les données de test
    r2_test = r2_score(y_test, y_pred)
    
    # 6. Vérification des hypothèses
    print("\nVérification des hypothèses de la Régression Linéaire:\n")
    
    # 7.1 Vérification de la linéarité
    print("1. Vérification de la linéarité :")
    fig, axes = plt.subplots(1, len(X_train.columns) - 1, figsize=(15, 4))
    fig.suptitle('Relation entre les variables indépendantes significatives et la variable cible')
    for i, column in enumerate(X_train.columns[1:]):
        sns.scatterplot(x=X_train[column], y=y_train, ax=axes[i])
        sns.regplot(x=X_train[column], y=y_train, scatter=False, color='red', ax=axes[i])
    plt.tight_layout()
    plt.show()

    # Test de linéarité
    lin_test_results = sm.stats.diagnostic.linear_rainbow(results)
    print("Résultats du test de linéarité :", lin_test_results)
    if lin_test_results[1] > alpha:
        print("\n\033[1mL'hypothèse de linéarité est satisfaite.\033[0m Cela signifie que la relation entre les variables indépendantes et la variable \ndépendante peut être modélisée par une relation linéaire.\n")
    else:
        print("\n\033[1mL'hypothèse de linéarité n'est pas satisfaite.\033[0mCela signifie que la relation entre les variables indépendantes et la variable \ndépendante ne peut pas être modélisée par une relation linéaire.\n")

    # 7.2 Détection de la colinéarité (Facteur d'inflation de la variance - VIF)
    print("2. Détection de la colinéarité (Facteur d'inflation de la variance - VIF) :")
    # - Hypothèse de colinéarité
    print("\nVérification de la Colinéarité des variables indépendantes.")
    vif = pd.DataFrame()
    vif["Features"] = X_train.columns[1:]
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(1, X_train.shape[1])]
    print(vif)
    if all(vif["VIF"] < 5):
        print("\n\033[1mAucune colinéarité détectée.\033[0m Les variables indépendantes ne présentent pas de corrélation parfaite entre elles.\n")
    else:
        print("\n\033[1mLa colinéarité est détectée\033[0m. Certaines variables indépendantes présentent une corrélation élevée.\n")

    # 7.3 Vérification de l'homoscédasticité
    print("3. Vérification de l'homoscédasticité :")
    homoscedasticity_test_results = sm.stats.diagnostic.het_breuschpagan(results.resid, X_train.iloc[:, 1:])
    print("Résultats du test d'homoscédasticité :", homoscedasticity_test_results)
    if homoscedasticity_test_results[1] > alpha:
        print("\033[1mL'hypothèse d'homoscédasticité est satisfaite.\033[0m Cela signifie que la variance des résidus est constante et ne pas peut être influencée par les valeurs prédites.\n")
    else:
        print("\033[1mL'hypothèse d'homoscédasticité n'est pas satisfaite.\033[0m Cela signifie que la variance des résidus n'est pas constante et peut être influencée par les valeurs prédites.\n")

    # 7.4 Vérification de la normalité des résidus
    print("4. Vérification de la normalité des résidus :")
    fig, ax = plt.subplots()
    sm.qqplot(results.resid, line='s', ax=ax)
    plt.title("Test de normalité des résidus (QQ-plot)")
    plt.show()

    jb_test_results = sm.stats.stattools.jarque_bera(results.resid)
    print("Résultats du test de Jarque-Bera :", jb_test_results)
    if jb_test_results[1] > alpha:
        print("Les résidus suivent une distribution normale. \n\033[1mL'hypothèse de normalité des résidus est satisfaite.\033[0m\n")
    else:
        print("Les résidus ne suivent pas une distribution normale. \n\033[1mL'hypothèse de normalité des résidus n'est pas satisfaite.\033[0m\n")

    # 7.5 Vérification de l'indépendance des résidus
    print("5. Vérification de l'indépendance des résidus :")
    durbin_watson_test_results = sm.stats.stattools.durbin_watson(results.resid)
    print("Résultats du test de Durbin-Watson :", durbin_watson_test_results)
    if durbin_watson_test_results > 1.5 and durbin_watson_test_results < 2.5:
        print("\033[1mL'indépendance des résidus est satisfaisante.\033[0m\n")
    else:
        print("\033[1mL'indépendance des résidus n'est pas satisfaisante.\033[0m\n")

    # 8. Création du dictionnaire des résultats
    results_dict = {
        "summary": summary,
        'R2_test': r2_test
    }
    
    return results_dict

def reg_log(df, feature_cols, target_col):
    """
    Effectue une régression logistique binaire sur un DataFrame.

    Arguments :
    - df : DataFrame contenant les données. Les colonnes doivent inclure à la fois les variables indépendantes spécifiées dans feature_cols et la variable cible spécifiée dans target_col.
    - feature_cols : Liste des noms des colonnes des variables indépendantes.
    - target_col : Nom de la colonne de la variable cible.

    Retour :
    - Dictionnaire contenant les résultats de la régression logistique, comprenant la matrice de confusion et les métriques de classification.
    """

    # Séparation des variables indépendantes et variable cible
    X = df[feature_cols]
    y = df[target_col]

    # Ajout d'une constante aux variables indépendantes pour l'interception
    X = sm.add_constant(X)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciation du modèle de régression logistique
    logreg_model = sm.Logit(y_train, X_train)

    # Entraînement du modèle
    logreg_results = logreg_model.fit()

    # Prédictions sur l'ensemble de test
    y_pred = logreg_results.predict(X_test)

    # Transformation des prédictions en classes binaires
    y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
    
    # Matrice de confusion
    confusion = confusion_matrix(y_test, y_pred_binary)

    # Métriques de classification
    metrics = classification_report(y_test, y_pred_binary)

    # Affichage du résumé du modèle logistique
    print(logreg_results.summary())

    # Affichage de la matrice de confusion et des métriques
    print("\nMatrice de confusion:")
    print(confusion)
    print("\nMétriques de classification:")
    print(metrics)
    
    # Stockage des résultats dans un dictionnaire
    reg_log_results = {}
    reg_log_results['confusion_matrix'] = confusion
    reg_log_results['classification_report'] = metrics

    # Affichage de la courbe ROC
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC - Régression logistique')
    plt.legend(loc='lower right')
    plt.show()
    
    return reg_log_results

def predict_billets(data, feature_cols, target_col, billet_test):
    """
    Effectue la prédiction de l'authenticité des billets en utilisant un modèle de régression logistique.

    Arguments :
    - data : DataFrame contenant les données d'entraînement. Les colonnes doivent inclure à la fois les variables indépendantes spécifiées dans feature_cols et la variable cible spécifiée dans target_col.
    - feature_cols : Liste des noms des colonnes des variables indépendantes.
    - target_col : Nom de la colonne de la variable cible.
    - billet_test : DataFrame contenant les nouvelles données à prédire. Les colonnes doivent correspondre aux variables indépendantes spécifiées dans feature_cols.

    Retour :
    - DataFrame contenant les résultats de la prédiction, comprenant l'identifiant des billets et les résultats de prédiction (Vrai ou Faux).
    """

    # Séparation des variables indépendantes et variable cible
    X = data[feature_cols]
    y = data[target_col]

    # Ajout d'une constante aux variables indépendantes pour l'interception
    X = sm.add_constant(X)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciation du modèle de régression logistique
    logreg_model = sm.Logit(y_train, X_train)

    # Entraînement du modèle
    logreg_results = logreg_model.fit()
    
    # Ajouter une colonne constante pour l'interception dans les nouvelles données
    billet_test = sm.add_constant(billet_test)

    # Effectuer les prédictions sur les nouvelles données
    y_pred_new = logreg_results.predict(billet_test)

    # Transformer les prédictions en classes binaires
    y_pred_binary = np.where(y_pred_new >= 0.5, "Vrai", "Faux")

    # Ajouter les résultats au DataFrame billet_test
    billet_test['Resultat'] = y_pred_binary
    
    # Supprimer la constante:
    billet_test = billet_test.drop('const', axis=1)

    return billet_test


def reg_kmeans(df, feature_cols, target_col):
    """
    Effectue une classification binaire à l'aide de l'algorithme K-means sur un DataFrame.

    Arguments :
    - df : DataFrame contenant les données. Les colonnes doivent inclure les variables indépendantes spécifiées dans feature_cols et la variable cible spécifiée dans target_col.
    - feature_cols : Liste des noms des colonnes des variables indépendantes.
    - target_col : Nom de la colonne de la variable cible.

    Retour :
    - Dictionnaire contenant les résultats de la classification K-means, comprenant la matrice de confusion et les métriques de classification.
    """

    # Séparation des variables indépendantes
    X = df[feature_cols]

    # Instanciation du modèle K-means
    kmeans_model = KMeans(n_clusters=2, random_state=42)

    # Entraînement du modèle
    kmeans_model.fit(X)

    # Prédictions sur les données
    y_pred = kmeans_model.labels_

    # Conversion des prédictions en classes binaires
    y_pred_binary = np.where(y_pred >= 0.5, 1, 0)

    # Matrice de confusion
    confusion = confusion_matrix(df[target_col], y_pred_binary)

    # Métriques de classification
    metrics = classification_report(df[target_col], y_pred_binary)

    # Affichage de la matrice de confusion et des métriques
    print("\nMatrice de confusion:")
    print(confusion)
    print("\nMétriques de classification:")
    print(metrics)

    # Affichage des données avec les classes prédites
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred_binary)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title("K-means - Prédictions")
    plt.show()
    
    # Stockage des résultats dans un dictionnaire
    kmeans_results = {}
    kmeans_results['confusion_matrix'] = confusion
    kmeans_results['classification_report'] = metrics

    return kmeans_results
