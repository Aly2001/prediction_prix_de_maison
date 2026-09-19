HEAD
# Prédiction du prix de vente des maisons (House Prices)

Projet de Machine Learning (régression) visant à prédire le prix de vente d'une maison à partir de ses caractéristiques (surface, qualité, quartier, équipements...).

## Contexte

Ce projet s'appuie sur un dataset immobilier riche (80+ variables) pour construire un modèle de régression capable d'estimer le prix de vente d'un bien. Contrairement à un problème de classification, l'objectif ici est de prédire une valeur continue.

## Dataset

**Source** : [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Kaggle)

- 1460 maisons (1458 après suppression des outliers)
- 80 variables explicatives (numériques et catégorielles)
- Variable cible : `SalePrice` (prix de vente en dollars)

## Démarche

### 1. Diagnostic et nettoyage des valeurs manquantes

Ce dataset présente un défi particulier : de nombreuses valeurs manquantes (`NaN`) ne signifient pas une vraie donnée absente, mais l'**absence d'une caractéristique** (ex. pas de piscine, pas de garage). Chaque colonne a été analysée individuellement à l'aide de la documentation (`data_description.txt`) et croisée avec les données réelles pour déterminer la bonne stratégie :

- **Remplacement par `'None'`** (absence logique d'une caractéristique) : `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `BsmtQual`...
- **Remplacement par `0`** (valeur numérique cohérente avec l'absence) : `MasVnrArea`, `GarageYrBlt`
- **Imputation statistique** (vraie donnée manquante) : `LotFrontage` (médiane), `Electrical` (mode)

### 2. Détection et traitement des outliers

L'analyse du nuage de points `GrLivArea` (surface habitable) vs `SalePrice` a révélé deux valeurs aberrantes : des maisons à très grande surface vendues à un prix anormalement bas, incohérentes avec la tendance générale. Ces deux observations ont été retirées du dataset.

### 3. Analyse exploratoire (EDA)

- Étude des corrélations entre les variables numériques et `SalePrice` (`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF` en tête)
- Visualisations (scatterplots, boxplots) pour explorer les relations entre variables clés et le prix

### 4. Encodage des variables catégorielles

- **Encodage ordinal manuel** pour les variables à échelle de qualité (`ExterQual`, `KitchenQual`, `BsmtQual`...), en respectant l'ordre logique (Excellent > Good > Average > Fair > Poor)
- **One-Hot Encoding** (`pd.get_dummies()`, `drop_first=True`) pour les variables catégorielles nominales (`Neighborhood`, `MSZoning`, `Exterior1st`...)

Le dataset final compte 229 colonnes après encodage.

### 5. Modélisation

Plusieurs modèles ont été entraînés et comparés :

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Régression linéaire simple | 23 107 $ | 67 730 $ | 0.170 |
| Ridge (alpha=1.0) | 18 946 $ | 25 936 $ | 0.878 |
| Lasso (alpha=1.0) | 20 508 $ | 31 141 $ | 0.824 |
| **Ridge optimisé (GridSearchCV)** | **18 182 $** | **24 618 $** | **0.890** |

### 6. Optimisation des hyperparamètres

La régression linéaire simple s'est révélée instable face au grand nombre de variables (228 features), entraînant un R² très faible. L'application de la régularisation (Ridge, Lasso) a considérablement amélioré la stabilité et la performance du modèle. Une recherche d'hyperparamètre (`GridSearchCV`, validation croisée à 5 folds) a permis d'identifier le meilleur coefficient de régularisation (`alpha`) pour Ridge, améliorant encore les résultats.

## Résultat final

Le modèle retenu est une **régression Ridge avec alpha optimisé**, atteignant un **R² de 0.89** sur le jeu de test — le modèle explique environ 89% de la variance des prix de vente.

## Pistes d'amélioration

- Tester des modèles non linéaires (Random Forest Regressor, XGBoost)
- Feature engineering supplémentaire (création de variables combinées, ex. surface totale = sous-sol + habitable)
- Analyse des résidus pour identifier d'éventuels biais systématiques du modèle
- Réduction de dimension (PCA) pour limiter le nombre de variables issues du One-Hot Encoding

## Outils utilisés

- Python
- Pandas, NumPy
- Scikit-learn (LinearRegression, Ridge, Lasso, GridSearchCV)
- Matplotlib, Seaborn

## Structure du projet

```
├── data/
│   ├── train.csv
│   └── data_description.txt
├── notebook.ipynb
└── README.md
```

# prediction_prix_de_maison
Prédiction du prix de vente de maisons (régression) — nettoyage avancé de données, encodage ordinal/One-Hot, Ridge/Lasso, optimisation d'hyperparamètres avec GridSearchCV

