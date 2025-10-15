import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- 1. Charger le dataset ---
df = pd.read_csv("data/5-Camp_Market.csv", sep=';')

# --- 2. Initialiser un DataFrame pour suivre les suppressions ---
deletion_log = []  # Liste pour stocker les infos des lignes supprimées
original_count = len(df)  # Compter le nombre initial de lignes


# --- 2. Nettoyage préliminaire ---
# Supprimer les lignes avec Year_Birth invalide (ex: 1995 avec des enfants)
# df = df[~((df['Year_Birth'] >= 1990) & (df['Kidhome'] + df['Teenhome'] > 0))]

# Supprimer les doublons (mêmes données mais ID différents)
df = df.drop_duplicates(subset=df.columns.difference(['ID']))

# --- 3. Créer les colonnes Age et Ancienneté ---
reference_year = 2015
reference_date = pd.to_datetime('2015-12-31')

df['Age'] = reference_year - df['Year_Birth']
df['start_membership'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
df['Anciennete'] = (reference_date - df['start_membership']).dt.days

# --- 4. Gérer les valeurs manquantes dans Income ---
# Option 1: Remplacer par la médiane groupée par Education (si Age n'est pas encore fiable)
median_income_by_education = df.groupby('Education')['Income'].transform('median')
df['Income'] = df['Income'].fillna(median_income_by_education)

# Option 2: Remplacer par la médiane globale (si groupby échoue)
# df['Income'] = df['Income'].fillna(df['Income'].median())

# --- 5. Corriger les valeurs aberrantes ---
# Remplacer Income = 666666 par la médiane
df.loc[df['Income'] == 666666, 'Income'] = df['Income'].median()

# --- 6. SUPPRIMER (pas remplacer) les statuts maritaux invalides ---
invalid_marital_mask = df['Marital_Status'].isin(['YOLO', 'Absurd'])
invalid_marital_rows = df[invalid_marital_mask].copy()  # Sauvegarde pour logging

# Log des lignes supprimées (optionnel mais recommandé)
print(f"\n⚠️ Suppression de {len(invalid_marital_rows)} lignes avec Marital_Status invalide (YOLO/Absurd) :")
print(invalid_marital_rows[['ID', 'Marital_Status', 'Education', 'Income']])

# Suppression effective
df = df[~invalid_marital_mask]

# --- 6. SUPPRIMER (pas remplacer) les statuts maritaux invalides ---
invalid_marital_mask = df['Marital_Status'].isin(['YOLO', 'Absurd'])
invalid_marital_rows = df[invalid_marital_mask].copy()  # Sauvegarde pour logging

# Log des lignes supprimées (optionnel mais recommandé)
print(f"\n⚠️ Suppression de {len(invalid_marital_rows)} lignes avec Marital_Status invalide (YOLO/Absurd) :")
print(invalid_marital_rows[['ID', 'Marital_Status', 'Education', 'Income']])

# Suppression effective
df = df[~invalid_marital_mask]

invalid_birth_mask = df['Year_Birth'] < 1940
invalid_birth_rows = df[invalid_birth_mask].copy()

print(f"\n⚠️ Suppression de {len(invalid_birth_rows)} lignes avec Year_Birth < 1940 :")
print(invalid_birth_rows[['ID', 'Year_Birth', 'Age', 'Income']])

df = df[~invalid_birth_mask]
print(f"✅ Lignes restantes : {len(df)}")

# --- 7. Encodage des variables catégorielles ---
# Education
education_mapping = {
    'Basic': 1,
    '2n Cycle': 2,
    'Graduation': 3,
    'Master': 4,
    'PhD': 5
}
df['Education'] = df['Education'].map(education_mapping)

# Marital_Status
marital_mapping = {
    'Single': 1,
    'Married': 2,
    'Together': 3,
    'Divorced': 4,
    'Alone': 5,
    'Widow': 6,
    'Other': 7
}
df['Marital_Status'] = df['Marital_Status'].map(marital_mapping)

# --- 8. Feature Engineering ---
# Total_Spend
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spend'] = df[spend_cols].sum(axis=1)

# Total_Purchases
purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

# Total_Children
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# Spend_Income_Ratio (éviter la division par zéro)
df['Spend_Income_Ratio'] = df['Total_Spend'] / df['Income'].replace(0, np.nan)

# --- 9. Supprimer les colonnes inutiles ---
df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1, inplace=True, errors='ignore')

# --- 2. Réorganiser MANUELLEMENT les colonnes ---
# Définissez ici l'ordre SOUHAITÉ pour TOUTES les colonnes
# (Incluez toutes les colonnes existantes, sinon elles seront placées à la fin)
custom_order = [
    'ID', 'Year_Birth', 'Education', 'Marital_Status',  # Démographie
    'Income', 'Kidhome', 'Teenhome', 'Total_Children',  # Famille/Revenu
    'Recency', 'start_membership', 'Anciennete',        # Temps
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',  # Dépenses (à renommer)
    'MntSweetProducts', 'MntGoldProds', 'Total_Spend', 'Spend_Income_Ratio', 'Age',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',  # Achats
    'NumStorePurchases', 'Total_Purchases', 'NumWebVisitsMonth',
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',  # Campagnes (ORDRE MANUEL)
    'Response',  # Variable cible (si elle existe)
    # Ajoutez ici les autres colonnes si nécessaire
]

# --- Appliquer l'ordre personnalisé ---
# 1. Vérifier que toutes les colonnes existent (sinon, les ajouter à la fin)
missing_cols = [col for col in df.columns if col not in custom_order]
final_order = custom_order + missing_cols


# 2. Réorganiser le DataFrame
df = df[final_order]

# 3. Renommer les colonnes (comme demandé dans la todo list)
df.rename(columns={
    'Dt_Customer': 'Start_Membership',  # Déjà fait dans votre code, mais cohérent ici
    'MntWines': 'Amt_Wines',
    'MntFruits': 'Amt_Fruits',
    'MntMeatProducts': 'Amt_Meat_Products',
    'MntFishProducts': 'Amt_Fish_Products',
    'MntSweetProducts': 'Amt_Sweet_Products',
    'MntGoldProds': 'Amt_Gold_Products', # Optionnel : uniformiser les noms
    'Recency': 'Days_Since_Last_Purchase', 
    'NumCatalogPurchases': 'Num_Catalog_Purchases',
    'NumStorePurchases': 'Num_Store_Purchases',
    'NumDealsPurchases': 'Num_Deals_Purchases',
    'NumStorePurchases': 'Num_Store_Purchases',
    'NumWebPurchases': 'Num_Web_Purchases',
    'Anciennete': 'Days_Since_Join',
    'Response': 'Accepted_Last_Cmp',
    'AcceptedCmp1': 'Accepted_Cmp_1',
    'AcceptedCmp2': 'Accepted_Cmp_2',
    'AcceptedCmp3': 'Accepted_Cmp_3',
    'AcceptedCmp4': 'Accepted_Cmp_4',
    'AcceptedCmp5': 'Accepted_Cmp_5',
    'Kidhome': 'Kid_Home',
    'Teenhome': 'Teen_Home'
          # Optionnel : clarifier l'unité
}, inplace=True)

os.makedirs('output', exist_ok=True)

# Chemin du fichier de sortie
output_clean_csv = 'output/clean_marketing_data.csv'

# Sauvegarder le DataFrame nettoyé
df.to_csv(output_clean_csv, index=False, sep=';')  # Utilisation de ';' pour cohérence avec l'import
print(f"\n✅ Dataset nettoyé sauvegardé : {output_clean_csv}")


# --- 10. Vérification finale ---
print("Valeurs manquantes après nettoyage :")
print(df.isnull().sum())
print("\nAperçu du dataset :")
print(df.head())
