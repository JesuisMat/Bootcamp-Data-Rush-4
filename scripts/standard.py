# ============================================
# SCRIPT 3 : STANDARDISATION DES DONNÉES
# Fichier : 03_standardization.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("="*60)
print("🎯 STANDARDISATION DES DONNÉES POUR PCA ET K-MEANS")
print("="*60)

# ============================================
# 1. CHARGEMENT DES DONNÉES ENRICHIES
# ============================================
print("\n📂 Chargement des données enrichies...")
df = pd.read_csv("output/enriched_marketing_data.csv", sep=';')
print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ============================================
# 2. SÉLECTION DES FEATURES POUR LE CLUSTERING
# ============================================
print("\n" + "="*60)
print("📋 SÉLECTION DES FEATURES")
print("="*60)

clustering_features = [
    'Income', 
    'Age', 
    'Total_Spend', 
    'Days_Since_Last_Purchase', 
    'Days_Since_Join', 
    'Total_Purchases', 
    'Avg_Purchase_Value', 
    'Spend_Income_Ratio', 
    'NumWebVisitsMonth', 
    'Web_Preference', 
    'Deal_Seeker', 
    'Total_Children', 
    'family_size', 
    'is_parent', 
    'is_couple', 
    'education2_encoded'
]

print(f"\n✅ {len(clustering_features)} features sélectionnées pour le clustering")
print("\nListe des features :")
for i, feat in enumerate(clustering_features, 1):
    print(f"   {i:2d}. {feat}")

# Vérifier que toutes les colonnes existent
missing_cols = [col for col in clustering_features if col not in df.columns]
if missing_cols:
    print(f"\n❌ ERREUR : Colonnes manquantes dans le dataset : {missing_cols}")
    exit(1)

# Créer le subset
X = df[clustering_features].copy()
print(f"\n📊 Dimensions du subset : {X.shape}")

# Vérifier les valeurs manquantes
missing_values = X.isnull().sum().sum()
if missing_values > 0:
    print(f"\n⚠️ Attention : {missing_values} valeurs manquantes détectées")
    print(X.isnull().sum()[X.isnull().sum() > 0])
    print("Suppression des lignes avec valeurs manquantes...")
    X = X.dropna()
    print(f"✅ Nouvelles dimensions : {X.shape}")

# ============================================
# 3. STANDARDISATION
# ============================================
print("\n" + "="*60)
print("🔧 STANDARDISATION")
print("="*60)

print("\n📊 Statistiques AVANT standardisation :")
print(X.describe())

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir en DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features, index=X.index)

print("\n✅ Standardisation terminée avec StandardScaler")

# ============================================
# 4. VÉRIFICATION
# ============================================
print("\n" + "="*60)
print("✅ VÉRIFICATION DE LA STANDARDISATION")
print("="*60)

print("\n📊 Statistiques APRÈS standardisation :")
print(X_scaled_df.describe())

# Vérifications critiques
mean_max = X_scaled_df.mean().abs().max()
std_mean = X_scaled_df.std().mean()

print(f"\n🔍 Vérifications critiques :")
print(f"   • Moyenne maximale (doit être ~0) : {mean_max:.6f}")
print(f"   • Écart-type moyen (doit être ~1) : {std_mean:.6f}")

if mean_max < 0.0001 and 0.99 < std_mean < 1.01:
    print("   ✅ Standardisation validée !")
else:
    print("   ⚠️ Valeurs inhabituelles détectées, vérifier les données")

print("\n📋 Aperçu des données standardisées (5 premières lignes) :")
print(X_scaled_df.head())

# ============================================
# 5. SAUVEGARDE
# ============================================
print("\n" + "="*60)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*60)

# Créer le dossier output si nécessaire
os.makedirs('models', exist_ok=True)

# Sauvegarder les données standardisées
output_scaled = "models/scaled_features_for_clustering.csv"
X_scaled_df.to_csv(output_scaled, index=False, sep=';')
print(f"✅ Données standardisées sauvegardées : {output_scaled}")

# Sauvegarder le scaler
scaler_file = 'output/scaler.pkl'
joblib.dump(scaler, scaler_file)
print(f"✅ Scaler sauvegardé : {scaler_file}")

# Sauvegarder aussi les index pour retrouver les clients plus tard
index_file = "output/clustering_indices.csv"
pd.DataFrame({'ID': df.loc[X.index, 'ID']}).to_csv(index_file, index=False)
print(f"✅ Indices des clients sauvegardés : {index_file}")

# ============================================
# 6. RÉSUMÉ FINAL
# ============================================
print("\n" + "="*60)
print("📊 RÉSUMÉ FINAL")
print("="*60)

print(f"\n✅ Nombre de clients après preprocessing : {X_scaled_df.shape[0]}")
print(f"✅ Nombre de features : {X_scaled_df.shape[1]}")
print(f"\n📂 Fichiers générés :")
print(f"   1. {output_scaled}")
print(f"   2. {scaler_file}")
print(f"   3. {index_file}")

print("\n" + "="*60)
print("✅ STANDARDISATION TERMINÉE AVEC SUCCÈS !")
print("="*60)
print("\n🎯 PROCHAINES ÉTAPES :")
print("   1. Lancer le script 04_pca.py pour la réduction de dimensionnalité")
print("   2. Déterminer le nombre optimal de clusters (Elbow + Silhouette)")
print("   3. Appliquer K-Means")
print("   4. Analyser les profils clients")
print("\n🚀 Dataset prêt pour la modélisation !")