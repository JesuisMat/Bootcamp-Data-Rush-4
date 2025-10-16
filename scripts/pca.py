# ============================================
# SCRIPT 4 : PCA (ANALYSE EN COMPOSANTES PRINCIPALES)
# Fichier : 04_pca.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuration graphique
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("📊 PCA - RÉDUCTION DE DIMENSIONNALITÉ")
print("="*60)

# ============================================
# 1. CHARGEMENT DES DONNÉES STANDARDISÉES
# ============================================
print("\n📂 Chargement des données standardisées...")
X_scaled_df = pd.read_csv("output/scaled_features_for_clustering.csv", sep=';')
print(f"✅ Dataset chargé : {X_scaled_df.shape[0]} lignes × {X_scaled_df.shape[1]} colonnes")

# Convertir en array numpy pour la PCA
X_scaled = X_scaled_df.values
feature_names = X_scaled_df.columns.tolist()

print(f"\n📋 Features utilisées : {len(feature_names)}")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")

# ============================================
# 2. APPLIQUER LA PCA AVEC 3 COMPOSANTES
# ============================================
print("\n" + "="*60)
print("🔧 APPLICATION DE LA PCA")
print("="*60)

# Appliquer PCA avec 3 composantes
n_components = 3
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"\n✅ PCA appliquée avec {n_components} composantes principales")
print(f"📊 Dimensions après PCA : {X_pca.shape}")

# ============================================
# 3. VARIANCE EXPLIQUÉE
# ============================================
print("\n" + "="*60)
print("📊 VARIANCE EXPLIQUÉE PAR CHAQUE COMPOSANTE")
print("="*60)

# D'abord, PCA avec TOUTES les composantes pour analyser la variance
print("\n📊 Étape 1 : PCA avec toutes les composantes pour analyse de variance...")
pca_full = PCA()
pca_full.fit(X_scaled)

n_features = len(feature_names)
variance_ratio_full = pca_full.explained_variance_ratio_
cumulative_variance_full = np.cumsum(variance_ratio_full)

print(f"✅ PCA complète appliquée ({n_features} composantes)")

# Afficher les premières composantes
print("\n📊 Variance expliquée par les 10 premières composantes :")
for i in range(min(10, n_features)):
    print(f"   PC{i+1:2d} : {variance_ratio_full[i]*100:5.2f}%  (Cumul: {cumulative_variance_full[i]*100:5.2f}%)")

# Trouver le nombre de PC pour atteindre 80%, 90%, 95%
for threshold in [0.80, 0.90, 0.95]:
    n_pc = np.argmax(cumulative_variance_full >= threshold) + 1
    print(f"\n🎯 Pour capturer {threshold*100:.0f}% de variance → {n_pc} composantes nécessaires")


# Variance expliquée par chaque PC
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

print("\nVariance expliquée par composante :")
for i, var in enumerate(variance_ratio, 1):
    print(f"   PC{i} : {var*100:.2f}%")

print(f"\nVariance cumulée :")
for i, cum_var in enumerate(cumulative_variance, 1):
    print(f"   PC1 à PC{i} : {cum_var*100:.2f}%")

print(f"\n🎯 Les {n_components} premières composantes capturent {cumulative_variance[-1]*100:.2f}% de la variance totale")

# 3. GRAPHIQUE VARIANCE CUMULÉE
# ============================================
print("\n📈 Génération du graphique de variance cumulée...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Graphique 1 : Variance par composante (barres)
ax1.bar(range(1, n_features + 1), variance_ratio_full, alpha=0.7, color='steelblue')
ax1.set_xlabel('Nombre de Composantes Principales', fontsize=12)
ax1.set_ylabel('Variance Expliquée', fontsize=12)
ax1.set_title('Variance Expliquée par Composante', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')
ax1.set_xlim(0, n_features + 1)

# Graphique 2 : Variance cumulée (ligne)
ax2.plot(range(1, n_features + 1), cumulative_variance_full, marker='o', 
         linewidth=2, markersize=4, color='darkblue')
ax2.axhline(y=0.80, color='green', linestyle='--', linewidth=1.5, label='80% variance')
ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=1.5, label='90% variance')
ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% variance')
ax2.set_xlabel('Nombre de Composantes Principales', fontsize=12)
ax2.set_ylabel('Variance Cumulée', fontsize=12)
ax2.set_title('Variance Cumulée en Fonction du Nombre de PC', fontsize=14, fontweight='bold')
ax2.set_xlim(0, n_features + 1)
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig("output/graphiques/pca_variance_cumulee.png", dpi=150, bbox_inches='tight')
print("✅ Graphique variance cumulée sauvegardé : pca_variance_cumulee.png")
plt.close()

# ============================================
# 4. SCREE PLOT (Variance par composante)
# ============================================
print("\n📈 Génération du Scree Plot...")

plt.figure(figsize=(10, 6))
plt.bar(range(1, n_components + 1), variance_ratio, alpha=0.7, color='steelblue', label='Variance par PC')
plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', color='red', linewidth=2, label='Variance cumulée')
plt.xlabel('Composantes Principales', fontsize=12)
plt.ylabel('Variance Expliquée', fontsize=12)
plt.title('Scree Plot - Variance Expliquée par Composante', fontsize=14, fontweight='bold')
plt.xticks(range(1, n_components + 1))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/graphiques/pca_scree_plot.png", dpi=150, bbox_inches='tight')
print("✅ Scree Plot sauvegardé : pca_scree_plot.png")
plt.close()

# ============================================
# 5. VISUALISATION 2D (PC1 vs PC2)
# ============================================
print("\n📈 Génération de la projection 2D (PC1 vs PC2)...")

plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50, c='steelblue', edgecolors='k', linewidth=0.5)
plt.xlabel(f"PC1 ({variance_ratio[0]*100:.1f}% de variance)", fontsize=12)
plt.ylabel(f"PC2 ({variance_ratio[1]*100:.1f}% de variance)", fontsize=12)
plt.title("Projection PCA 2D (PC1 vs PC2)", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("output/graphiques/pca_projection_2d.png", dpi=150, bbox_inches='tight')
print("✅ Projection 2D sauvegardée : pca_projection_2d.png")
plt.close()

# ============================================
# 6. VISUALISATION 3D STATIQUE (Matplotlib)
# ============================================
print("\n📈 Génération de la projection 3D statique...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
           alpha=0.6, s=50, c='steelblue', edgecolors='k', linewidth=0.5)

ax.set_xlabel(f"PC1 ({variance_ratio[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({variance_ratio[1]*100:.1f}%)", fontsize=11)
ax.set_zlabel(f"PC3 ({variance_ratio[2]*100:.1f}%)", fontsize=11)
ax.set_title("Projection PCA 3D", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("output/graphiques/pca_projection_3d.png", dpi=150, bbox_inches='tight')
print("✅ Projection 3D statique sauvegardée : pca_projection_3d.png")
plt.close()

# ============================================
# 7. VISUALISATION 3D INTERACTIVE (Plotly)
# ============================================
print("\n📈 Génération de la projection 3D INTERACTIVE...")

# Créer un DataFrame pour Plotly
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PC3': X_pca[:, 2],
    'Client_ID': range(len(X_pca))
})

# Créer le scatter 3D interactif avec Plotly Express
fig = px.scatter_3d(
    pca_df, 
    x='PC1', 
    y='PC2', 
    z='PC3',
    color='PC1',  # Coloration par PC1
    color_continuous_scale='Turbo',
    opacity=0.7,
    hover_data={'Client_ID': True, 'PC1': ':.2f', 'PC2': ':.2f', 'PC3': ':.2f'},
    labels={
        'PC1': f'PC1 ({variance_ratio[0]*100:.1f}%)',
        'PC2': f'PC2 ({variance_ratio[1]*100:.1f}%)',
        'PC3': f'PC3 ({variance_ratio[2]*100:.1f}%)'
    },
    title=f"Projection PCA 3D Interactive<br><sub>Variance capturée: {cumulative_variance[-1]*100:.1f}%</sub>"
)

# Améliorer le style
fig.update_traces(
    marker=dict(size=4, line=dict(width=0.5, color='white')),
    hovertemplate='<b>Client %{customdata[0]}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
)

fig.update_layout(
    width=1200,
    height=900,
    scene=dict(
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        xaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        yaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        zaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white")
    ),
    font=dict(size=12)
)

# Sauvegarder en HTML
fig.write_html("output/graphiques/pca_projection_3d_interactive.html")
print("✅ Projection 3D interactive sauvegardée : pca_projection_3d_interactive.html")
print("   💡 Ouvrez ce fichier dans votre navigateur pour interagir !")

# Afficher (optionnel - commentez si vous ne voulez pas ouvrir automatiquement)
# fig.show()

# ============================================
# 8. INTERPRÉTATION DES COMPOSANTES
# ============================================
print("\n" + "="*60)
print("🔍 INTERPRÉTATION DES COMPOSANTES PRINCIPALES")
print("="*60)

# Créer un DataFrame avec les loadings (contributions des features)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=feature_names
)

print("\n📊 Contributions des features aux composantes principales (loadings) :")
print(loadings)

# Top 5 features par composante
print("\n🎯 Top 5 features contribuant le plus à chaque composante :\n")
for i in range(n_components):
    pc_name = f'PC{i+1}'
    top_features = loadings[pc_name].abs().sort_values(ascending=False).head(5)
    print(f"{pc_name} ({variance_ratio[i]*100:.1f}% de variance) :")
    for feat, val in top_features.items():
        sign = '+' if loadings.loc[feat, pc_name] > 0 else '-'
        print(f"   {sign} {feat:30s} : {abs(val):.3f}")
    print()

# ============================================
# 9. HEATMAP DES LOADINGS
# ============================================
print("📈 Génération de la heatmap des loadings...")

plt.figure(figsize=(10, 12))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title("Contributions des Features aux Composantes Principales", 
          fontsize=14, fontweight='bold')
plt.xlabel("Composantes Principales", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig("output/graphiques/pca_loadings_heatmap.png", dpi=150, bbox_inches='tight')
print("✅ Heatmap des loadings sauvegardée : pca_loadings_heatmap.png")
plt.close()

# ============================================
# 10. SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n" + "="*60)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*60)

# Sauvegarder les données projetées en PCA
X_pca_df = pd.DataFrame(
    X_pca, 
    columns=[f'PC{i+1}' for i in range(n_components)]
)
X_pca_df.to_csv("output/pca_transformed_data.csv", index=False, sep=';')
print("✅ Données PCA sauvegardées : pca_transformed_data.csv")

# Sauvegarder les loadings
loadings.to_csv("output/pca_loadings.csv", sep=';')
print("✅ Loadings sauvegardés : pca_loadings.csv")

# Sauvegarder le modèle PCA
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(pca, 'models/pca_model.pkl')
print("✅ Modèle PCA sauvegardé : pca_model.pkl")

# ============================================
# 11. RÉSUMÉ FINAL
# ============================================
print("\n" + "="*60)
print("📊 RÉSUMÉ FINAL")
print("="*60)

print(f"\n✅ Réduction de {len(feature_names)} features → {n_components} composantes principales")
print(f"✅ Variance totale capturée : {cumulative_variance[-1]*100:.2f}%")
print(f"\n📂 Fichiers générés :")
print(f"   1. pca_scree_plot.png")
print(f"   2. pca_projection_2d.png")
print(f"   3. pca_projection_3d.png (statique)")
print(f"   4. pca_projection_3d_interactive.html (INTERACTIF 🎯)")
print(f"   5. pca_loadings_heatmap.png")
print(f"   6. pca_transformed_data.csv")
print(f"   7. pca_loadings.csv")
print(f"   8. pca_model.pkl")

print("\n" + "="*60)
print("✅ PCA TERMINÉE AVEC SUCCÈS !")
print("="*60)
print("\n🎯 INTERPRÉTATION DES RÉSULTATS :")
print("   • Analyser les loadings pour comprendre ce que représentent PC1, PC2, PC3")
print("   • PC1 capture souvent le 'pouvoir d'achat' (Income, Total_Spend)")
print("   • PC2 capture souvent la 'fidélité/engagement' (Recency, Web visits)")
print("\n🚀 PROCHAINE ÉTAPE : K-Means clustering sur les données PCA !")
print("\n💡 Ouvrez pca_projection_3d_interactive.html pour explorer vos données en 3D !")

print("\n" + "="*60)
print("🎯 CHOIX DU NOMBRE DE COMPOSANTES")
print("="*60)

# Recommandation automatique (80% de variance)
n_components_recommended = np.argmax(cumulative_variance_full >= 0.80) + 1
print(f"\n💡 Recommandation : {n_components_recommended} composantes pour 80% de variance")

# Pour la visualisation, on garde 3 PC
n_components = 3
print(f"📊 Pour les visualisations 3D : {n_components} composantes")

# Appliquer la PCA finale avec 3 composantes
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"\n✅ PCA finale appliquée avec {n_components} composantes")
print(f"📊 Dimensions après PCA : {X_pca.shape}")
print(f"🎯 Variance capturée : {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# Variance expliquée par les 3 PC choisis
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)