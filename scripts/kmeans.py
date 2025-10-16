# ============================================
# SCRIPT 5 : K-MEANS CLUSTERING (K-Means++)
# Fichier : 05_kmeans_clustering.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import os

# Configuration graphique
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("🎯 K-MEANS CLUSTERING (avec K-Means++)")
print("="*60)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("\n📂 Chargement des données...")

# Données PCA (pour le clustering - RECOMMANDÉ)
X_pca_df = pd.read_csv("output/pca_transformed_data.csv", sep=';')
X_pca = X_pca_df.values

# Données standardisées (pour l'analyse post-clustering)
X_scaled_df = pd.read_csv("output/scaled_features_for_clustering.csv", sep=';')

# Dataset enrichi original (pour les analyses détaillées)
df_enriched = pd.read_csv("output/enriched_marketing_data.csv", sep=';')

print(f"✅ Données PCA : {X_pca.shape}")
print(f"✅ Données standardisées : {X_scaled_df.shape}")
print(f"✅ Dataset enrichi : {df_enriched.shape}")

print("\n💡 Workflow appliqué : PCA → K-Means++ → Méthode du Coude")
print("   • PCA : Réduction de dimension + décorrélation des features")
print("   • K-Means++ : Initialisation optimisée des centroïdes (Arthur & Vassilvitskii, 2007)")
print("   • Méthode du coude : Détermination du K optimal via l'inertie")

# ============================================
# 2. MÉTHODE DU COUDE (ELBOW METHOD) + SILHOUETTE
# ============================================
print("\n" + "="*60)
print("📊 MÉTHODE DU COUDE - Détermination du nombre optimal de clusters")
print("="*60)

# Tester différents nombres de clusters (à partir de 1 pour le graphique)
k_range = range(1, 11)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []

print("\n🔄 Test de K clusters de 1 à 10 avec K-Means++...")
print("\n   K  | Inertie  | Silhouette | Davies-Bouldin")
print("   " + "-"*50)

for k in k_range:
    kmeans = KMeans(
        n_clusters=k, 
        init='k-means++',
        n_init=20,
        max_iter=500,
        random_state=42
    )
    kmeans.fit(X_pca)
    
    inertias.append(kmeans.inertia_)
    
    if k >= 2:
        silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(X_pca, kmeans.labels_))
        print(f"   {k:2d} | {kmeans.inertia_:8.2f} | {silhouette_scores[-1]:10.3f} | {davies_bouldin_scores[-1]:14.3f}")
    else:
        silhouette_scores.append(None)
        davies_bouldin_scores.append(None)
        print(f"   {k:2d} | {kmeans.inertia_:8.2f} | {'N/A':>10s} | {'N/A':>14s}")

# ============================================
# 3. VISUALISATIONS DES MÉTRIQUES
# ============================================
print("\n📈 Génération des graphiques d'évaluation...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graphique 1 : Méthode du coude (Inertie) - AVEC K=1 ✅
axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertie (WCSS)', fontsize=12)
axes[0].set_title('Méthode du Coude\n(chercher le point de cassure)', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].set_xticks(k_range)  # ✅ Affiche 1, 2, 3, ..., 10
axes[0].set_xlim(0.5, 10.5)  # ✅ Élargit pour voir K=1

# Graphique 2 : Silhouette Score (commence à K=2)
k_range_valid = list(range(2, 11))
silhouette_valid = [s for s in silhouette_scores if s is not None]
axes[1].plot(k_range_valid, silhouette_valid, marker='o', linewidth=2, markersize=8, color='green')
best_silhouette_k = k_range_valid[np.argmax(silhouette_valid)]
axes[1].axvline(best_silhouette_k, color='red', linestyle='--', linewidth=1.5, label=f'Meilleur K={best_silhouette_k}')
axes[1].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score\n(plus élevé = clusters cohérents)', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].set_xticks(k_range_valid)
axes[1].legend()

# Graphique 3 : Davies-Bouldin Index (commence à K=2)
davies_valid = [d for d in davies_bouldin_scores if d is not None]
axes[2].plot(k_range_valid, davies_valid, marker='o', linewidth=2, markersize=8, color='red')
best_davies_k = k_range_valid[np.argmin(davies_valid)]
axes[2].axvline(best_davies_k, color='green', linestyle='--', linewidth=1.5, label=f'Meilleur K={best_davies_k}')
axes[2].set_xlabel('Nombre de Clusters (K)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[2].set_title('Davies-Bouldin Index\n(plus bas = clusters séparés)', fontsize=14, fontweight='bold')
axes[2].grid(alpha=0.3)
axes[2].set_xticks(k_range_valid)
axes[2].legend()

plt.tight_layout()
os.makedirs('output/graphiques', exist_ok=True)
plt.savefig("output/graphiques/kmeans/kmeans_evaluation_metrics.png", dpi=150, bbox_inches='tight')
print("✅ Métriques d'évaluation sauvegardées : kmeans_evaluation_metrics.png")
plt.close()

# ============================================
# 4. RECOMMANDATION DU NOMBRE OPTIMAL DE CLUSTERS
# ============================================
print("\n" + "="*60)
print("🎯 RECOMMANDATION DU NOMBRE OPTIMAL DE CLUSTERS")
print("="*60)

# Meilleur K selon Silhouette
best_k_silhouette = k_range_valid[np.argmax(silhouette_valid)]
print(f"\n💡 Meilleur K selon Silhouette Score : {best_k_silhouette}")
print(f"   → Score : {max(silhouette_valid):.3f}")

# Meilleur K selon Davies-Bouldin
best_k_davies = k_range_valid[np.argmin(davies_valid)]
print(f"\n💡 Meilleur K selon Davies-Bouldin Index : {best_k_davies}")
print(f"   → Index : {min(davies_valid):.3f}")

# Calcul automatique du coude
inertias_valid = inertias[1:]  # Exclure K=1
diff1 = np.diff(inertias_valid)
diff2 = np.diff(diff1)
elbow_k = k_range_valid[np.argmax(diff2) + 1]
print(f"\n💡 Coude détecté automatiquement à K={elbow_k}")

# Recommandation finale
if best_k_silhouette == elbow_k or abs(best_k_silhouette - elbow_k) <= 1:
    optimal_k = elbow_k
    reason = "Méthode du coude"

print(f"\n🎯 K OPTIMAL CHOISI : {optimal_k}")
print(f"   Raison : {reason}")
print(f"\n📊 Métriques pour K={optimal_k} :")
idx = optimal_k - 2
print(f"   • Inertie : {inertias[optimal_k]:.2f}")
print(f"   • Silhouette Score : {silhouette_valid[idx]:.3f}")
print(f"   • Davies-Bouldin Index : {davies_valid[idx]:.3f}")

# ============================================
# 5. APPLICATION DU K-MEANS FINAL
# ============================================
print("\n" + "="*60)
print(f"🔧 APPLICATION DE K-MEANS++ AVEC K={optimal_k}")
print("="*60)

kmeans_final = KMeans(
    n_clusters=3, 
    init='k-means++',
    n_init=50,
    max_iter=1000,
    random_state=42,
    verbose=0
)

print("\n🔄 Entraînement en cours...")
clusters = kmeans_final.fit_predict(X_pca)

df_enriched['Cluster'] = clusters
X_pca_df['Cluster'] = clusters

print(f"\n✅ K-Means++ appliqué avec succès")
print(f"📊 Nombre d'itérations : {kmeans_final.n_iter_}")
print(f"📊 Inertie finale : {kmeans_final.inertia_:.2f}")
print(f"📊 Silhouette Score : {silhouette_score(X_pca, clusters):.3f}")

# Distribution des clusters
print("\n📊 Distribution des clients par cluster :")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print("\n   Cluster | Nombre | Pourcentage")
print("   " + "-"*35)
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(clusters)) * 100
    print(f"   {cluster_id:7d} | {count:6d} | {percentage:5.1f}%")

# Équilibre
min_size = cluster_counts.min()
max_size = cluster_counts.max()
balance = min_size / max_size

print(f"\n📊 Équilibre des clusters :")
print(f"   • Plus petit : {min_size} | Plus grand : {max_size}")
print(f"   • Ratio : {balance:.2f}")
print(f"   {'⚠️ Déséquilibré' if balance < 0.3 else '✅ Équilibré'}")

# ============================================
# 6. VISUALISATION 2D
# ============================================
print("\n📈 Génération visualisation 2D...")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
           marker='X', s=300, c='red', edgecolors='black', linewidth=2, label='Centroïdes')
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title(f'K-Means++ (K={optimal_k}) - 2D', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/graphiques/kmeans/kmeans_clusters_2d.png", dpi=150, bbox_inches='tight')
print("✅ Clusters 2D sauvegardés")
plt.close()

# ============================================
# 7. VISUALISATION 3D INTERACTIVE
# ============================================
print("\n📈 Génération visualisation 3D interactive...")

X_pca_df['Cluster'] = X_pca_df['Cluster'].astype(str)

fig = px.scatter_3d(
    X_pca_df, x='PC1', y='PC2', z='PC3', color='Cluster',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    opacity=0.7, title=f'K-Means++ (K={optimal_k}) - 3D Interactive'
)
fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='white')))
fig.update_layout(width=1200, height=900)
fig.write_html("output/graphiques/kmeans/kmeans_clusters_3d_interactive.html")
print("✅ Clusters 3D interactifs sauvegardés")

# ============================================
# 8. SAUVEGARDE
# ============================================
print("\n💾 Sauvegarde...")

df_enriched.to_csv("output/data_with_clusters.csv", index=False, sep=';')
os.makedirs('models', exist_ok=True)

import joblib
joblib.dump(kmeans_final, 'models/kmeans_model.pkl')

centroids_df = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
)
centroids_df.to_csv("output/kmeans_centroids.csv", index=False, sep=';')

print("✅ Tous les fichiers sauvegardés")

# ============================================
# 9. RÉSUMÉ
# ============================================
print("\n" + "="*60)
print("✅ K-MEANS++ TERMINÉ !")
print("="*60)
print(f"\n🎯 K optimal : {optimal_k}")
print(f"📊 Silhouette : {silhouette_score(X_pca, clusters):.3f}")
print(f"\n🚀 Prochaine étape : python scripts/06_cluster_analysis.py")