import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configuration graphique
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
print("="*60)
print("🚀 RUSH 4 - SEGMENTATION CLIENTS")
print("="*60)

df = pd.read_csv("output/clean_marketing_data.csv", sep=';')
print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

# ============================================
# 1. VÉRIFICATION ET NETTOYAGE
# ============================================
print("="*60)
print("🧹 PHASE 1 : VÉRIFICATION DES DONNÉES")
print("="*60)

print(f"\n📊 Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"📊 Doublons : {df.duplicated().sum()}")

# ============================================
# 2. FEATURE ENGINEERING - MAPPINGS SIMPLES
# ============================================
print("\n" + "="*60)
print("🔧 PHASE 2 : CRÉATION DES VARIABLES")
print("="*60)

# --- 1. is_couple (basé sur Marital_Status numérique) ---
print("\n1️⃣ Création de 'is_couple'")
couple_mapping = {
    1: 0,  # Single → Célibataire
    2: 1,  # Married → Couple
    3: 1,  # Together → Couple
    4: 0,  # Divorced → Célibataire
    5: 0,  # Alone → Célibataire
    6: 0   # Widow → Célibataire
}
df['is_couple'] = df['Marital_Status'].map(couple_mapping)
print(f"   ✅ En couple : {df['is_couple'].sum()} ({df['is_couple'].mean()*100:.1f}%)")

# --- 2. family_size ---
print("\n2️⃣ Création de 'family_size'")
df['family_size'] = 1 + df['is_couple'] + df['Total_Children']
print(f"   ✅ Taille moyenne du foyer : {df['family_size'].mean():.2f} personnes")

# --- 3. is_parent ---
print("\n3️⃣ Création de 'is_parent'")
df['is_parent'] = (df['Total_Children'] > 0).astype(int)
print(f"   ✅ Parents : {df['is_parent'].sum()} ({df['is_parent'].mean()*100:.1f}%)")

# --- 4. education2_encoded (basé sur Education numérique) ---
print("\n4️⃣ Création de 'education2_encoded'")
education_simple_mapping = {
    1: 0,  # Basic → undergrad
    2: 0,  # 2n Cycle → undergrad
    3: 1,  # Graduation → grad
    4: 2,  # Master → postgrad
    5: 2   # PhD → postgrad
}
df['education2_encoded'] = df['Education'].map(education_simple_mapping)
print(f"   ✅ Distribution : {df['education2_encoded'].value_counts().sort_index().to_dict()}")

# ============================================
# 3. VARIABLES BUSINESS SUPPLÉMENTAIRES
# ============================================
print("\n" + "="*60)
print("💡 PHASE 3 : VARIABLES BUSINESS")
print("="*60)

print("\n5️⃣ Création de 'Avg_Purchase_Value'")
df['Avg_Purchase_Value'] = df['Total_Spend'] / (df['Total_Purchases'] + 1)
print(f"   ✅ Valeur moyenne d'achat : {df['Avg_Purchase_Value'].mean():.2f}€")

print("\n6️⃣ Création de 'Web_Preference'")
df['Web_Preference'] = df['Num_Web_Purchases'] / (df['Total_Purchases'] + 1)
print(f"   ✅ Proportion achats web : {df['Web_Preference'].mean()*100:.1f}%")

print("\n7️⃣ Création de 'Total_Accepted_Campaigns'")
campaign_cols = ['Accepted_Cmp_1', 'Accepted_Cmp_2', 'Accepted_Cmp_3', 
                 'Accepted_Cmp_4', 'Accepted_Cmp_5', 'Accepted_Last_Cmp']
df['Total_Accepted_Campaigns'] = df[campaign_cols].sum(axis=1)
print(f"   ✅ Moyenne campagnes acceptées : {df['Total_Accepted_Campaigns'].mean():.2f}")

print("\n8️⃣ Création de 'Deal_Seeker'")
df['Deal_Seeker'] = (df['Num_Deals_Purchases'] > df['Num_Deals_Purchases'].median()).astype(int)
print(f"   ✅ Chasseurs de promos : {df['Deal_Seeker'].mean()*100:.1f}%")

print("\n8️⃣ Création de 'Web_Preference")
df['Web_Preference'] = df['Num_Web_Purchases'] / (df['Total_Purchases'] + 1)
print(f"   ✅ Chasseurs de promos : {df['Deal_Seeker'].mean()*100:.1f}%")



# ============================================
# 4. SAUVEGARDER
# ============================================
print("\n" + "="*60)
print("💾 SAUVEGARDE")
print("="*60)

output_file = "output/enriched_marketing_data.csv"
df.to_csv(output_file, index=False, sep=';')
print(f"✅ Dataset enrichi sauvegardé : {output_file}")
print(f"📊 Dimensions finales : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ============================================
# 5. VISUALISATIONS
# ============================================
print("\n" + "="*60)
print("📊 PHASE 4 : VISUALISATIONS")
print("="*60)

# Graphique 1 : Distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 10))
variables_to_plot = ['Age', 'Income', 'Total_Spend', 'Days_Since_Last_Purchase', 
                     'Total_Children', 'Days_Since_Join', 'Web_Preference' ]

for i, var in enumerate(variables_to_plot):
    ax = axes[i // 3, i % 3]
    sns.histplot(df[var], kde=True, ax=ax, color='steelblue', bins=30)
    ax.set_title(f"Distribution de {var}", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("output/distributions_variables_cles.png", dpi=150, bbox_inches='tight')
print("✅ Graphique 1 sauvegardé")
plt.close()

# Graphique 2 : Revenu vs Dépenses
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df, x='Income', y='Total_Spend', hue='education2_encoded', 
                alpha=0.6, s=50, palette='Set2')
plt.title("Revenu vs Dépenses Totales", fontsize=14, fontweight='bold')
plt.xlabel("Revenu (€)", fontsize=12)
plt.ylabel("Dépenses (€)", fontsize=12)
plt.savefig("output/revenu_vs_depenses.png", dpi=150, bbox_inches='tight')
print("✅ Graphique 2 sauvegardé")
plt.close()

# Graphique 3 : Matrice de corrélation
print("\n🔥 Matrice de corrélation...")
numeric_features = ['Income', 'Age', 'Total_Spend', 'Days_Since_Last_Purchase', 
                    'Days_Since_Join', 'Total_Purchases', 'Avg_Purchase_Value',
                    'Spend_Income_Ratio', 'NumWebVisitsMonth', 'Web_Preference',
                    'Total_Children', 'family_size', 'is_parent', 'is_couple', 
                    'education2_encoded', 'Deal_Seeker']

corr_matrix = df[numeric_features].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, vmin=-1, vmax=1, center=0)
plt.title("Matrice de Corrélation", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("output/matrice_correlation.png", dpi=150, bbox_inches='tight')
print("✅ Graphique 3 sauvegardé")
plt.close()

# ============================================
# 6. INSIGHTS CLÉS
# ============================================
print("\n" + "="*60)
print("💡 INSIGHTS CLÉS")
print("="*60)

print(f"\n📊 Âge moyen : {df['Age'].mean():.1f} ans")
print(f"📊 Revenu moyen : {df['Income'].mean():.0f}€")
print(f"📊 Dépenses moyennes : {df['Total_Spend'].mean():.0f}€")
print(f"📊 Parents : {df['is_parent'].mean()*100:.1f}%")
print(f"📊 En couple : {df['is_couple'].mean()*100:.1f}%")
print(f"📊 Taux acceptation campagnes : {df['Accepted_Last_Cmp'].mean()*100:.1f}%")
print(f"📊 Proportion d'achats en ligne : {df['Accepted_Last_Cmp'].mean()*100:.1f}%")
