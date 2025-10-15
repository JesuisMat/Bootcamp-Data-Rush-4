import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement
df = pd.read_csv("output/clean_marketing_data.csv", sep=';')

# Inspection initiale
print("=" * 50)
print("APERÇU DU DATASET")
print("=" * 50)
print(df.head())
print("\n" + "=" * 50)
print("INFORMATIONS GÉNÉRALES")
print("=" * 50)
print(df.info())
print("\n" + "=" * 50)
print("VALEURS MANQUANTES")
print("=" * 50)
print(df.isnull().sum())
print("\n" + "=" * 50)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 50)
print(df.describe())
