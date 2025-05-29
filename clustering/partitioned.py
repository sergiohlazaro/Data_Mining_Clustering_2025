import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# === CONFIGURACIÓN ===
DATA_PATH = "data/mlb_teams_cleaned.csv"
FIGURE_DIR = "figures/kmeans/"
os.makedirs(FIGURE_DIR, exist_ok=True)

# === 1. Cargar datos ===
df = pd.read_csv(DATA_PATH)

# Eliminar filas con NaN (importante para clustering)
df.dropna(inplace=True)

X = df.values

# === 2. Probar varios valores de k ===
range_k = range(2, 11)
inertias = []
silhouettes = []

for k in range_k:
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = model.fit_predict(X)

    inertias.append(model.inertia_)
    silhouette = silhouette_score(X, labels)
    silhouettes.append(silhouette)

    # Visualización 2D usando las 2 primeras variables
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set2", legend="full")
    plt.title(f"K-Means con k={k}")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}clusters_k{k}.png")
    plt.close()

# === 3. Graficar evaluación ===
plt.figure(figsize=(8, 4))
plt.plot(range_k, inertias, marker='o', label='Inercia (SSW)')
plt.plot(range_k, silhouettes, marker='s', label='Silhouette')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Métrica')
plt.title('Evaluación de K-Means')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}evaluacion_kmeans.png")
plt.show()

# === 4. Mostrar tabla resumen ===
resultados = pd.DataFrame({
    "k": list(range_k),
    "Inercia": inertias,
    "Silhouette": silhouettes
})
print("\nResumen de métricas por valor de k:")
print(resultados.to_string(index=False))
print("\n")

