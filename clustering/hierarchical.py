import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# === Configuración ===
DATA_PATH = "data/mlb_teams_cleaned.csv"
FIGURE_DIR = "figures/hierarchical/"
os.makedirs(FIGURE_DIR, exist_ok=True)

# === 1. Cargar datos ===
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)  # Eliminar posibles NaN
X = df.values

# === 2. Dendrograma ===
linked = linkage(X, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5, no_labels=True)
plt.title("Dendrograma (Ward linkage)")
plt.xlabel("Observaciones")
plt.ylabel("Distancia")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}dendrograma.png")
plt.show()

# === 3. Evaluar distintos k ===
range_k = range(2, 11)
silhouettes = []

for k in range_k:
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouettes.append(score)

    # Gráfico 2D simple (usamos primeras dos variables)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set2", legend="full")
    plt.title(f"Clustering jerárquico con k={k}")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}clusters_k{k}.png")
    plt.close()

# === 4. Gráfico de evaluación ===
plt.figure(figsize=(8, 4))
plt.plot(range_k, silhouettes, marker='o', color='darkorange')
plt.title("Silhouette score por número de clusters (jerárquico)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}evaluacion_silhouette.png")
plt.show()

# === 5. Tabla resumen ===
print("\nResumen de Silhouette scores:")
for k, s in zip(range_k, silhouettes):
    print(f"k={k}: Silhouette={s:.4f}")
print("\n")