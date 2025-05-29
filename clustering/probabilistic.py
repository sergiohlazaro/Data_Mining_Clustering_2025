import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os

# === Configuración ===
DATA_PATH = "data/mlb_teams_cleaned.csv"
FIGURE_DIR = "figures/gmm/"
os.makedirs(FIGURE_DIR, exist_ok=True)

# === 1. Cargar datos ===
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
X = df.values

# === 2. Evaluación para varios valores de k ===
range_k = range(2, 11)
bics = []
silhouettes = []

for k in range_k:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)

    bics.append(gmm.bic(X))
    silhouettes.append(silhouette_score(X, labels))

    # Visualización 2D (usando 2 primeras variables)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set2', legend="full")
    plt.title(f"GMM con k={k}")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}clusters_k{k}.png")
    plt.close()

# === 3. Gráfica BIC y Silhouette ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range_k, bics, marker='o', color='steelblue')
plt.title("BIC por número de componentes")
plt.xlabel("Número de clusters (k)")
plt.ylabel("BIC")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range_k, silhouettes, marker='o', color='darkorange')
plt.title("Silhouette score (GMM)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}evaluacion_gmm.png")
plt.show()

# === 4. Imprimir resultados ===
print("\nResumen de evaluación por k:")
for k, bic, sil in zip(range_k, bics, silhouettes):
    print(f"k={k}: BIC={bic:.2f}, Silhouette={sil:.4f}")
print("\n")