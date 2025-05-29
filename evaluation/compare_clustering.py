import pandas as pd
import matplotlib.pyplot as plt

# === 1. Resultados manuales cargados aquí ===
# Puedes copiar directamente los valores calculados anteriormente

data = {
    "k": list(range(2, 11)),
    "KMeans_Silhouette":     [0.6167, 0.1942, 0.1847, 0.1968, 0.1758, 0.1640, 0.1311, 0.1276, 0.1412],
    "Hierarchical_Silhouette": [0.6167, 0.1874, 0.1993, 0.1668, 0.1417, 0.1333, 0.1287, 0.1228, 0.1096],
    "GMM_Silhouette":        [0.4436, 0.0402, -0.0123, 0.0178, -0.0060, -0.0135, -0.0644, -0.0739, 0.0014],
    "GMM_BIC":               [-2185.47, -17512.10, -15184.42, -13240.97, -11264.97, -8115.93, -3069.49, -191.44, -438.04]
}

df = pd.DataFrame(data)

# === 2. Tabla resumen ===
print("\nComparativa de Silhouette:")
print(df[["k", "KMeans_Silhouette", "Hierarchical_Silhouette", "GMM_Silhouette"]])

# === 3. Gráfico comparativo ===
plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["KMeans_Silhouette"], label="K-Means", marker="o")
plt.plot(df["k"], df["Hierarchical_Silhouette"], label="Jerárquico", marker="s")
plt.plot(df["k"], df["GMM_Silhouette"], label="GMM", marker="^")
plt.title("Comparación de Silhouette Score")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/compare_silhouette.png")

# === 4. BIC opcional (solo GMM) ===
plt.figure(figsize=(6, 4))
plt.plot(df["k"], df["GMM_BIC"], marker="o", color="steelblue")
plt.title("BIC en GMM")
plt.xlabel("Número de clusters (k)")
plt.ylabel("BIC")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/gmm_bic.png")
print("\n")