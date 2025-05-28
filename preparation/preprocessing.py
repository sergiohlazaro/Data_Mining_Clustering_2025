import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Cargar dataset
df = pd.read_csv("data/mlb_teams.csv") 

# Seleccionar variables numéricas
X = df.select_dtypes(include=['float64', 'int64'])

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Guardar dataset normalizado para uso posterior
df_scaled.to_csv("data/mlb_teams_scaled.csv", index=False)

# Crear carpeta figures si no existe
os.makedirs("figures", exist_ok=True)

# Histograma
df_scaled.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.suptitle('Distribución de variables normalizadas', y=1.02)
plt.savefig("figures/histograms.png")
plt.show()

# Diagrama de dispersión simple
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1])
plt.xlabel(df_scaled.columns[0])
plt.ylabel(df_scaled.columns[1])
plt.title('Scatter plot entre dos variables')
plt.grid(True)
plt.savefig("figures/scatterplot.png")
plt.show()
