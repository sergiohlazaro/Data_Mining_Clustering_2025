import pandas as pd
import os

# Leer el dataset escalado
df_scaled = pd.read_csv("data/mlb_teams_scaled.csv")

# Comprobar que las columnas existen antes de eliminarlas
cols_a_eliminar = ['rownames', 'year']
cols_presentes = [col for col in cols_a_eliminar if col in df_scaled.columns]

# Eliminar columnas no informativas
df_limpio = df_scaled.drop(columns=cols_presentes)

# Guardar nuevo archivo limpio
output_path = "data/mlb_teams_cleaned.csv"
df_limpio.to_csv(output_path, index=False)

print(f"Columnas eliminadas: {cols_presentes}")
print(f"Dataset limpio guardado en: {output_path}")
print(f"Columnas finales: {list(df_limpio.columns)}")
