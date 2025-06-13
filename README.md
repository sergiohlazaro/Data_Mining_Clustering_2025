# 🧠 Data_Mining_Clustering

---

## 📌 Objetivo

Aplicar y evaluar tres enfoques de agrupamiento:

1. **Clustering Particional** → K-Means  
2. **Clustering Jerárquico** → Aglomerativo con enlace Ward  
3. **Clustering Probabilista** → Gaussian Mixture Models (GMM)

---

## 📁 Estructura del proyecto
# ├── clustering/
# │ ├── partitioned.py # K-Means clustering
# │ ├── hierarchical.py # Clustering jerárquico
# │ ├── probabilistic.py # Clustering GMM
# │
# ├── evaluation/
# │ └── compare_clustering.py # Comparación de resultados
# │
# ├── data/
# │ ├── mlb_teams.csv # Dataset original
# │ └── mlb_teams_cleaned.csv # Dataset normalizado y limpio
# │ └── mlb_teams_scaled.csv
# │
# ├── figures/ # Visualizaciones generadas
# │
# ├── preparation/
# │ ├── phases.txt
# │ └── clean_columns.py # Elimina columnas no relevantes
# │
# ├── informe.md / informe.pdf # Informe con resultados
# └── presentacion.pdf # Diapositivas para exposición

---

## 🛠️ Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

# 🚀 Ejecución
```bash
python preparation/preprocessing.py
python preparation/clean_columns.py

python clustering/partitioned.py      # K-Means
python clustering/hierarchical.py     # Jerárquico
python clustering/probabilistic.py    # GMM

python evaluation/compare_clustering.py
```

