# ProyectoModelosII

Proyecto de análisis y predicción del nivel de uso problemático de Internet
(`sii`) para la competencia **Child Mind Institute - Problematic Internet Use**.

## Requisitos

- Python 3.13.1.
- VS Code con las extensiones **Python** y **Jupyter**, o un entorno equivalente
  para ejecutar notebooks.
- Una cuenta de Kaggle.
- Acceso aceptado a la competencia y sus condiciones de uso.

## Instalación

Se recomienda crear un entorno virtual desde la raíz del repositorio:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Librerías utilizadas:

- `kagglehub`: autenticación y descarga de los archivos de la competencia.
- `numpy`: operaciones numéricas.
- `pandas`: lectura y manipulación de datos.
- `scipy`: pruebas estadísticas e intervalos de confianza.
- `scikit-learn`: preprocesamiento, imputación, validación cruzada,
  selección de hiperparámetros y modelos.
- `joblib`: caché del preprocesamiento y serialización de modelos.
- `matplotlib`: generación de figuras.
- `umap-learn`: reducción de dimensión UMAP.
- `pyarrow`: lectura del archivo de actigrafía en formato Parquet.
- `ipykernel`: registro del entorno virtual como kernel de Jupyter.

Las versiones exactas utilizadas y reproducibles están fijadas en
`requirements.txt`.

## Autenticación y descarga de Kaggle

La descarga **no funcionará solamente instalando `kagglehub`**. Antes de
ejecutar el notebook de carga:

1. Crear o iniciar sesión en una cuenta de Kaggle.
2. Entrar a la página de la competencia
   [Child Mind Institute - Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).
3. Leer y aceptar las condiciones y reglas de la competencia. Kaggle no
   permite descargar sus datos si no se han aceptado previamente.
4. Generar un token de Kaggle desde la configuración de la cuenta. Según la
   versión de Kaggle, esto puede ser un token/API key descargable o un token
   que se introduce durante el inicio de sesión.
5. Ejecutar la celda de autenticación en `Carga_Datos.ipynb`:

```python
import kagglehub

kagglehub.login()
path = kagglehub.competition_download(
    "child-mind-institute-problematic-internet-use"
)
print("Ruta descargada:", path)
```

`kagglehub.login()` puede solicitar el token de forma interactiva. También se
puede configurar la autenticación siguiendo la documentación oficial de
`kagglehub`. **No subir tokens, claves API ni archivos de credenciales a
GitHub.**

Los datos se guardan en la caché local de Kaggle de cada usuario. Por eso el
proyecto no depende de una ruta como `C:\Users\...\` y no se deben copiar esas
rutas absolutas al código.

## Estructura del proyecto

```text
ProyectoModelosII/
├── notebooks/
│   ├── Carga_Datos.ipynb
│   ├── Desarrollo.ipynb
│   ├── results/          # tablas CSV y resultados de las mallas
│   └── figures/          # figuras del informe 
├── src/
│   └── utils_piu.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Archivos principales

- `notebooks/Carga_Datos.ipynb`: inicia sesión en Kaggle y descarga la
  competencia. La variable `path` contiene la ruta local devuelta por
  `kagglehub`; no se debe reemplazar por una ruta absoluta personal.
- `notebooks/Desarrollo.ipynb`: notebook principal con todo el desarrollo del
  Entregable II, en este orden:
  1. Carga y verificación de datos.
  2. Separación del conjunto de test y esquema de validación.
  3. Análisis del sesgo de selección de las muestras sin etiqueta.
  4. Preprocesamiento (imputación KNN, escalado, indicadores de bloque) y
     umbrales ordinales, con verificación de no fuga de información.
  5. Modelo base y auditoría de predicciones por clase.
  6. Búsqueda de hiperparámetros de cinco modelos (Ridge, KNN, Random Forest,
     MLP y SVR), con intervalos de confianza.
  7. Tratamiento del desbalance (pesos por clase) y comparación con
     clasificación directa.
  8. Reducción de dimensión: análisis individual de variables, PCA y UMAP.
  9. Experimento de actigrafía (exclusión frente a imputación).
  10. Evaluación final en test, matriz de confusión y tabla consolidada de
      entrenamiento, validación y test.
  11. Discusión y conclusiones.
- `src/utils_piu.py`: contiene las clases personalizadas utilizadas por el
  notebook de desarrollo:
  - `BlockMissingIndicator`.
  - `OrdinalThresholdRegressor`.

## Orden de ejecución

1. Abrir la carpeta raíz del repositorio en VS Code.
2. Seleccionar como kernel de Jupyter el intérprete de `.venv`.
3. Abrir `notebooks/Carga_Datos.ipynb`.
4. Ejecutar la celda de autenticación y descarga.
5. Confirmar que se imprimió la ruta descargada y que se aceptaron las
   condiciones de la competencia.
6. Abrir `notebooks/Desarrollo.ipynb`.
7. Ejecutar sus celdas en orden.

El notebook de desarrollo vuelve a obtener la ruta con
`kagglehub.competition_download(...)`, por lo que también puede ejecutarse
después de una reinicialización del kernel, siempre que la sesión de Kaggle
esté configurada.

La importación del módulo se realiza sin rutas específicas del equipo:

```python
from pathlib import Path
import sys

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "src" / "utils_piu.py").exists():
    ROOT = ROOT.parent

if not (ROOT / "src" / "utils_piu.py").exists():
    raise FileNotFoundError("No se encontró src/utils_piu.py.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import utils_piu
```

## Archivos generados

El desarrollo puede generar modelos `.joblib`, tablas `.csv`, figuras,
resultados y cachés de `joblib`. Estos artefactos están incluidos en
`.gitignore` para evitar subir datos, resultados temporales o archivos
dependientes del equipo.

Los archivos de la competencia, especialmente `train.csv`, `test.csv`,
`series_train.parquet` y `series_test.parquet`, deben descargarse mediante
Kaggle y no deben subirse al repositorio.

## Resumen del proyecto

Predicción del Severity Impairment Index (`sii`, ordinal de 0 a 3) a partir de
biomarcadores físicos, mediciones fisiológicas y datos contextuales del Healthy
Brain Network (Child Mind Institute). Se comparan cinco familias de modelos
(Ridge, KNN, Random Forest, MLP y SVR) con regresión y umbrales ordinales que
maximizan el QWK, y se evalúan PCA y UMAP como reducción de dimensión.


## Resultados principales

| Configuración | QWK validación (CV 5×3) | QWK test |
|---|---|---|
| SVR sin reducción (modelo seleccionado) | 0.456 [0.417, 0.494] | 0.444 [0.374, 0.513] |
| SVR + PCA (k = 11) | 0.440 [0.395, 0.484] | 0.450 [0.378, 0.524] |
| SVR + UMAP (3 comp.) | 0.371 [0.299, 0.443] | 0.439 [0.371, 0.507] |

Tablas completas en `notebooks/results/` y figuras en `notebooks/figures/`.

## Reproducibilidad

- Semilla global `SEED = 42`. UMAP usa `random_state=SEED` y
  `compatibility_layout=True`.
- Versiones usadas: Python 3.13.1, kagglehub 1.0.3, NumPy 2.2.5,
  pandas 2.3.2, SciPy 1.17.1, scikit-learn 1.8.0, joblib 1.5.3,
  Matplotlib 3.11.2, umap-learn 0.6.0, pyarrow 21.0.0 e ipykernel 7.4.0.
  Todas están fijadas en `requirements.txt`.

### Dos formas de ejecutar el notebook

1. **Rápida (recomendada para revisar):** con la carpeta `notebooks/results/` del
   repositorio, las celdas de entrenamiento imprimen "ya existe, se omite" y
   cargan los resultados guardados, por lo que el notebook se ejecuta en pocos
   minutos.
2. **Completa:** borrar `notebooks/results/`, `grid_*.csv`, `best_*.joblib` y
   `notebooks/cache_prep/` y ejecutar todo. Las figuras se regeneran en
   `notebooks/figures/`.

Los archivos se escriben relativos al directorio del notebook
(`notebooks/results/`, `notebooks/figures/`).

## Datos de actigrafía

`series_train.parquet` pesa varios GB y solo se lee en la sección 9 del
notebook (experimento de actigrafía). El resto del notebook usa únicamente
`train.csv`. Los agregados calculados se guardan en
`notebooks/results/actigraphy_agg_tr_v2.csv`, así que la lectura se hace una sola vez.

## Solución de problemas

- **403 o "acepta las reglas" al descargar:** aceptar las condiciones de la
  competencia en la web de Kaggle.
- **`ModuleNotFoundError: src`:** abrir la carpeta raíz del repositorio en VS Code
  (el notebook busca `src/utils_piu.py` hacia arriba desde su carpeta).
- **Modelos `.joblib` antiguos:** se regeneran al ejecutar las mallas.
- **Advertencias `Pandas4Warning`:** son informativas. Usar la versión de pandas
  de `requirements.txt`.

## Autores

Xiomara Betancur Isaza y Oscar Albeiro Blandón Agudelo.
Universidad de Antioquia, Modelos y Simulación de Sistemas II.