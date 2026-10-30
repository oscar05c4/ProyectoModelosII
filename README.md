# ProyectoModelosII

Proyecto de análisis y predicción del nivel de uso problemático de Internet
(`sii`) para la competencia **Child Mind Institute - Problematic Internet Use**.

## Requisitos

- Python 3.10 o superior.
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
python -m pip install kagglehub numpy pandas scipy scikit-learn joblib matplotlib
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

Si se ejecutan análisis adicionales de reducción no lineal con UMAP, instalar
también:

```powershell
python -m pip install umap-learn
```

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
│   └── Desarrollo.ipynb
├── src/
│   └── utils_piu.py
├── report/
│   └── reporte.md
├── .gitignore
└── README.md
```

### Archivos principales

- `notebooks/Carga_Datos.ipynb`: inicia sesión en Kaggle y descarga la
  competencia. La variable `path` contiene la ruta local devuelta por
  `kagglehub`; no se debe reemplazar por una ruta absoluta personal.
- `notebooks/Desarrollo.ipynb`: carga los datos, analiza las muestras sin
  etiqueta, prepara las variables, entrena y compara los modelos.
- `src/utils_piu.py`: contiene las clases personalizadas utilizadas por el
  notebook de desarrollo:
  - `BlockMissingIndicator`.
  - `OrdinalThresholdRegressor`.
- `report/reporte.md`: espacio destinado a la documentación del informe.

En `Desarrollo.ipynb` existe una celda comentada con `%%writefile` que muestra
cómo crear `utils_piu.py` en un entorno como Google Colab, donde el archivo
puede no existir todavía. En este repositorio el módulo fue creado de forma
permanente en `src/utils_piu.py`, por lo que esa celda no necesita ejecutarse.

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
