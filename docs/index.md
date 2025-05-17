# Temario Detallado con Librerías: Plan de Carrera Técnico
## 1. Python

### 1. Nivel Básico
1. [ ] **1. Sintaxis y Tipos de Datos**
   - **Descripción:** Fundamentos del lenguaje, cómo escribir código Python válido, variables, operadores y los tipos de datos fundamentales (números, cadenas, booleanos, listas, tuplas, diccionarios, conjuntos).
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/01_sintaxis_tipos.ipynb)  
   - Librerías: *No específicas* (uso de sintaxis nativa)  
2. [ ] **2. Control de Flujo**
   - **Descripción:** Estructuras para controlar el orden de ejecución del código, incluyendo condicionales (`if/elif/else`), bucles (`for` y `while`), y sentencias como `break`, `continue` y `pass`.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/02_control_flujo.ipynb)  
   - Librerías: *No específicas*  
3. [ ] **3. Funciones**
   - **Descripción:** Definición y uso de bloques de código reutilizables para realizar tareas específicas, incluyendo parámetros (posicionales, por defecto, `*args`, `**kwargs`), valores de retorno y ámbito de variables (scope). `functools` es un poco avanzado para lo básico, pero introducir `partial` podría ser útil más adelante.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/03_funciones.ipynb)  
   - Librerías: *No específicas* (uso de `functools`)  
4. [ ] **4. Módulos y Paquetes**
   - **Descripción:** Organización del código en archivos (`.py`) y directorios para crear proyectos estructurados y reutilizables, y cómo importar funcionalidades de la librería estándar y de terceros.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/04_modulos_paquetes.ipynb)  
   - Librerías: *No específicas*  
5. [ ] **5. Manejo de Errores**
   - **Descripción:** Uso de bloques `try-except-else-finally` para capturar y gestionar excepciones (errores) que ocurren durante la ejecución del programa, y cómo lanzar excepciones propias.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/05_errores.ipynb)  
   - Librerías: *No específicas*  
6. [ ] **6. Comprensión de Listas y Generadores**
   - **Descripción:** Sintaxis concisa y eficiente para crear listas (list comprehensions), diccionarios y conjuntos, y una introducción a las expresiones generadoras para procesar datos de manera eficiente en memoria.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/06_comprension_listas_generadores.ipynb)  
   - Librerías: *No específicas*  
7. [ ] **7. Funciones Lambda**
   - **Descripción:** Creación de pequeñas funciones anónimas de una sola expresión, útiles para operaciones sencillas o como argumentos de funciones de orden superior (como `map`, `filter`).
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/07_lambda.ipynb)  
   - Librerías: *No específicas*  
8. [ ] **8. Manejo de Archivos**
   - **Descripción:** Lectura y escritura de datos en archivos de texto y binarios, y manipulación básica de archivos y directorios utilizando librerías como `os`, `shutil` y el enfoque moderno de `pathlib`.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/08_manejo_archivos.ipynb)  
   - Librerías: `os`, `shutil`, `pathlib`  
9. [ ] **9. Entorno Virtual**
   - **Descripción:** Creación de entornos aislados para proyectos Python para gestionar dependencias específicas del proyecto y evitar conflictos de versiones, utilizando herramientas como `venv` o `virtualenv`.
   - [Notebook](../1._Python/1._Nivel_Básico/notebooks/09_entorno_virtual.ipynb)  
   - Herramientas: `venv`, `virtualenv`, 
10. [ ] **10. Gestión de Dependencias**
    - **Descripción:** Uso de `pip` para instalar, actualizar y desinstalar paquetes, y cómo definir las dependencias de un proyecto en un archivo `requirements.txt` para facilitar la replicación del entorno.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/10_gestion_dependencias.ipynb)  
    - Herramientas: `pip`, `requirements.txt`, `pipenv`
11. [ ] **11. Documentación y Estilo de Código**
    - **Descripción:** Importancia de escribir código legible y mantenible siguiendo guías de estilo (PEP 8), y herramientas para generar documentación (docstrings, `pydoc`). `Sphinx` es potente pero quizás más para intermedio/avanzado para la generación. `pylint` es un linter, bueno mencionarlo aquí.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/11_documentacion_estilo.ipynb)  
    - Herramientas: `pydoc`, `Sphinx`, `pylint`  
12. [ ] **12. Version Control**
    - **Descripción:** Uso de sistemas de control de versiones, principalmente Git, para rastrear cambios en el código, colaborar en equipo, gestionar ramas y fusionar cambios. Introducción a plataformas como GitHub/GitLab.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/12_version_control.ipynb)  
    - Herramientas: Git, GitHub, GitLab  
13. [ ] **13. Pruebas Unitarias Básicas**
    - **Descripción:** Introducción a la escritura de pruebas para verificar el correcto funcionamiento de pequeñas unidades de código (funciones, métodos) de forma aislada, utilizando el módulo `unittest`.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/13_pruebas_unitarias.ipynb)  
    - Librerías: `unittest`  
14. [ ] **14. Tipado Estático**
    - **Descripción:** Uso de anotaciones de tipo (type hints) para mejorar la legibilidad del código, facilitar la detección temprana de errores con herramientas como `mypy`, y mejorar la experiencia de desarrollo.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/14_tipado_estatico.ipynb)  
    - Librerías: `typing`, `mypy`  
15. [ ] **15. Registro y Depuración (Logging & Debugging)**
    - **Descripción:** Técnicas y herramientas para registrar eventos importantes durante la ejecución de 
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/15_log_debug.ipynb)  
    - Librerías: `logging`, `pdb`  
16. [ ] **16. Argumentos de Línea de Comandos**
    - **Descripción:** Creación de scripts de Python que pueden aceptar argumentos y opciones desde la terminal, utilizando librerías como `argparse` o `click` (click es más moderno y amigable).
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/16_argumentos_linea.ipynb)  
    - Librerías: `argparse`, `click`  
17. [ ] **17. Testing de Integración Ligera**
    - **Descripción:** Pruebas que verifican la interacción entre varios componentes o módulos del sistema, por ejemplo, cómo una función interactúa con otra o con un mock de un servicio externo. `pytest` es excelente aquí, y `requests-mock` es específico pero ilustrativo.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/17_testing_integracion.ipynb)  
    - Librerías: `pytest`, `requests-mock`  
18. [ ] **18. Manejo de Datos**
    - **Descripción:** Introducción a librerías fundamentales para el análisis y manipulación de datos tabulares (`Pandas`), operaciones numéricas eficientes (`NumPy`) y lectura/escritura de archivos Excel (`openpyxl`).
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/18_manejo_datos.ipynb)  
    - Librerías: `pandas`, `numpy`, `openpyxl`  
19. [ ] **19. Visualización de Datos**
    - **Descripción:** Creación de gráficos y visualizaciones estáticas e interactivas para entender y comunicar patrones y resultados a partir de los datos, utilizando librerías como `matplotlib`, `seaborn` y `plotly`.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/19_visualizacion_datos.ipynb)  
    - Librerías: `matplotlib`, `seaborn`, `plotly`  
20. [ ] **20. APIs y WebScraping**
    - **Descripción:** Interacción con Interfaces de Programación de Aplicaciones (APIs) web para obtener y enviar datos (`requests`), y técnicas para extraer información de páginas web (`BeautifulSoup4`, `Scrapy`, `Selenium` para sitios dinámicos).
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/20_apis_webscraping.ipynb)  
    - Librerías: `requests`, `beautifulsoup4`, `scrapy`, `selenium`  
21. [ ] **21. Desarrollo Web Básico**
    - **Descripción:** Fundamentos para crear aplicaciones web simples y APIs RESTful utilizando microframeworks como `Flask` o `FastAPI`.
    - [Notebook](../1._Python/1._Nivel_Básico/notebooks/21_desarrollo_web.ipynb)  
    - Librerías: `flask`, `fastapi` 
22. [ ] **22. Trabajo con Fechas y Horas**
    - **Descripción:** Manipulación de fechas, horas, deltas de tiempo y zonas horarias.
    - Librerías: `datetime`, `pytz`
23. [ ] **23. Expresiones Regulares (Básico)**
    - **Descripción:** Introducción al uso de expresiones regulares para búsqueda y manipulación de patrones en texto.
    - Librerías: `re`

### 2. Nivel Intermedio
1. [ ] **1. POO (Programación Orientada a Objetos)**
   - **Descripción:** POO BASICO: Introducción a los conceptos de clases, objetos, atributos, métodos, encapsulamiento y herencia simple. Es fundamental y debería estar antes de temas como APIs o Desarrollo Web.
   POO AVANZADO: herencia múltiple, polimorfismo, métodos especiales (dunder methods como `__init__`, `__str__`), propiedades, decoradores de clase/método, clases abstractas (`abc`) y `dataclasses` para estructuras de datos concisas. 
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/01_poo.ipynb)  
   - Librerías: *No específicas* (`abc`, `dataclasses`)  
2. [ ] **2. Decoradores**
   - **Descripción:** Creación y uso de decoradores para modificar o mejorar funciones o métodos de manera declarativa, entendiendo cómo funcionan y utilizando `functools.wraps`.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/02_decoradores.ipynb)  
   - Librerías: `functools`  
3. [ ] **3. Context Managers**
   - **Descripción:** Uso de la sentencia `with` para gestionar recursos (como archivos, conexiones de red/BD) asegurando su correcta inicialización y liberación. Creación de context managers propios con clases o `contextlib`.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/03_context_managers.ipynb)  
   - Librerías: `contextlib`  
4. [ ] **4. Generadores e Iteradores**
   - **Descripción:** Profundización en el protocolo de iteración de Python, creación de iteradores personalizados y uso avanzado de generadores (funciones con `yield`) para procesamiento de datos eficiente en memoria y flujos de datos.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/04_generadores_iteradores.ipynb)  
   - Librerías: *No específicas*  
5. [ ] **5. Manejo de Archivos Avanzado**
   - **Descripción:** Trabajo con formatos de archivo estructurados comunes como JSON (intercambio de datos web), CSV (datos tabulares) y Pickle (serialización de objetos Python).
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/05_archivos_Avanzado.ipynb)  
   - Librerías: `json`, `pickle`, `csv`  
6. [ ] **6. Anotaciones de Tipo Avanzadas**
   - **Descripción:** Uso de características más avanzadas del sistema de tipos de Python, como `Generic[T]`, `TypeVar`, `Protocol`, `Callable` y otras herramientas de `typing` y `typing_extensions` para crear código más robusto y expresivo.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/06_anotaciones_tipo_avanzadas.ipynb)  
   - Librerías: `typing_extensions`  
7. [ ] **7. Async I/O y Redes**
   - **Descripción:** Introducción a la programación asíncrona con `async` y `await` para manejar operaciones de entrada/salida (red, archivos) de forma no bloqueante, mejorando el rendimiento de aplicaciones con alta concurrencia, usando `asyncio` y `aiohttp` para clientes/servidores HTTP asíncronos.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/07_async_redes.ipynb)  
   - Librerías: `asyncio`, `aiohttp`  
8. [ ] **8. Profiling y Performance**
   - **Descripción:** Adaptación de aplicaciones para soportar múltiples idiomas y configuraciones regionales utilizando herramientas como el módulo `gettext`.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/08_performance.ipynb)  
   - Librerías: `cProfile`, `memory_profiler`  
9. [ ] **9. Internacionalización (i18n)**
   - **Descripción:** Herramientas y técnicas para identificar cuellos de botella en el rendimiento del código (CPU y memoria) y optimizar su ejecución.
   - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/09_i18n.ipynb)  
   - Librerías: `gettext`  
10. [ ] **10. Patrones de Diseño**
    - **Descripción:** Introducción a patrones de diseño creacionales, estructurales y de comportamiento comunes (ej. Factory, Singleton, Observer, Strategy, Decorator) y cómo implementarlos en Python para escribir código más modular, flexible y reutilizable.
    - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/10_patrones_diseno.ipynb)
    - Librerías: *No específicas*  
11. [ ] **11. Interacción con Bases de Datos (SQL y NoSQL)**
     - **Descripción:** Conexión y ejecución de consultas a bases de datos relacionales (usando `sqlite3` para empezar, y luego `psycopg2` para PostgreSQL o `mysql-connector-python` para MySQL) y una introducción a bases de datos NoSQL (e.g., MongoDB con `pymongo`). Introducción a ORMs básicos (ej. SQLAlchemy Core o PonyORM).
     - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/10_patrones_diseno.ipynb)
     - Librerías: `sqlite3`, `psycopg2`, `mysql-connector-python`, `pymongo`, `SQLAlchemy`
12. [ ] **12. Programación Funcional (Avanzado)**
     - **Descripción:** Profundizar en conceptos de programación funcional más allá de lambdas y comprensiones: funciones de orden superior, clausuras (closures), currying, composición de funciones. Uso avanzado de `itertools` y `functools`.
     - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/10_patrones_diseno.ipynb)
     - Librerías: `itertools`, `functools`
13. [ ] **13. Seguridad Básica en Aplicaciones**
     - **Descripción:** Conciencia sobre vulnerabilidades comunes (ej. Inyección SQL, XSS - si se ve desarrollo web), manejo seguro de contraseñas (hashing), validación de entradas.
    - [Notebook](../1._Python/2._Nivel_Intermedio/notebooks/10_patrones_diseno.ipynb)
     - Librerías: `hashlib`, `secrets`

### 3. Nivel Avanzado
1. [ ] **1. Metaprogramación y Data Model**
   - **Descripción:** Técnicas que permiten a un programa tratar a otros programas (o a sí mismo) como datos. Incluye la creación y manipulación de clases y funciones en tiempo de ejecución, descriptores, metaclases y personalización del modelo de datos de Python (métodos especiales).
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/01_metaprogramacion_data_model.ipynb)  
   - Librerías: *No específicas* (`inspect`, `dataclasses`)  
2. [ ] **2. Concurrencia y Paralelismo**
   - **Descripción:** Estrategias para ejecutar múltiples tareas "simultáneamente": `threading` para concurrencia (I/O-bound, limitado por el GIL), `multiprocessing` para paralelismo real (CPU-bound), y profundización en `asyncio` para concurrencia cooperativa. Sincronización entre hilos/procesos. 
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/02_concurrencia_paralelismo.ipynb)  
   - Librerías: `threading`, `multiprocessing`, `asyncio`  
3. [ ] **3. Empaquetado y Distribución**
   - **Descripción:** Creación de paquetes distribuibles de Python para ser instalados vía `pip` y publicados en PyPI (Python Package Index) utilizando herramientas como `setuptools` y `wheel`.
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/03_empaquetado_distribucion.ipynb)  
   - Librerías: `setuptools`, `wheel`, `pip`  
4. [ ] **4. Testing y Calidad de Código**
   - **Descripción:** Estrategias avanzadas de testing como mocking y patching (`unittest.mock`, `pytest-mock`), TDD (Test-Driven Development), BDD (Behavior-Driven Development con `behave` o `pytest-bdd`). Uso avanzado de linters (`flake8`, `pylint`) y formateadores automáticos (`black`, `isort`).
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/04_testing_calidad_codigo.ipynb)  
   - Librerías: `pytest`, `unittest`, `flake8`, `pylint`, `black`  
5. [ ] **5. CI/CD para Proyectos Python**
   - **Descripción:** Automatización de los procesos de integración, prueba y despliegue continuo de código utilizando herramientas y plataformas de CI/CD como GitHub Actions, GitLab CI. `tox` para automatizar pruebas en múltiples entornos Python.
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/05_cicd_proyectos.ipynb)  
   - Herramientas: GitHub Actions, GitLab CI, `tox`  
6. [ ] **6. Extensión en C/Cython**
   - **Descripción:** Escritura de módulos de extensión en C/C++ (usando la API C de Python o `pybind11`) o usando `Cython` para optimizar partes críticas de rendimiento o interactuar con librerías C/C++ existentes.
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/06_extension_c_cython.ipynb)  
   - Herramientas: `Cython`, `pybind11`  
7. [ ] **7. Embebido y Distribución Avanzada**
   - **Descripción:** Integrar el intérprete de Python en aplicaciones escritas en otros lenguajes. Herramientas avanzadas para la gestión de dependencias y construcción de proyectos (`Poetry`), y creación de ejecutables autocontenidos (`PyInstaller`, `Nuitka`) o instaladores (`pynsist`).
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/07_embebido_distribucion.ipynb)  
   - Librerías: `Poetry`, `pynsist`  
8. [ ] **8. Seguridad y Criptografía**
   - **Descripción:** Principios de seguridad en el desarrollo de software Python, OWASP Top 10 (si aplica a web). Uso de librerías para implementar funcionalidades criptográficas como hashing seguro, encriptación simétrica/asimétrica, firmas digitales (`cryptography`, `pyOpenSSL`).
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/08_seguridad_criptografia.ipynb)  
   - Librerías: `cryptography`, `pyOpenSSL`, `pycryptodome`
9. [ ] **9. Contenedorización y Orquestación**
   - **Descripción:** Creación de imágenes Docker para aplicaciones Python, buenas prácticas para Dockerfiles. Introducción a la orquestación de contenedores con Kubernetes (conceptos básicos).
   - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/09_contenedorizacion_y_orquestacion.ipynb)  
   - Herramientas: Docker, Kubernetes (kubectl)
10. [ ] **10 .Bases de Datos Avanzadas y ORMs (Avanzado)**
    - **Descripción:** Uso avanzado de ORMs como SQLAlchemy (manejo de relaciones complejas, migraciones con Alembic, optimización de queries) o Django ORM (si se cubre Django). Técnicas de optimización de bases de datos, indexing.
    - [Notebook](../1._Python/3._Nivel_Avanzado/notebooks/10_bases_de_datos_avanzadas.ipynb)  
    - Librerías: `SQLAlchemy`, `Alembic`, `Django ORM`

## 2. Machine Learning

### 1. Nivel Básico
1. [ ] **1. Introducción General al Machine Learning y Flujo de Trabajo Típico**
   - **Descripción:** Qué es el Machine Learning, tipos de problemas que resuelve, diferencias con la programación tradicional. Presentación de un flujo de trabajo estándar en un proyecto de ML (definición del problema, recolección de datos, preprocesamiento, entrenamiento, evaluación, despliegue).
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/01_introduccion_al_ML.ipynb)  
    *   Librerías: *Conceptual*
2. [ ] **2. Preprocesamiento de Datos (Básico)**
   - **Descripción:** Técnicas esenciales para preparar los datos antes de entrenar modelos: manejo de valores faltantes (imputación), codificación de variables categóricas (One-Hot Encoding, Label Encoding), escalado/normalización de características numéricas.
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/02_preprocesamiento_de_datos.ipynb)  
   - Librerías: `pandas`, `scikit-learn` (ej. `SimpleImputer`, `OneHotEncoder`, `StandardScaler`)
3. [ ] **3. Matemáticas Fundamentales para ML (Conceptual)**
   - **Descripción:** Revisión conceptual (sin profundizar en demostraciones) de los elementos de álgebra lineal (vectores, matrices), probabilidad y estadística básica (media, varianza, distribuciones), y cálculo (derivadas, gradientes) que sustentan muchos algoritmos de ML. El objetivo es la intuición, no la maestría matemática.
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/03_matematicas_fundamentales_para_ML.ipynb)  
   - Librerías: *Conceptual*, `numpy` para ejemplos.
4. [ ] **4. Aprendizaje Supervisado**
   - **Descripción:** Fundamentos del aprendizaje a partir de datos etiquetados. Introducción a problemas de clasificación (predecir categorías) y regresión (predecir valores continuos). Se exploran algoritmos básicos como Regresión Lineal/Logística, K-Vecinos Cercanos (KNN), Árboles de Decisión.
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/04_aprendizaje_supervisado.ipynb)  
   - Librerías: `scikit-learn`, `pandas`, `numpy`  
5. [ ] **5. Aprendizaje No Supervisado**
   - **Descripción:** Descubrimiento de patrones y estructura en datos no etiquetados. Introducción a algoritmos de clustering (agrupamiento, ej. K-Means) y reducción de dimensionalidad (simplificación de datos, ej. PCA).
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/05_aprendizaje_no_supervisado.ipynb)  
   - Librerías: `scikit-learn`, `numpy`  
6. [ ] **6. Validación de Modelos**
   - **Descripción:** Técnicas para evaluar el rendimiento de los modelos y asegurar su generalización a datos nuevos. Cubre división de datos (train/test/validation), validación cruzada (cross-validation) y métricas comunes (accuracy, precision, recall, F1-score para clasificación; MSE, RMSE, MAE para regresión). Introducción a los conceptos de sobreajuste (overfitting) y subajuste (underfitting).
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/06_validacion_modelos.ipynb)  
   - Librerías: `scikit-learn`  
7. [ ] **7. Visualización de Datos**
   - **Descripción:** Uso de herramientas de visualización para el Análisis Exploratorio de Datos (EDA) específico para ML, como la visualización de distribuciones de características, relaciones entre variables, resultados de clustering, o la representación de límites de decisión (si es posible de forma simple).
   - [Notebook](../2._Machine_Learning/1._Nivel_Básico/notebooks/07_visualizacion_datos.ipynb)  
   - Librerías: `matplotlib`, `seaborn`  


### 2. Nivel Intermedio
1. [ ] **1. Ingeniería de Características**
   - **Descripción:** Creación de nuevas características (features) a partir de las existentes o de conocimiento del dominio para mejorar el rendimiento del modelo. Incluye transformaciones, creación de términos de interacción, binning, y selección de características.
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/01_ingenieria_caracteristicas.ipynb)  
   - Librerías: `pandas`, `scikit-learn`  
2. [ ] **2. Modelos de Ensamblado**
   - **Descripción:** Técnicas que combinan las predicciones de múltiples modelos base para obtener un rendimiento superior. Incluye Bagging (Random Forests), Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM) y Stacking.
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/02_modelos_ensamblado.ipynb)  
   - Librerías: `scikit-learn`, `xgboost`, `lightgbm`  
3. [ ] **3. Optimización de Hiperparámetros**
   - **Descripción:** Métodos para encontrar la mejor configuración de los parámetros de un modelo que no se aprenden durante el entrenamiento (hiperparámetros). Incluye Grid Search, Random Search y enfoques más avanzados como la optimización bayesiana (ej. con Optuna).
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/03_optimizacion_hiperparametros.ipynb)  
   - Librerías: `scikit-learn`, `optuna`  
4. [ ] **4. Reinforcement Learning**
   - **Descripción:** Introducción a los conceptos del aprendizaje por refuerzo donde un agente aprende a tomar decisiones interactuando con un entorno para maximizar una recompensa. Conceptos de agente, entorno, estado, acción, recompensa. Algoritmos básicos como Q-Learning.
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/04_reinforcement_learning.ipynb)  
   - Librerías: `stable-baselines3`, `gymnasium`  
5. [ ] **5. Interpretabilidad y Fairness**
   - **Descripción:** Técnicas para entender por qué un modelo toma ciertas decisiones (interpretabilidad) y para evaluar y mitigar sesgos injustos en sus predicciones (fairness). Introducción a métodos como SHAP y LIME.
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/05_interpretabilidad_fairness.ipynb)  
   - Librerías: `SHAP`, `LIME`, `fairlearn`
6. [ ] **6. Manejo de Datos Desbalanceados**  
   - **Descripción:** Técnicas para abordar problemas de clasificación donde las clases no están representadas equitativamente, como oversampling (ej. SMOTE), undersampling, y uso de métricas apropiadas.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/06_manejo_datos_desbalanceados.ipynb)  
   - Librerías: `scikit-learn`, `imblearn`  

7. [ ] **7. Detección de Anomalías (Outlier Detection)**  
   - **Descripción:** Métodos para identificar puntos de datos que son significativamente diferentes del resto. Algoritmos como Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/07_deteccion_anomalias.ipynb)  
   - Librerías: `scikit-learn`  

8. [ ] **8. Pipelines de Scikit-learn**  
   - **Descripción:** Construcción de flujos de trabajo de preprocesamiento y modelado de forma encadenada y eficiente, facilitando la experimentación y el despliegue.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/08_pipelines_scikit_learn.ipynb)  
   - Librerías: `scikit-learn` (Pipeline)  

9. [ ] **9. Introducción a Sistemas de Recomendación**  
   - **Descripción:** Conceptos básicos de los sistemas que sugieren ítems a los usuarios. Filtrado colaborativo, filtrado basado en contenido.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/09_introduccion_sistemas_recomendacion.ipynb)  
   - Librerías: `scikit-learn` (para componentes), `surprise` (opcional, para un enfoque más dedicado)  


### 3. Nivel Avanzado
1. [ ] **1. Deep Learning Básico**  
   - **Descripción:** Fundamentos de las redes neuronales profundas. Perceptrón multicapa (MLP), funciones de activación, funciones de pérdida, optimizadores (ej. SGD, Adam), y el concepto de backpropagation.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/01_deep_learning_basico.ipynb)  
   - Librerías: `tensorflow` (con `keras`), `pytorch`  
2. [ ] **2. Redes Convolucionales (CNN)**  
   - **Descripción:** Arquitecturas de redes neuronales especializadas en el procesamiento de datos con estructura de rejilla (ej. imágenes). Capas convolucionales, pooling, arquitecturas comunes (LeNet, AlexNet, VGG conceptualmente). Aplicaciones en clasificación de imágenes.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/02_redes_convolucionales.ipynb)  
   - Librerías: `tensorflow` (con `keras`), `pytorch` (con `torchvision`)  
3. [ ] **3. Redes RNN y Transformers**  
   - **Descripción:** Arquitecturas para datos secuenciales (texto, series temporales). Redes Neuronales Recurrentes (RNNs), LSTMs, GRUs. Introducción a la arquitectura Transformer y el mecanismo de atención, base de modelos como BERT y GPT.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/03_redes_rnn_transformers.ipynb)  
   - Librerías: `tensorflow` (con `keras`), `pytorch`, `transformers` (Hugging Face)  
4. [ ] **4. MLOps y Despliegue**  
   - **Descripción:** Prácticas para llevar modelos de ML a producción de forma robusta y escalable. Empaquetado de modelos, creación de APIs de inferencia (ej. con FastAPI) y containerización con Docker.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/04_mlops_despliegue.ipynb)  
   - Librerías: `mlflow`, `docker`, `fastapi`, `bentoml`  
5. [ ] **5. Federated Learning y Privacidad Diferencial**  
   - **Descripción:** Técnicas avanzadas de privacidad en ML. Federated Learning para entrenar modelos en datos distribuidos sin centralizarlos. Privacidad Diferencial para añadir ruido y proteger la privacidad individual en los resultados del modelo.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/05_federated_learning_privacidad.ipynb)  
   - Librerías: `TensorFlow Federated`, `PySyft`, `OpenDP`  
6. [ ] **6. Modelos Generativos (GANs, VAEs)**  
   - **Descripción:** Redes neuronales capaces de generar nuevos datos similares a los datos de entrenamiento. Redes Generativas Antagónicas (GANs) y Autoencoders Variacionales (VAEs). Aplicaciones en generación de imágenes, texto, etc.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/06_modelos_generativos.ipynb)  
   - Librerías: `tensorflow` (con `keras`), `pytorch`  
7. [ ] **7. Optimización de Inferencia y Pruning**  
   - **Descripción:** Técnicas para hacer los modelos de DL más eficientes (rápidos y pequeños) para su despliegue, especialmente en dispositivos con recursos limitados. Incluye cuantización, poda (pruning) y destilación de conocimiento. Formato ONNX para interoperabilidad.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/07_optimizacion_inferencia_pruning.ipynb)  
   - Librerías: `onnx`, `TensorRT`, `TensorFlow Lite`, `PyTorch Mobile`  
8. [ ] **8. MLOps: Monitoreo y Versionado**  
   - **Descripción:** Ciclo de vida completo de MLOps. Versionado de datos y modelos (DVC), seguimiento de experimentos (MLflow), orquestación de pipelines (Kubeflow, Airflow) y monitoreo de modelos en producción (detección de data drift, model drift).  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/08_mlops_monitoreo_versionado.ipynb)  
   - Librerías: `MLflow`, `DVC`, `Kubeflow`, `Prometheus/Grafana`  
9. [ ] **9. NLP (Procesamiento de Lenguaje Natural)**  
   - **Descripción:** Aplicación de ML y DL para que las computadoras entiendan y procesen el lenguaje humano. Cobertura de preprocesamiento de texto, embeddings (Word2Vec, GloVe, FastText) y tareas como clasificación, análisis de sentimientos, modelado de tópicos, traducción automática y QA con Transformers.  
   - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/09_nlp_procesamiento_lenguaje.ipynb)  
   - Librerías: `nltk`, `spacy`, `gensim`, `transformers`  
10. [ ] **10. Visión por Computadora (Computer Vision)**  
    - **Descripción:** Técnicas de ML y DL para que las computadoras “vean” e interpreten imágenes y videos. Tareas como clasificación, detección de objetos, segmentación semántica e instanciada y generación de imágenes.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/10_vision_computadora.ipynb)  
    - Librerías: `opencv`, `pillow`, `tensorflow` (con `keras`), `pytorch` (con `torchvision`), `mmdetection`  
11. [ ] **11. Series Temporales (Time Series Analysis & Forecasting)**  
    - **Descripción:** Análisis y modelado de datos ordenados en el tiempo. Métodos estadísticos clásicos (ARIMA, SARIMA, ETS), modelos de ML (XGBoost) y DL (RNNs, LSTMs, Transformers) para predicción.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/11_series_temporales.ipynb)  
    - Librerías: `pandas`, `statsmodels`, `prophet`, `sktime`, `tensorflow`, `pytorch`  
12. [ ] **12. Explicabilidad y Fairness (Avanzado)**  
    - **Descripción:** Profundización en técnicas para explicar modelos complejos (especialmente de DL) y auditar, medir y mitigar sesgos algorítmicos. Contrafactuales, análisis de influencia y técnicas específicas para equidad.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/12_explicabilidad_fairness_avanzado.ipynb)  
    - Librerías: `shap`, `lime`, `eli5`, `AI Fairness 360`, `Captum`  
13. [ ] **13. AutoML (Automated Machine Learning)**  
    - **Descripción:** Herramientas y técnicas que automatizan el proceso de selección de modelos, ingeniería de características y optimización de hiperparámetros.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/13_automl.ipynb)  
    - Librerías/Herramientas: `TPOT`, `auto-sklearn`, `Google AutoML`, `Azure ML Studio`  
14. [ ] **14. Graph Neural Networks (GNNs) - Introducción**  
    - **Descripción:** Redes neuronales diseñadas para operar sobre datos estructurados como grafos, con aplicaciones en redes sociales, sistemas de recomendación y descubrimiento de fármacos.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/14_graph_neural_networks.ipynb)  
    - Librerías: `torch-geometric` (PyG), `dgl` (Deep Graph Library)  
15. [ ] **15. ML Escalable y Distribuido**  
    - **Descripción:** Técnicas para entrenar modelos en conjuntos de datos muy grandes (que no caben en memoria) o para acelerar el entrenamiento distribuyéndolo en múltiples GPUs o máquinas.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/15_ml_escalable_distribuido.ipynb)  
    - Librerías/Herramientas: `dask`, `spark-mllib`, `horovod`, `torch.distributed`, `tensorflow` (MirroredStrategy)  
16. [ ] **16. Inferencia Causal (Introducción)**  
    - **Descripción:** Métodos para ir más allá de la correlación y tratar de inferir relaciones de causa-efecto a partir de datos observacionales. Diferencia entre predicción e inferencia causal.  
    - [Notebook](../2._Machine_Learning/2._Nivel_Intermedio/notebooks/16_inferencia_causal.ipynb)  
    - Librerías/Conceptos: `DoWhy`, `CausalML`, conceptos de grafos causales  

## 3. Bases de Datos

### Fundamentos de Bases de Datos

1. [ ] **1. Introducción a las Bases de Datos**  
   - **Descripción:** Qué son las bases de datos, por qué se utilizan, ventajas sobre archivos planos. Tipos de bases de datos (Relacionales, NoSQL: Documentales, Clave-Valor, Orientadas a Columnas, Grafos, Series Temporales).  
   - [Notebook](../3._Bases_de_Datos/Fundamentos/notebooks/01_introduccion_bases_datos.ipynb)  
   - Librerías/Herramientas: Conceptual  

2. [ ] **2. Conceptos Clave de Bases de Datos Relacionales**  
   - **Descripción:** Tablas, filas (registros), columnas (campos), llaves primarias, llaves foráneas, índices básicos, restricciones (NOT NULL, UNIQUE, CHECK).  
   - [Notebook](../3._Bases_de_Datos/Fundamentos/notebooks/02_conceptos_clave_relacionales.ipynb)  
   - Librerías/Herramientas: `sqlite3` (cliente SQL), `psycopg2`, `mysql-connector-python`  

3. [ ] **3. Introducción a SQL (Lenguaje de Consulta Estructurado)**  
   - **Descripción:** Comandos DQL básicos (`SELECT`, `FROM`, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`), DML (`INSERT`, `UPDATE`, `DELETE`), DDL (`CREATE TABLE`, `ALTER TABLE`, `DROP TABLE`). Joins básicos (`INNER`, `LEFT`, `RIGHT`).  
   - [Notebook](../3._Bases_de_Datos/Fundamentos/notebooks/03_introduccion_sql.ipynb)  
   - Librerías: `sqlite3`, `psycopg2`, `mysql-connector-python`  

### Modelado
1. [ ] **1. Modelado SQL**
   - **Descripción:** Creación de esquemas para bases de datos relacionales. Incluye la identificación de entidades, atributos, relaciones (uno a uno, uno a muchos, muchos a muchos) y su representación mediante diagramas Entidad-Relación (ERD). Definición de tablas, columnas, tipos de datos, y restricciones. 
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/01_modelado_sql.ipynb)  
   - Librerías: *No específicas*  
2. [ ] **2. Normalización**
   - **Descripción:** Proceso de organizar los datos en una base de datos relacional para reducir la redundancia y mejorar la integridad de los datos. Cubre las formas normales (1NF, 2NF, 3NF, BCNF) y sus objetivos.
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/02_normalizacion.ipynb)  
   - Librerías: *No específicas* 
2. [ ] **2. Modelado NoSQL**
   - **Descripción:** Principios y técnicas para diseñar esquemas en bases de datos NoSQL, considerando los patrones de acceso, la necesidad de denormalización, y las características específicas de cada tipo (documental, clave-valor, etc.). Con `pymongo`, se enfoca en bases de datos documentales.
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/02_modelado_nosql.ipynb)  
   - Librerías: `pymongo`  
4. [ ] **4. Bases de Datos de Serie Temporal**
   - **Descripción:** Diseño de esquemas y consideraciones específicas para el almacenamiento y consulta eficiente de datos indexados por tiempo (métricas, logs, datos de sensores). Introducción a tecnologías como InfluxDB y TimescaleDB.
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/04_serie_temporal.ipynb)  
   - Tecnologías: InfluxDB, TimescaleDB  
5. [ ] **5. Bases de Datos en Grafos**
   -  **Descripción:** Modelado de datos como una red de nodos y relaciones (aristas), ideal para representar y consultar datos altamente interconectados (redes sociales, recomendaciones, detección de fraude).
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/05_bases_grafos.ipynb)  
   - Tecnologías: Neo4j, JanusGraph  


### SQL Avanzado
1. [ ] **1. Consultas Complejas**
   - **Descripción:** Uso avanzado de SQL para extraer información compleja: subconsultas (correlacionadas y no correlacionadas), Common Table Expressions (CTEs), funciones de ventana (window functions), JOINs avanzados (self-join, cross-join), operaciones PIVOT/UNPIVOT (si el SGBD lo soporta o mediante CASE).
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/01_consultas_complejas.ipynb)  
   - Librerías: *No específicas* (clientes: `psycopg2`, `mysql-connector-python`)  
2. [ ] **2. Optimización de Consultas**
   - **Descripción:** Técnicas para mejorar el rendimiento de las consultas SQL. Análisis de planes de ejecución (`EXPLAIN`), creación y uso efectivo de índices (B-tree, hash, GIN, GiST, etc.), reescritura de consultas, estadísticas de la base de datos.
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/02_optimizacion_consultas.ipynb)  
   - Herramientas: `EXPLAIN`, `pgcli`  
3. [ ] **3. Data Warehousing y OLAP**
   - **Descripción:** Conceptos de almacenamiento de datos para análisis (Data Warehouse). Diferencias entre OLTP y OLAP. Modelado dimensional (esquemas de estrella y copo de nieve). Operaciones OLAP (slice, dice, drill-down, roll-up). Introducción a plataformas de DWH.
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/03_data_warehousing_olap.ipynb)  
   - Herramientas: Snowflake, Amazon Redshift  
4. [ ] **4. ETL y Calidad de Datos**
   - **Descripción:** Procesos de Extracción, Transformación y Carga donde SQL juega un papel importante (especialmente en la 'T'). Técnicas de limpieza, validación y enriquecimiento de datos usando SQL. Introducción a herramientas de orquestación y calidad.
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/04_etl_calidad_datos.ipynb)  
   - Herramientas: Apache Airflow, Great Expectations  


### NoSQL Avanzado
1. [ ] **1. Modelado de Documentos**
   - **Descripción:** Patrones de diseño avanzados para bases de datos documentales (ej. MongoDB). Manejo de relaciones (embedding vs referencing), estrategias de indexación, consideraciones de rendimiento para diferentes patrones de consulta. Uso de ODMs (Object Document Mappers).
   - [Notebook](../3._Bases_de_Datos/NoSQL_Avanzado/notebooks/01_modelado_documentos.ipynb)  
   - Librerías: `pymongo`, `mongoengine`  
2. [ ] **2. Bases de Datos de Grafos**
   - **Descripción:** Interacción y consulta de bases de datos de grafos. Lenguajes de consulta como Cypher (para Neo4j) o Gremlin. Casos de uso avanzados y algoritmos de grafos.
   - [Notebook](../3._Bases_de_Datos/NoSQL_Avanzado/notebooks/02_nosql_grafos.ipynb)  
   - Librerías: `neo4j`, `py2neo`  
3. [ ] **3. Bases de Datos en Memoria**
   - **Descripción:** Uso de bases de datos que almacenan datos principalmente en RAM para alta velocidad (ej. Redis, Memcached). Casos de uso: caching, gestión de sesiones, contadores, colas. Estructuras de datos avanzadas en Redis. Persistencia.
   - [Notebook](../3._Bases_de_Datos/NoSQL_Avanzado/notebooks/03_bases_memoria.ipynb)  
   - Librerías: `redis-py`  


### Integración de Datos
1. [ ] **1. ETL y Pipelines**  
   - **Descripción:** Diseño e implementación de flujos de trabajo para extraer, transformar y cargar datos desde diversas fuentes (bases de datos, APIs, archivos) a destinos (data warehouses, data lakes, otras bases de datos). Herramientas de orquestación.  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/01_etl_pipelines.ipynb)  
   - Librerías: `apache-airflow`, `prefect`, `luigi`
2. [ ] **2. Data Warehousing**  
   - **Descripción:** Construcción y mantenimiento de Data Warehouses. Procesos de ingesta de datos, transformaciones para el modelo dimensional, carga de datos. `sqlalchemy` puede usarse para interactuar programáticamente con el DWH.  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/02_data_warehousing_integration.ipynb)  
   - Librerías: `sqlalchemy`
3. [ ] **3. Data Lakes**  
   - **Descripción:** Concepto, arquitectura, ventajas y desventajas. Almacenamiento de datos brutos y procesados. Herramientas y tecnologías asociadas (ej. S3, HDFS, Delta Lake).  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/03_data_lakes.ipynb)  
   - Tecnologías: AWS S3, Azure Data Lake Storage, Apache Hudi/Iceberg/Delta Lake
4. [ ] **4. Streaming de Datos / Procesamiento en Tiempo Real**  
   - **Descripción:** Introducción a la ingesta y procesamiento de flujos de datos continuos. Casos de uso.  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/04_streaming_datos.ipynb)  
   - Tecnologías: Apache Kafka, Apache Flink / Spark Streaming / Kafka Streams
5. [ ] **5. Formatos de Datos para Intercambio y Almacenamiento Eficiente**  
   - **Descripción:** Formatos como Avro, Parquet, ORC y sus ventajas en pipelines de datos (compresión, esquemas, rendimiento de lectura).  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/05_formatos_datos.ipynb)  
   - Tecnologías: Avro, Parquet, ORC
6. [ ] **6. Virtualización de Datos**  
   - **Descripción:** Concepto de acceder a datos de múltiples fuentes como si fuera una sola, sin moverlos físicamente.  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/06_virtualizacion_datos.ipynb)  
   - Tecnologías: Denodo, Presto/Trino  


### Administración
1. [ ] **1. Instalación y Configuración**
   - **Descripción:** Procedimientos para instalar y configurar servidores de bases de datos (ej. PostgreSQL, MySQL). Parámetros de configuración iniciales importantes. Gestión de servicios.
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/01_instalacion_configuracion.ipynb)  
   - Herramientas: `psycopg2`, `mysqlclient`  
2. [ ] **2. Backup y Recuperación**
   - **Descripción:** Estrategias y herramientas para realizar copias de seguridad (full, incremental, diferencial) y restaurar datos. Point-in-Time Recovery (PITR).
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/02_backup_recuperacion.ipynb)  
   - Herramientas: `pg_dump`  
3. [ ] **3. Replicación**
   - **Descripción:** Configuración de réplicas de solo lectura para escalabilidad de lectura y redundancia. Replicación en streaming (física) vs. lógica.
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/03_replicacion.ipynb)  
   - Herramientas: `repmgr`, configuración nativa de PostgreSQL  
4. [ ] **4. Seguridad y Roles**
   - **Descripción:** Gestión de usuarios, roles y permisos (GRANT, REVOKE). Métodos de autenticación. Seguridad a nivel de red (firewalls, SSL/TLS para conexiones). Encriptación de datos en reposo y en tránsito. Auditoría. 
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/04_seguridad_roles.ipynb)  
   - Conceptos: encriptación en reposo, gestión de roles  
5. [ ] **5. Replicación Multimaestro y Alta Disponibilidad**
   - **Descripción:** Configuración de sistemas donde múltiples nodos pueden aceptar escrituras. Estrategias para lograr alta disponibilidad y failover automático.
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/05_replicacion_multimaestro_ha.ipynb)  
   - Tecnologías: Galera Cluster, Patroni  


## 4. Cloud Computing (AWS)

### 1. Nivel Básico
1. [ ] **1. Conceptos de la Nube**
   - **Descripción:** Entiende qué es el cloud computing, sus modelos de servicio (IaaS, PaaS, SaaS), modelos de despliegue (público, privado, híbrido), beneficios clave y cómo se estructura la infraestructura global de AWS (Regiones, Zonas de Disponibilidad, Edge Locations).
   - [Notebook](../4._Cloud_Computing/1._Nivel_Básico/aws/notebooks/01_conceptos_nube.ipynb)  
   - Librerías: *No específicas*  
2.  [ ] **2. Introducción a la Consola de AWS y Servicios Fundamentales (EC2 y S3)**
    - **Descripción:** Familiarízate con la consola de administración de AWS, aprende a navegar por ella e introduce los conceptos básicos y casos de uso de Amazon EC2 (cómputo virtual) y Amazon S3 (almacenamiento de objetos).
    - *Podría ser un nuevo notebook o integrado en el de conceptos/IAM.*
    - Librerías: `awscli` (para exploración básica)
3.  [ ] **3. IAM (Identity and Access Management)**
    - **Descripción:** Aprende a gestionar de forma segura el acceso a los servicios y recursos de AWS. Cubre usuarios, grupos, roles, políticas y las mejores prácticas de seguridad como el principio de mínimo privilegio y MFA.
    - [Notebook](../4._Cloud_Computing/1._Nivel_Básico/aws/notebooks/03_iam.ipynb)
    - Librerías: `boto3`
4. [ ] **4. Fundamentos de Contenedores y Primeros Pasos con Amazon ECS/Fargate**
   - **Descripción:** Introduce los conceptos básicos de la contenerización (ej. Docker) y cómo desplegar y gestionar aplicaciones en contenedores de forma sencilla utilizando Amazon Elastic Container Service (ECS) y AWS Fargate.
   - [Notebook](../4._Cloud_Computing/1._Nivel_Básico/aws/notebooks/04_contenedores_nube.ipynb)  
   - Servicios: Amazon ECS, AWS Fargate  


### 2. Nivel Intermedio
1. [ ] **1. Compute y Networking**
   - **Descripción:** Explora en detalle Amazon EC2: tipos de instancias, imágenes (AMIs), volúmenes EBS, grupos de seguridad, y cómo escalar aplicaciones automáticamente con Auto Scaling Groups y distribuir el tráfico con Elastic Load Balancing (ALB, NLB).
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/01_compute_networking.ipynb)  
   - Librerías: `boto3`, `awscli`
   - Servicios: EC2 (Tipos de instancia, AMIs, EBS, Security Groups, Key Pairs, Auto Scaling Groups), ELB (ALB, NLB).
2.  [ ] **2. Networking Avanzado con VPC**
    - **Descripción:** Domina la creación y gestión de redes privadas virtuales (VPC) en AWS, incluyendo subredes, tablas de rutas, gateways (Internet, NAT), listas de control de acceso a la red (NACLs), VPC Endpoints y VPC Peering.
    - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/02_compute_networking_vpc.ipynb)
    - Librerías: `boto3`, `awscli`
    - Servicios: VPC (Subredes públicas/privadas, Tablas de rutas, Internet Gateway, NAT Gateway, NACLs, VPC Endpoints, Peering).
3. [ ] **3. Storage y Bases de Datos Gestionadas**
   - **Descripción:** Profundiza en los servicios de almacenamiento: S3 (clases, versionado, ciclo de vida), EBS (tipos, snapshots), EFS y Glacier. Explora bases de datos gestionadas: RDS para bases de datos relacionales (MySQL, PostgreSQL, etc.) y DynamoDB para NoSQL.
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/03_storage_bases_datos.ipynb)  
   - Librerías: `boto3`
   - Servicios: S3 (Clases de almacenamiento, Versionado, Ciclo de vida, Replicación), EBS (Tipos de volúmenes, Snapshots), EFS, Glacier, RDS (Motores, Multi-AZ, Read Replicas), DynamoDB.
4. [ ] **4. Orquestación con Kubernetes**
   - **Descripción:** Aprende a desplegar, gestionar y escalar aplicaciones en contenedores utilizando Kubernetes en AWS a través de Amazon Elastic Kubernetes Service (EKS).
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/04_orquestacion_kubernetes.ipynb)  
   - Servicios: Amazon EKS  
5. [ ] **5. CI/CD en AWS**
    - **Descripción:** Implementa pipelines de integración continua y entrega continua (CI/CD) utilizando el conjunto de herramientas de DevOps de AWS: CodeCommit (control de versiones), CodeBuild (compilación), CodeDeploy (despliegue) y CodePipeline (orquestación del pipeline).
    - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/04_cicd_aws.ipynb)  
    - Herramientas: CodePipeline, CodeBuild
6.  [ ] **6. Gestión de Costos y Optimización en AWS**
    - **Descripción:** Aprende a monitorizar, controlar y optimizar tus gastos en AWS utilizando herramientas como Cost Explorer y AWS Budgets, y aplicando buenas prácticas como el etiquetado de recursos.
    - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/05_gestion_costos_aws.ipynb) `(nombre de notebook sugerido)`
    - Herramientas: AWS Cost Explorer, AWS Budgets, Tagging, Trusted Advisor (pilar de costos).
7.  [ ] **7. AWS Well-Architected Framework**
    - **Descripción:** Comprende los pilares del AWS Well-Architected Framework (Excelencia Operativa, Seguridad, Fiabilidad, Eficiencia del Rendimiento, Optimización de Costos) para diseñar y operar sistemas robustos y eficientes en la nube.

### 3. Nivel Avanzado
1. [ ] **1. Serverless**
   - **Descripción:** Diseña y despliega aplicaciones sin servidor (serverless) utilizando AWS Lambda para el cómputo, API Gateway para la exposición de APIs, Step Functions para la orquestación de flujos de trabajo, y servicios de mensajería como SQS/SNS y EventBridge para arquitecturas basadas en eventos.
   - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/01_serverless.ipynb)  
   - Librerías: `boto3`, `chalice`
   - Servicios: AWS Lambda, API Gateway, Step Functions, DynamoDB (como backend común), EventBridge, SQS, SNS.
2. [ ] **2. Infraestructura como Código**
   - **Descripción:** Automatiza el aprovisionamiento y la gestión de tu infraestructura en AWS utilizando código. Compara y utiliza herramientas populares como AWS CloudFormation (nativo), AWS Cloud Development Kit (CDK) y Terraform (multi-nube).
   - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/02_infraestructura_codigo.ipynb)  
   - Herramientas: `aws-cdk`, `cloudformation`, `terraform`
3. [ ] **3. Monitorización y Seguridad**  
   - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/03_Monitorizacion_y_seguridad.ipynb)  
4.  [ ] **4. Seguridad Avanzada y Gobierno en AWS**
    - **Descripción:** Profundiza en las mejores prácticas y servicios de seguridad de AWS. Cubre la gestión avanzada de identidades (IAM), protección contra ataques (WAF, Shield), gestión de secretos y claves (KMS, Secrets Manager), detección de amenazas (GuardDuty), gestión de la postura de seguridad (Security Hub), cumplimiento (Config) y evaluación de vulnerabilidades (Inspector).
    - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/04_Seguridad_avanzada.ipynb)
    - Servicios: IAM (Roles Avanzados, Federación), AWS WAF, AWS Shield, AWS KMS, AWS Secrets Manager, Amazon GuardDuty, AWS Security Hub, AWS Config, Amazon Inspector.
5.  [ ] **5. Arquitecturas de Alta Disponibilidad y Recuperación ante Desastres (HA/DR)**
    - **Descripción:** Diseña arquitecturas resilientes que aseguren la alta disponibilidad y la continuidad del negocio. Aprende sobre estrategias de recuperación ante desastres (RTO/RPO), el uso de múltiples Zonas de Disponibilidad y Regiones, y servicios como Route 53 para failover y AWS Backup/DRS para la protección de datos.
    - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/05_HA_DR_aws.ipynb)
    - Servicios/Conceptos: Route 53 (failover, enrutamiento), Multi-AZ y Multi-Region, RTO/RPO, estrategias de backup y restauración, AWS Backup, Elastic Disaster Recovery (AWS DRS).
6.  [ ] **6. Patrones de Diseño en la Nube y Mejores Prácticas de Arquitectura**
    - **Descripción:** Explora patrones de diseño comunes para construir aplicaciones escalables, desacopladas y resilientes en AWS (ej. microservicios, event-driven, fan-out, circuit breaker), y revisa las mejores prácticas de arquitectura más allá del Well-Architected Framework.



## 5. Desarrollo Web con Python

### 1. Nivel Básico
1. [ ] **1. Fundamentos del Desarrollo Web**
    - **Descripción:** Repaso conceptual de cómo funciona la web: roles de HTML (estructura), CSS (presentación) y JavaScript (interactividad en el cliente). Entender la arquitectura cliente -servidor y el rol de Python en el backend.
    - [Notebook](../5._Desarrollo_Web_con_Python/1._Nivel_Básico/notebooks/01_fundamentos.)
2.  [ ] **2. Introducción a HTTP, Servidores Web y APIs**
    - **Descripción:** Comprende los fundamentos del protocolo HTTP (métodos, cabeceras, códigos de estado), cómo funciona un servidor web básico y la comunicación cliente -servidor. Realiza tus primeras peticiones HTTP con `requests`. Introduce el concepto de API.
    - [Notebook](../5._Desarrollo_Web_con_Python/1._Nivel_Básico/notebooks/01_introduccion_http_servidores.ipynb)
    - Librerías: `http.server`, `requests`.
3.  [ ] **3. Desarrollo con Microframeworks: Flask**
    - **Descripción:** Comienza a construir aplicaciones web con Flask. Aprende sobre rutas, manejo de peticiones (GET, POST), plantillas con Jinja2 para generar HTML dinámico y manejo básico de formularios.
    - [Notebook](../5._Desarrollo_Web_con_Python/1._Nivel_Básico/notebooks/02_flask_basico.ipynb)
    - Librerías: `Flask`, `Jinja2`.
4.  [ ] **4. Desarrollo con Microframeworks Modernos: FastAPI**
    - **Descripción:** Explora FastAPI para construir APIs de alto rendimiento. Aprende sobre rutas, validación de datos con Pydantic, la documentación automática de APIs (Swagger UI/ReDoc) y los fundamentos de la programación asíncrona en el contexto web.
    - [Notebook](../5._Desarrollo_Web_con_Python/1._Nivel_Básico/notebooks/03_fastapi_basico.ipynb)
    - Librerías: `FastAPI`, `Uvicorn`, `Pydantic`.

### 2. Nivel Intermedio
1.  [ ] **1. Frameworks Completos: Django - Fundamentos**
    - **Descripción:** Sumérgete en Django, un framework "con baterías incluidas". Aprende su estructura de proyectos y aplicaciones, el potente ORM de Django para interactuar con bases de datos, el sistema de migraciones y el panel de administración automático.
    - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/01_django_fundamentos.ipynb)
    - Librerías/Conceptos: `Django` (Proyectos, Apps, ORM, Migraciones, Admin).
2.  [ ] **2. Django - Vistas, Plantillas y Formularios**
    - **Descripción:** Profundiza en cómo Django maneja la lógica de negocio con vistas (basadas en funciones y clases), cómo renderiza contenido dinámico usando su sistema de plantillas y cómo gestiona la entrada de datos del usuario a través de formularios.
    - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/02_django_vistas_templates_forms.ipynb)
    - Librerías/Conceptos: `Django` (Vistas basadas en funciones y clases, Sistema de plantillas de Django, Formularios).
3.  [ ] **3. Django - Autenticación y Autorización**
    - **Descripción:** Implementa sistemas de autenticación (registro, inicio de sesión, cierre de sesión) y autorización (gestión de permisos y grupos) utilizando las robustas herramientas incorporadas en Django.
    - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/03_django_auth.ipynb)
    - Librerías/Conceptos: `Django` (Sistema de autenticación de usuarios, permisos, grupos).
4.  [ ] **4. Desarrollo de APIs REST con Django REST Framework (DRF)**
    - **Descripción:** Construye APIs RESTful robustas y escalables sobre Django utilizando Django REST Framework. Aprende sobre serializadores, vistas (ViewSets), routers, y autenticación/permisos para APIs.
    - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/04_apis_rest_drf.ipynb)
    - Librerías: `Django REST Framework`.
5.  [ ] **5. Introducción a GraphQL con Python**
    - **Descripción:** Entiende los principios de GraphQL como alternativa a REST. Aprende a definir esquemas, tipos, queries y mutations para construir APIs flexibles con Graphene (integrado con Django) o Strawberry.
    - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/05_apis_graphql.ipynb)
    - Librerías: `graphene-django` (para Django) o `strawberry-graphql` (más moderno, puede usarse con FastAPI/Flask/Django).


### 3. Nivel Avanzado

1.  [ ] **1. Programación Asíncrona en Python**
    -   **Descripción:** Domina los conceptos de programación asíncrona en Python utilizando `asyncio`, `async` y `await`. Entiende los bucles de eventos, corrutinas y tareas, esenciales para aplicaciones web de alta concurrencia.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/01_programacion_asincrona.ipynb)
    -   Librerías: `asyncio`.
2.  [ ] **2. WebSockets y Comunicación en Tiempo Real**
    -   **Descripción:** Implementa comunicación bidireccional en tiempo real entre el cliente y el servidor utilizando WebSockets. Explora casos de uso como chats, notificaciones en vivo y dashboards dinámicos con librerías adecuadas para tu framework elegido.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/02_websockets_real_time.ipynb)
    -   Librerías: `FastAPI WebSockets` / `Starlette WebSockets`, `python-socketio` (con Flask/Django), `channels` (para Django).
3.  [ ] **3. Colas de Tareas Asíncronas con Celery y Message Brokers**
    -   **Descripción:** Aprende a desacoplar y escalar tu aplicación web mediante el uso de colas de tareas asíncronas con Celery. Entiende el papel de los message brokers como Redis o RabbitMQ para gestionar tareas en segundo plano y mejorar la responsividad.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/03_celery_message_brokers.ipynb) 
    -   Herramientas/Librerías: `Celery`, `Redis` (como broker/backend), `RabbitMQ` (como broker).
4.  [ ] **4. Estrategias de Cacheo Avanzado**
    -   **Descripción:** Implementa diversas estrategias de cacheo (en memoria, en base de datos, distribuido con Redis/Memcached) para optimizar el rendimiento de tu aplicación web, reduciendo la carga en la base de datos y los tiempos de respuesta.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/04_cacheo_avanzado.ipynb) 
    -   Herramientas/Librerías: `Redis`, `Memcached`, cacheo a nivel de framework (Django cache framework).
5.  [ ] **5. Seguridad en Aplicaciones Web**
    -   **Descripción:** Conoce las vulnerabilidades web más comunes (basadas en el OWASP Top 10) y aprende las mejores prácticas y herramientas que ofrecen los frameworks de Python para prevenir ataques y construir aplicaciones seguras.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/05_seguridad_web.ipynb) 
    -   Conceptos: XSS, CSRF, SQL Injection, Insecure Deserialization, etc., y cómo los frameworks ayudan a mitigar.
6.  [ ] **6. Testing de Aplicaciones Web en Python**
    -   **Descripción:** Aprende a escribir pruebas unitarias, de integración y funcionales (end-to-end) para tus aplicaciones web Python, utilizando `unittest`, `pytest` y las utilidades de testing que proveen los frameworks.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/06_testing_web.ipynb) 
    -   Librerías: `unittest`, `pytest`, `Selenium` (para E2E), herramientas de testing de frameworks (Django Test Client, FastAPI TestClient).
7.  [ ] **7. Despliegue de Aplicaciones Python (WSGI/ASGI, Contenedores)**
    -   **Descripción:** Entiende cómo desplegar tus aplicaciones web Python en entornos de producción. Cubre servidores WSGI/ASGI (Gunicorn, Uvicorn), el uso de un proxy inverso como Nginx, y la contenerización con Docker para facilitar el despliegue y la escalabilidad.
    -   [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/07_despliegue_python_web.ipynb) 
    -   Herramientas/Conceptos: Gunicorn, Uvicorn, Nginx, Docker, principios de CI/CD para web.


## 6. DevOps y Contenedores --pendiente actulizar

### 1. Nivel Básico
1. [ ] **1. Introducción a Docker**  
   - [Notebook](../6._DevOps_y_Contenedores/1._Nivel_Básico/notebooks/01_introduccion_docker.ipynb)  
   - Conceptos: imágenes, contenedores, Dockerfile  
2. [ ] **2. CI/CD Básico**  
   - [Notebook](../6._DevOps_y_Contenedores/1._Nivel_Básico/notebooks/02_cicd_basico.ipynb)  
   - Herramientas: GitHub Actions, GitLab CI  
3. [ ] **3. Orquestación**  
   - [Notebook](../6._DevOps_y_Contenedores/1._Nivel_Básico/notebooks/03_orquestacion.ipynb)  
   - Herramientas: Kubernetes, `kubectl`, `helm`  


### 2. Nivel Intermedio
1. [ ] **1. Kubernetes**  
   - [Notebook](../6._DevOps_y_Contenedores/2._Nivel_Intermedio/notebooks/01_kubernetes_fundamentos.ipynb)  
   - Conceptos: Pods, Deployments, Services  
2. [ ] **2. Infraestructura como Código**  
   - [Notebook](../6._DevOps_y_Contenedores/2._Nivel_Intermedio/notebooks/02_infraestructura_codigo.ipynb)  
   - Herramientas: Terraform, AWS CDK  


### 3. Nivel Avanzado
1. [ ] **1. GitOps y Automatización**  
   - [Notebook](../6._DevOps_y_Contenedores/3._Nivel_Avanzado/notebooks/01_gitops_automatizacion.ipynb)  
   - Herramientas: Argo CD, Flux  
2. [ ] **2. Seguridad en CI/CD**  
   - [Notebook](../6._DevOps_y_Contenedores/3._Nivel_Avanzado/notebooks/02_seguridad_cicd.ipynb)  
   - Prácticas: escaneo de contenedores, SAST/DAST  


## 7. Seguridad y Buenas Prácticas

### 1. Nivel Básico
1. [ ] **1. OWASP Top Ten**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/1._Nivel_Básico/notebooks/01_owasp_top_ten.ipynb)  
   - Conceptos: inyección, XSS, CSRF  
2. [ ] **2. Gestión de Secretos**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/1._Nivel_Básico/notebooks/02_gestion_secretos.ipynb)  
   - Herramientas: HashiCorp Vault, AWS Secrets Manager  
3. [ ] **3. Principios de Seguridad**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/1._Nivel_Básico/notebooks/03_principios_seguridad.ipynb)  
   - Librerías: *No específicas*  


### 2. Nivel Intermedio
1. [ ] **1. Hardening de Servidores**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/2._Nivel_Intermedio/notebooks/01_hardening_servidores.ipynb)  
   - Prácticas: firewalls, SSH seguro  
2. [ ] **2. Autenticación y Autorización**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/2._Nivel_Intermedio/notebooks/02_autenticacion_autorizacion.ipynb)  
   - Librerías: `OAuthlib`, `PyJWT`  


### 3. Nivel Avanzado
1. [ ] **1. Auditorías y Pentesting Automatizado**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/3._Nivel_Avanzado/notebooks/01_auditorias_pentesting.ipynb)  
   - Herramientas: OWASP ZAP, Metasploit  
2. [ ] **2. Respuesta a Incidentes y Forense**  
   - [Notebook](../7._Seguridad_y_Buenas_Prácticas/3._Nivel_Avanzado/notebooks/02_respuesta_incidentes_forense.ipynb)  
   - Conceptos: logs, trazabilidad  


## 8. Habilidades Blandas y Metodologías Ágiles

### 1. Nivel Básico
1. [ ] **1. Git y Flujos de Trabajo**  
   - [Notebook](../8._Habilidades_Blandas_y_Metodologías_Ágiles/1._Nivel_Básico/notebooks/01_git_flujos_trabajo.ipynb)  
2. [ ] **2. Escritura de Documentación Técnica**  
   - [Notebook](../8._Habilidades_Blandas_y_Metodologías_Ágiles/1._Nivel_Básico/notebooks/02_documentacion_tecnica.ipynb)  
   - Herramientas: Markdown, MkDocs  


### 2. Nivel Intermedio
1. [ ] **1. Metodologías Ágiles (Scrum, Kanban)**  
   - [Notebook](../8._Habilidades_Blandas_y_Metodologías_Ágiles/2._Nivel_Intermedio/notebooks/01_metodologias_agiles.ipynb)  
2. [ ] **2. Gestión de Proyectos**  
   - [Notebook](../8._Habilidades_Blandas_y_Metodologías_Ágiles/2._Nivel_Intermedio/notebooks/02_gestion_proyectos.ipynb)  
   - Herramientas: Jira, Trello  


### 3. Nivel Avanzado

1. [ ] **1. Comunicación y Trabajo en Equipo**  
   - [Notebook](../8._Habilidades_Blandas_y_Metodologías_Ágiles/3._Nivel_Avanzado/notebooks/01_comunicacion_trabajo_equipo.ipynb)  

2. [ ] **2. Mentoría y Revisión de Código**  
   - [Notebook](../8._Habilidades_Blandas_y_Metodologías_Ágiles/3._Nivel_Avanzado/notebooks/02_mentoria_revision_codigo.ipynb)  