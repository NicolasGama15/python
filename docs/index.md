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

### Modelado
1. [ ] **1. Normalización**  
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/01_normalizacion.ipynb)  
   - Librerías: *No específicas*  
2. [ ] **2. Modelado NoSQL**  
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/02_modelado_nosql.ipynb)  
   - Librerías: `pymongo`  
3. [ ] **3. Modelado SQL**  
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/03_modelado_sql.ipynb)  
   - Librerías: *No específicas*  
4. [ ] **4. Bases de Datos de Serie Temporal**  
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/04_serie_temporal.ipynb)  
   - Tecnologías: InfluxDB, TimescaleDB  
5. [ ] **5. Bases de Datos en Grafos**  
   - [Notebook](../3._Bases_de_Datos/Modelado/notebooks/05_bases_grafos.ipynb)  
   - Tecnologías: Neo4j, JanusGraph  


### SQL Avanzado
1. [ ] **1. Consultas Complejas**  
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/01_consultas_complejas.ipynb)  
   - Librerías: *No específicas* (clientes: `psycopg2`, `mysql-connector-python`)  
2. [ ] **2. Optimización de Consultas**  
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/02_optimizacion_consultas.ipynb)  
   - Herramientas: `EXPLAIN`, `pgcli`  
3. [ ] **3. Data Warehousing y OLAP**  
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/03_data_warehousing_olap.ipynb)  
   - Herramientas: Snowflake, Amazon Redshift  
4. [ ] **4. ETL y Calidad de Datos**  
   - [Notebook](../3._Bases_de_Datos/SQL_Avanzado/notebooks/04_etl_calidad_datos.ipynb)  
   - Herramientas: Apache Airflow, Great Expectations  


### NoSQL Avanzado
1. [ ] **1. Modelado de Documentos**  
   - [Notebook](../3._Bases_de_Datos/NoSQL_Avanzado/notebooks/01_modelado_documentos.ipynb)  
   - Librerías: `pymongo`, `mongoengine`  
2. [ ] **2. Bases de Datos de Grafos**  
   - [Notebook](../3._Bases_de_Datos/NoSQL_Avanzado/notebooks/02_nosql_grafos.ipynb)  
   - Librerías: `neo4j`, `py2neo`  
3. [ ] **3. Bases de Datos en Memoria**  
   - [Notebook](../3._Bases_de_Datos/NoSQL_Avanzado/notebooks/03_bases_memoria.ipynb)  
   - Librerías: `redis-py`  


### Integración de Datos
1. [ ] **1. ETL y Pipelines**  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/01_etl_pipelines.ipynb)  
   - Librerías: `apache-airflow`, `prefect`, `luigi`  
2. [ ] **2. Data Warehousing**  
   - [Notebook](../3._Bases_de_Datos/Integracion_de_Datos/notebooks/02_data_warehousing_integration.ipynb)  
   - Librerías: `sqlalchemy`  


### Administración
1. [ ] **1. Instalación y Configuración**  
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/01_instalacion_configuracion.ipynb)  
   - Herramientas: `psycopg2`, `mysqlclient`  
2. [ ] **2. Backup y Recuperación**  
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/02_backup_recuperacion.ipynb)  
   - Herramientas: `pg_dump`  
3. [ ] **3. Replicación**  
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/03_replicacion.ipynb)  
   - Herramientas: `repmgr`, configuración nativa de PostgreSQL  
4. [ ] **4. Seguridad y Roles**  
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/04_seguridad_roles.ipynb)  
   - Conceptos: encriptación en reposo, gestión de roles  
5. [ ] **5. Replicación Multimaestro y Alta Disponibilidad**  
   - [Notebook](../3._Bases_de_Datos/Administracion/notebooks/05_replicacion_multimaestro_ha.ipynb)  
   - Tecnologías: Galera Cluster, Patroni  


## 4. Cloud Computing (AWS)

### 1. Nivel Básico
1. [ ] **1. Conceptos de la Nube**  
   - [Notebook](../4._Cloud_Computing/1._Nivel_Básico/aws/notebooks/01_conceptos_nube.ipynb)  
   - Librerías: *No específicas*  
2. [ ] **2. IAM**  
   - [Notebook](../4._Cloud_Computing/1._Nivel_Básico/aws/notebooks/02_iam.ipynb)  
   - Librerías: `boto3`  
3. [ ] **3. Contenedores en la Nube**  
   - [Notebook](../4._Cloud_Computing/1._Nivel_Básico/aws/notebooks/03_contenedores_nube.ipynb)  
   - Servicios: Amazon ECS, AWS Fargate  


### 2. Nivel Intermedio
1. [ ] **1. Compute y Networking**  
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/01_compute_networking.ipynb)  
   - Librerías: `boto3`, `awscli`  
2. [ ] **2. Storage y Bases de Datos Gestionadas**  
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/02_storage_bases_datos.ipynb)  
   - Librerías: `boto3`  
3. [ ] **3. Orquestación con Kubernetes**  
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/03_orquestacion_kubernetes.ipynb)  
   - Servicios: Amazon EKS  
4. [ ] **4. CI/CD en AWS**  
   - [Notebook](../4._Cloud_Computing/2._Nivel_Intermedio/aws/notebooks/04_cicd_aws.ipynb)  
   - Herramientas: CodePipeline, CodeBuild  


### 3. Nivel Avanzado
1. [ ] **1. Serverless**  
   - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/01_serverless.ipynb)  
   - Librerías: `boto3`, `chalice`  
2. [ ] **2. Infraestructura como Código**  
   - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/02_infraestructura_codigo.ipynb)  
   - Herramientas: `aws-cdk`, `cloudformation`, `terraform`  
3. [ ] **3. Monitorización y Seguridad**  
   - [Notebook](../4._Cloud_Computing/3._Nivel_Avanzado/aws/notebooks/03_Monitorizacion_y_seguridad.ipynb)  


## 5. Desarrollo Web con Python

### 1. Nivel Básico
1. [ ] **1. Introducción a HTTP y Servidores**  
   - [Notebook](../5._Desarrollo_Web_con_Python/1._Nivel_Básico/notebooks/01_introduccion_http_servidores.ipynb)  
   - Librerías: `http.server`, `requests`  
2. [ ] **2. Microframeworks**  
   - [Notebook](../5._Desarrollo_Web_con_Python/1._Nivel_Básico/notebooks/02_microframeworks.ipynb)  
   - Librerías: `Flask`, `FastAPI`  


### 2. Nivel Intermedio
1. [ ] **1. Frameworks Completos**  
   - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/01_frameworks_completos.ipynb)  
   - Librerías: `Django`, `Tortoise ORM`  
2. [ ] **2. APIs REST y GraphQL**  
   - [Notebook](../5._Desarrollo_Web_con_Python/2._Nivel_Intermedio/notebooks/02_apis_rest_graphql.ipynb)  
   - Librerías: `Django REST Framework`, `graphene`  


### 3. Nivel Avanzado
1. [ ] **1. WebSockets y Tiempo Real**  
   - [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/01_websockets_real_time.ipynb)  
   - Librerías: `socket.io`, `starlette`  
2. [ ] **2. Escalabilidad y Cacheo**  
   - [Notebook](../5._Desarrollo_Web_con_Python/3._Nivel_Avanzado/notebooks/02_escalabilidad_cacheo.ipynb)  
   - Herramientas: Redis, RabbitMQ, Kafka  


## 6. DevOps y Contenedores

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