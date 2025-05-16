import os
import json
import re
from urllib.parse import unquote

# Ruta al archivo Markdown con los enlaces
path_md = "docs/index.md"

# Leer todas las líneas del Markdown
with open(path_md, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Plantilla mínima para un notebook vacío
notebook_template = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 2
}

# Función para sanear nombres de archivos/carpetas
def sanear(nombre):
    nombre = nombre.strip()
    nombre = re.sub(r"[\\/:*?\"<>|]", "", nombre)
    nombre = nombre.replace(" ", "_")
    return nombre

# Extraer rutas de notebook del Markdown en estructura tema->subtema->rutas
temas = {}
for line in lines:
    match = re.search(r'\[Notebook\]\((.+?)\)', line)
    if not match:
        continue
    ruta_raw = match.group(1)
    ruta = unquote(ruta_raw)
    partes = ruta.split('/')
    if len(partes) < 3:
        # Se espera al menos tema/subtema/archivo
        continue
    tema, subtema = partes[0], partes[1]
    temas.setdefault(tema, {}).setdefault(subtema, []).append(ruta)

# Crear carpetas y notebooks vacíos con numeración por subtema
def crear_notebooks_por_subtema(temas, base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()
    for tema, subtemas in temas.items():
        for subtema, rutas in subtemas.items():
            for idx, ruta in enumerate(rutas, start=1):
                partes = ruta.split('/')
                *carpetas, archivo = partes
                carpetas_saneadas = [sanear(c) for c in carpetas]
                nombre_base, ext = os.path.splitext(archivo)
                nombre_saneado = sanear(nombre_base)
                # Prefijo de numeración por subtema
                prefijo = f"{idx:02d}"
                archivo_numerado = f"{prefijo}_{nombre_saneado}.ipynb"

                # Crear ruta de carpetas
                dir_path = os.path.join(base_dir, *carpetas_saneadas)
                os.makedirs(dir_path, exist_ok=True)

                # Ruta completa del notebook numerado
                notebook_path = os.path.join(dir_path, archivo_numerado)

                # Crear notebook vacío si no existe
                if not os.path.exists(notebook_path):
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        json.dump(notebook_template, f, ensure_ascii=False, indent=2)
                    print(f"Creado: {notebook_path}")
                else:
                    print(f"Ya existe: {notebook_path}")
    print("Proceso completado: notebooks numerados por subtema creados.")

# Ejecutar la función
tmp_base = os.getcwd()
crear_notebooks_por_subtema(temas, base_dir=tmp_base)
