#!/bin/bash
set -e 

echo " INICIANDO PIPELINE: OBSERVATORIO NACIONAL DE SEGURIDAD VIAL"

mkdir -p data/processed
mkdir -p results/plots

echo "-> [1/3] Ejecutando Ingesta (Leyendo archivos XLSX nativos)..."
python src/data_ingestion.py

echo "-> [2/3] Ejecutando Data Wrangling (Tratamiento de Nulos y Merges)..."
python src/data_wrangling.py

echo "-> [3/3] Ejecutando Análisis Estadístico y Modelado Espacial..."
python src/data_analysis.py

echo " PIPELINE FINALIZADO CON ÉXITO. Revisa la carpeta /results/plots"