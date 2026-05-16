import pandas as pd
import os
import sys

print("[Ingesta] Iniciando lectura de fuentes de datos (Archivos XLSX)...")

rutas = {
    'siniestros': 'data/raw/BBDD_ONSV_SINIESTROS_FATALES_2021_2025.xlsx',
    'vehiculos': 'data/raw/BBDD_ONSV_VEHICULOS_2021_2025.xlsx',
    'personas': 'data/raw/BBDD_ONSV_PERSONAS_2021_2025.xlsx',
    'poblacion': 'data/raw/GeoPeru-peru_distritos.xlsx'
}

for clave, ruta in rutas.items():
    if not os.path.exists(ruta):
        print(f"[Error] No se encontró el archivo: '{ruta}'")
        sys.exit(1)

try:
    # 1. Lectura de archivos ONSV (Saltando 4 filas de metadatos)
    df_siniestros = pd.read_excel(rutas['siniestros'], skiprows=4, dtype=str, engine='openpyxl')
    df_vehiculos = pd.read_excel(rutas['vehiculos'], skiprows=4, dtype=str, engine='openpyxl')
    df_personas = pd.read_excel(rutas['personas'], skiprows=4, dtype=str, engine='openpyxl')
    
    # 2. Lectura del archivo del INEI como Excel nativo (Sin saltar filas)
    df_poblacion = pd.read_excel(rutas['poblacion'], dtype=str, engine='openpyxl')

    # 3. Limpieza de nombres de columnas (Quitar espacios en blanco)
    df_siniestros.columns = df_siniestros.columns.str.strip()
    df_vehiculos.columns = df_vehiculos.columns.str.strip()
    df_personas.columns = df_personas.columns.str.strip()
    df_poblacion.columns = df_poblacion.columns.str.strip()

    print(f"[Ingesta] Filas -> Siniestros: {len(df_siniestros)} | Veh: {len(df_vehiculos)} | Per: {len(df_personas)} | Pob: {len(df_poblacion)}")

    # 4. Exportar a CSV ligero en procesados para agilizar las siguientes fases
    df_siniestros.to_csv('data/processed/01_siniestros.csv', index=False)
    df_vehiculos.to_csv('data/processed/01_vehiculos.csv', index=False)
    df_personas.to_csv('data/processed/01_personas.csv', index=False)
    df_poblacion.to_csv('data/processed/01_poblacion.csv', index=False)
    
    print("[Ingesta] Datos base convertidos a CSV y exportados exitosamente.")

except Exception as e:
    print(f"[Error Crítico en Ingesta]: {e}")
    sys.exit(1)