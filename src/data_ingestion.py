import pandas as pd
import os
import sys

print("[Ingesta] Iniciando lectura nativa de archivos Excel (XLSX)...")

# Definir las rutas exactas de los 3 archivos separados
rutas = {
    'siniestros': 'data/raw/BBDD_ONSV_SINIESTROS_FATALES_2021_2025.xlsx',
    'vehiculos': 'data/raw/BBDD_ONSV_VEHICULOS_2021_2025.xlsx',
    'personas': 'data/raw/BBDD_ONSV_PERSONAS_2021_2025.xlsx'
}

# Validación de existencia
for clave, ruta in rutas.items():
    if not os.path.exists(ruta):
        print(f"[Error] No se encontró el archivo: '{ruta}'")
        sys.exit(1)

try:
    # 1. Lectura de Excel nativo usando openpyxl.
    # skiprows=4 es vital porque las primeras 4 filas son títulos institucionales.
    # dtype=str convierte todo a texto temporalmente para evitar que Pandas borre ceros a la izquierda.
    df_siniestros = pd.read_excel(rutas['siniestros'], skiprows=4, dtype=str, engine='openpyxl')
    df_vehiculos = pd.read_excel(rutas['vehiculos'], skiprows=4, dtype=str, engine='openpyxl')
    df_personas = pd.read_excel(rutas['personas'], skiprows=4, dtype=str, engine='openpyxl')

    # 2. Limpieza estructural estricta de nombres de columnas
    # Esto elimina espacios en blanco como "CAUSA    " -> "CAUSA"
    df_siniestros.columns = df_siniestros.columns.str.strip()
    df_vehiculos.columns = df_vehiculos.columns.str.strip()
    df_personas.columns = df_personas.columns.str.strip()

    print(f"[Ingesta] Filas extraídas -> Siniestros: {len(df_siniestros)} | Vehículos: {len(df_vehiculos)} | Personas: {len(df_personas)}")

    # 3. Exportamos a CSV en la carpeta processed. 
    # Esto hace que las fases 2 y 3 corran en segundos en lugar de minutos.
    df_siniestros.to_csv('data/processed/01_siniestros.csv', index=False)
    df_vehiculos.to_csv('data/processed/01_vehiculos.csv', index=False)
    df_personas.to_csv('data/processed/01_personas.csv', index=False)
    
    print("[Ingesta] Datos base convertidos y exportados a data/processed/ exitosamente.")

except Exception as e:
    print(f"[Error Crítico en Ingesta]: {e}")
    sys.exit(1)