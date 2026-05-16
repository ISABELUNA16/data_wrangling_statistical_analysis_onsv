import pandas as pd
import numpy as np
import geopandas as gpd

print("[Limpieza] Iniciando auditoría y extracción de 16 Variables Críticas (Incluyendo Población)...")

# 1. Cargar datos desde los CSV ligeros generados en la Ingesta
df_siniestros = pd.read_csv('data/processed/01_siniestros.csv', dtype=str)
df_vehiculos = pd.read_csv('data/processed/01_vehiculos.csv', dtype=str)
df_personas = pd.read_csv('data/processed/01_personas.csv', dtype=str)
df_poblacion = pd.read_csv('data/processed/01_poblacion.csv', dtype=str)

auditoria = {
    'filas_iniciales': {'Siniestros': len(df_siniestros), 'Vehículos': len(df_vehiculos), 'Personas': len(df_personas), 'Población': len(df_poblacion)},
    'duplicados': {},
    'variables_auditadas': []
}

def auditar_y_limpiar_texto(df, tabla, columna, valor_reemplazo='NO ESPECIFICADO'):
    if columna in df.columns:
        nulos = df[columna].isna().sum()
        accion = f"Reemplazado por '{valor_reemplazo}'" if nulos > 0 else "Perfecto (Limpio)"
        auditoria['variables_auditadas'].append({'Tabla': tabla, 'Variable': columna, 'Corruptos/Nulos': nulos, 'Acción': accion})
        df[columna] = df[columna].fillna(valor_reemplazo).str.upper().str.strip()
    return df

# 1. DUPLICADOS
for tabla, nombre in [(df_siniestros, 'Siniestros'), (df_vehiculos, 'Vehículos'), (df_personas, 'Personas')]:
    auditoria['duplicados'][nombre] = tabla.duplicated().sum()
    tabla.drop_duplicates(inplace=True)
auditoria['duplicados']['Población'] = df_poblacion.duplicated().sum()

# 2. TABLA PERSONAS
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'SEXO', 'IGNORADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'TIPO PERSONA', 'IGNORADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'ESTADO LICENCIA', 'NO ESPECIFICADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO', 'NO SE REALIZÓ')

df_personas['EDAD'] = pd.to_numeric(df_personas['EDAD'], errors='coerce') 
df_personas.loc[(df_personas['EDAD'] < 0) | (df_personas['EDAD'] > 110), 'EDAD'] = np.nan
auditoria['variables_auditadas'].append({'Tabla': 'Personas', 'Variable': 'EDAD', 'Corruptos/Nulos': df_personas['EDAD'].isna().sum(), 'Acción': "Outliers a NaN"})

conductores = df_personas[df_personas['TIPO PERSONA'] == 'CONDUCTOR']
agg_personas = conductores.groupby('CÓDIGO SINIESTRO').agg(
    cond_hombres=('SEXO', lambda x: (x == 'MASCULINO').sum()),
    cond_mujeres=('SEXO', lambda x: (x == 'FEMENINO').sum()),
    cond_licencia_irregular=('ESTADO LICENCIA', lambda x: x.isin(['NO TIENE', 'VENCIDO', 'RETENIDO', 'CANCELADO']).sum()),
    cond_estado_ebriedad=('RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO', lambda x: (x == 'POSITIVO').sum())
).reset_index()

# 3. TABLA VEHÍCULOS
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'VEHÍCULO', 'IGNORADO')
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'ESTADO SOAT', 'NO ESPECIFICADO')
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'ESTADO CITV', 'NO ESPECIFICADO')

agg_vehiculos = df_vehiculos.groupby('CÓDIGO SINIESTRO').agg(
    total_vehiculos=('CÓDIGO VEHICULO', 'count'),
    motos_involucradas=('VEHÍCULO', lambda x: x.str.contains('MOTO').sum()),
    vehiculos_sin_soat=('ESTADO SOAT', lambda x: x.isin(['NO TIENE', 'VENCIDO']).sum()),
    vehiculos_sin_citv=('ESTADO CITV', lambda x: x.isin(['NO TIENE', 'VENCIDO']).sum())
).reset_index()

# 4. TABLA SINIESTROS
df_siniestros = df_siniestros.dropna(subset=['CÓDIGO SINIESTRO'])
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CLASE SINIESTRO', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CAUSA FACTOR PRINCIPAL', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CONDICIÓN CLIMÁTICA', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'ZONA', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'TIPO DE VÍA', 'NO ESPECIFICADO')

df_siniestros['HORA_ENTERA'] = pd.to_datetime(df_siniestros['HORA SINIESTRO'], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
auditoria['variables_auditadas'].append({'Tabla': 'Siniestros', 'Variable': 'HORA SINIESTRO', 'Corruptos/Nulos': (df_siniestros['HORA_ENTERA'] == -1).sum(), 'Acción': "Errores a -1"})

for col in ['CANTIDAD DE FALLECIDOS', 'CANTIDAD DE LESIONADOS']:
    df_siniestros[col] = pd.to_numeric(df_siniestros[col], errors='coerce').fillna(0).astype(int)

# Crear llave ase (Sin tildes ni espacios para asegurar el merge)
for col in ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']:
    df_siniestros[col] = df_siniestros[col].fillna('SIN_DATO').str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_siniestros['KEY_UBICACION'] = df_siniestros['DEPARTAMENTO'] + '-' + df_siniestros['PROVINCIA'] + '-' + df_siniestros['DISTRITO']

# 5. PUENTE GEOGRÁFICO: OBTENER POBLACIÓN (Variable 16)
# Cargamos el GeoJSON para construir el traductor UBIGEO -> Llave de texto
mapa_peru = gpd.read_file('data/raw/Distritos_Peru_v1.geojson')
for col in ['NOMBDEP', 'NOMBPROV', 'NOMBDIST']:
    mapa_peru[col] = mapa_peru[col].fillna('').str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

mapa_peru['KEY_UBICACION'] = mapa_peru['NOMBDEP'] + '-' + mapa_peru['NOMBPROV'] + '-' + mapa_peru['NOMBDIST']
puente = mapa_peru[['UBIGEO', 'KEY_UBICACION']].drop_duplicates()

# Limpiar cod_dist del INEI para cruzarlo (asegurar 6 dígitos numéricos)
df_poblacion['cod_dist'] = df_poblacion['cod_dist'].astype(str).str.zfill(6)
df_pob_cruzada = df_poblacion.merge(puente, left_on='cod_dist', right_on='UBIGEO', how='inner')

# Extraer y limpiar total_pers (Var 16)
df_pob_cruzada['total_pers'] = pd.to_numeric(df_pob_cruzada['total_pers'], errors='coerce')
nulos_pob = df_pob_cruzada['total_pers'].isna().sum()
auditoria['variables_auditadas'].append({'Tabla': 'Población', 'Variable': 'total_pers', 'Corruptos/Nulos': nulos_pob, 'Acción': "Convertido a numérico"})

# Agrupamos por la llave para evitar duplicados del shapefile
poblacion_limpia = df_pob_cruzada.groupby('KEY_UBICACION')['total_pers'].max().reset_index()

# 6. MERGE MAESTRO FINAL
df_master = pd.merge(df_siniestros, agg_personas, on='CÓDIGO SINIESTRO', how='left')
df_master = pd.merge(df_master, agg_vehiculos, on='CÓDIGO SINIESTRO', how='left')
# Aquí entra la población a la sábana principal
df_master = pd.merge(df_master, poblacion_limpia, on='KEY_UBICACION', how='left')

vars_agg = ['cond_hombres', 'cond_mujeres', 'cond_licencia_irregular', 'cond_estado_ebriedad', 'total_vehiculos', 'motos_involucradas', 'vehiculos_sin_soat', 'vehiculos_sin_citv']
df_master[vars_agg] = df_master[vars_agg].fillna(0).astype(int)

# Rellenar población vacía con la mediana para evitar NaN al hacer correlaciones o división por cero
mediana_pob = df_master['total_pers'].median()
df_master['total_pers'] = df_master['total_pers'].fillna(mediana_pob)

df_master.to_csv('data/processed/02_dataset_analitico.csv', index=False)

# 7. REPORTE
print("\n" + "="*95)
print(" REPORTE DE GOBIERNO DE DATOS: AUDITORÍA DE 16 VARIABLES CRÍTICAS")
print("="*95)
for i, reg in enumerate(auditoria['variables_auditadas'], 1):
    icono = "✅" if reg['Corruptos/Nulos'] == 0 else "⚠️"
    print(f" {i:>2}. {reg['Tabla']:<11} | {reg['Variable']:<35} | {reg['Corruptos/Nulos']:<8} | {icono} {reg['Acción']}")
print(f"\n DATASET MAESTRO (Con Población Integrada): {len(df_master)} incidentes listos.")
print("="*95 + "\n")