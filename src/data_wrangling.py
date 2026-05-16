import pandas as pd
import numpy as np

print("[Limpieza] Iniciando auditoría, Data Wrangling y consolidación de 10 Variables Críticas...")

#  CARGA DE DATOS GENERADOS POR LA INGESTA
df_siniestros = pd.read_csv('data/processed/01_siniestros.csv', dtype=str)
df_vehiculos = pd.read_csv('data/processed/01_vehiculos.csv', dtype=str)
df_personas = pd.read_csv('data/processed/01_personas.csv', dtype=str)

# Auditoría de datos: Estructura para registrar nulos, corruptos, outliers y acciones tomadas
auditoria = {
    'filas_iniciales': {'Siniestros': len(df_siniestros), 'Vehículos': len(df_vehiculos), 'Personas': len(df_personas)},
    'duplicados': {},
    'nulos_y_corruptos': []
}

def auditar_y_limpiar_texto(df, tabla, columna, valor_reemplazo='NO ESPECIFICADO'):
    #Función modular para auditar nulos en variables categóricas
    if columna in df.columns:
        nulos = df[columna].isna().sum()
        if nulos > 0:
            auditoria['nulos_y_corruptos'].append({
                'Tabla': tabla, 'Columna': columna, 'Cantidad': nulos, 'Acción': f"Reemplazado por '{valor_reemplazo}'"
            })
        df[columna] = df[columna].fillna(valor_reemplazo).str.upper().str.strip()
    return df

# A. ELIMINACIÓN DE DATOS DUPLICADOS
for tabla, nombre_tabla in [(df_siniestros, 'Siniestros'), (df_vehiculos, 'Vehículos'), (df_personas, 'Personas')]:
    auditoria['duplicados'][nombre_tabla] = tabla.duplicated().sum()
    tabla.drop_duplicates(inplace=True)

# B. TRATAMIENTO DE LA TABLA: PERSONAS (Variables 1, 2, 3 y 4)
# V1: Sexo | V3: Licencia | V4: Dosaje Etílico (Además limpiamos 'TIPO PERSONA')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'TIPO PERSONA', 'IGNORADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'SEXO', 'IGNORADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'ESTADO LICENCIA', 'NO ESPECIFICADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO', 'NO SE REALIZÓ')

# V2: Columna EDAD (Outliers y Corruptos)
edad_nula_orig = df_personas['EDAD'].isna().sum()
df_personas['EDAD'] = pd.to_numeric(df_personas['EDAD'], errors='coerce') 
edad_corrupta = df_personas['EDAD'].isna().sum() - edad_nula_orig

if edad_nula_orig > 0 or edad_corrupta > 0:
    auditoria['nulos_y_corruptos'].append({'Tabla': 'Personas', 'Columna': 'EDAD', 'Cantidad': edad_nula_orig + edad_corrupta, 'Acción': "Textos vacíos/corruptos -> NaN"})

outliers_edad = ((df_personas['EDAD'] < 0) | (df_personas['EDAD'] > 110)).sum()
if outliers_edad > 0:
    auditoria['nulos_y_corruptos'].append({'Tabla': 'Personas', 'Columna': 'EDAD (Outliers)', 'Cantidad': outliers_edad, 'Acción': "Edades imposibles -> NaN"})
    df_personas.loc[(df_personas['EDAD'] < 0) | (df_personas['EDAD'] > 110), 'EDAD'] = np.nan

# --- Agrupaciones de Personas ---
conductores = df_personas[df_personas['TIPO PERSONA'] == 'CONDUCTOR']
agg_personas = conductores.groupby('CÓDIGO SINIESTRO').agg(
    cond_hombres=('SEXO', lambda x: (x == 'MASCULINO').sum()),
    cond_mujeres=('SEXO', lambda x: (x == 'FEMENINO').sum()),
    cond_licencia_irregular=('ESTADO LICENCIA', lambda x: x.isin(['NO TIENE', 'VENCIDO', 'RETENIDO', 'CANCELADO']).sum()),
    cond_estado_ebriedad=('RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO', lambda x: (x == 'POSITIVO').sum())
).reset_index()

# C. TRATAMIENTO DE LA TABLA: VEHÍCULOS (Variables 5 y 6)
# V5: Clase de Vehículo | V6: Estado SOAT
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'VEHÍCULO', 'IGNORADO')
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'ESTADO SOAT', 'NO ESPECIFICADO')

# --- Agrupaciones de Vehículos ---
agg_vehiculos = df_vehiculos.groupby('CÓDIGO SINIESTRO').agg(
    total_vehiculos=('CÓDIGO VEHICULO', 'count'),
    motos_involucradas=('VEHÍCULO', lambda x: x.str.contains('MOTO').sum()),
    vehiculos_sin_soat=('ESTADO SOAT', lambda x: x.isin(['NO TIENE', 'VENCIDO']).sum())
).reset_index()

# D. TRATAMIENTO DE LA TABLA: SINIESTROS (Variables 7, 8, 9 y 10)
sin_codigo = df_siniestros['CÓDIGO SINIESTRO'].isna().sum()
if sin_codigo > 0:
    auditoria['nulos_y_corruptos'].append({'Tabla': 'Siniestros', 'Columna': 'CÓDIGO SINIESTRO', 'Cantidad': sin_codigo, 'Acción': "Fila huérfana eliminada"})
df_siniestros = df_siniestros.dropna(subset=['CÓDIGO SINIESTRO'])

# V7: Clase Siniestro | V8: Causa | V9: Clima
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CLASE SINIESTRO', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CAUSA FACTOR PRINCIPAL', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CONDICIÓN CLIMÁTICA', 'NO ESPECIFICADO')

# V10: HORA SINIESTRO (Conversión a Entero)
nulos_hora = df_siniestros['HORA SINIESTRO'].isna().sum()
hora_dt = pd.to_datetime(df_siniestros['HORA SINIESTRO'], format='%H:%M', errors='coerce')
corruptos_hora = hora_dt.isna().sum() - nulos_hora

if nulos_hora + corruptos_hora > 0:
    auditoria['nulos_y_corruptos'].append({'Tabla': 'Siniestros', 'Columna': 'HORA SINIESTRO', 'Cantidad': nulos_hora + corruptos_hora, 'Acción': "Errores/Vacíos extraídos como -1"})
df_siniestros['HORA_ENTERA'] = hora_dt.dt.hour.fillna(-1).astype(int)

# Tratamiento de Víctimas (No son las 10 var, pero es core del negocio)
for col in ['CANTIDAD DE FALLECIDOS', 'CANTIDAD DE LESIONADOS']:
    df_siniestros[col] = pd.to_numeric(df_siniestros[col], errors='coerce') 
    tot_nulos = df_siniestros[col].isna().sum()
    if tot_nulos > 0:
        auditoria['nulos_y_corruptos'].append({'Tabla': 'Siniestros', 'Columna': col, 'Cantidad': tot_nulos, 'Acción': "Imputado a 0"})
    df_siniestros[col] = df_siniestros[col].fillna(0).astype(int)

# LLAVE ESPACIAL COMPUESTA
for col in ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']:
    df_siniestros[col] = df_siniestros[col].fillna('SIN_DATO').str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_siniestros['KEY_UBICACION'] = df_siniestros['DEPARTAMENTO'] + '-' + df_siniestros['PROVINCIA'] + '-' + df_siniestros['DISTRITO']


# E. MERGES RELACIONALES (DATASET MAESTRO)
df_master = pd.merge(df_siniestros, agg_personas, on='CÓDIGO SINIESTRO', how='left')
df_master = pd.merge(df_master, agg_vehiculos, on='CÓDIGO SINIESTRO', how='left')

# Rellenar a 0 las métricas de agregación si hubo un accidente sin registros vinculados
vars_agregadas = ['cond_hombres', 'cond_mujeres', 'cond_licencia_irregular', 'cond_estado_ebriedad', 'total_vehiculos', 'motos_involucradas', 'vehiculos_sin_soat']
nulos_merge = df_master['total_vehiculos'].isna().sum()

if nulos_merge > 0:
    auditoria['nulos_y_corruptos'].append({'Tabla': 'Dataset Maestro', 'Columna': 'Métricas Agregadas', 'Cantidad': nulos_merge, 'Acción': "Rellenado con 0 (Sin vinculación secundaria)"})

df_master[vars_agregadas] = df_master[vars_agregadas].fillna(0).astype(int)

df_master.to_csv('data/processed/02_dataset_analitico.csv', index=False)

# F. REPORTE DE AUDITORÍA
print("\n" + "="*85)
print("REPORTE DE GOBIERNO DE DATOS: EXTRACCIÓN DE 10 VARIABLES CRÍTICAS")
print("="*85)

print("\n 1. FILAS INICIALES Y DUPLICADOS ELIMINADOS:")
for tabla, inicial in auditoria['filas_iniciales'].items():
    dups = auditoria['duplicados'][tabla]
    print(f"   - {tabla:<12}: {inicial:>7} filas crudas | {dups:>5} duplicados depurados")

print("\n 2. AUDITORÍA DE NULOS, CORRUPTOS Y OUTLIERS:")
print(f"   {'TABLA':<16} | {'COLUMNA / VARIABLE':<30} | {'CANTIDAD':<8} | {'ACCIÓN'}")
print("   " + "-"*82)
if not auditoria['nulos_y_corruptos']:
    print("   Todo el dataset estaba perfectamente limpio.")
else:
    for reg in auditoria['nulos_y_corruptos']:
        print(f"   {reg['Tabla']:<16} | {reg['Columna']:<30} | {reg['Cantidad']:<8} | {reg['Acción']}")

print(f"\nDATASET ANALÍTICO GENERADO:")
print(f"   - Siniestros enriquecidos listos para modelo y GeoJSON: {len(df_master)} incidentes.")
print("="*85 + "\n")