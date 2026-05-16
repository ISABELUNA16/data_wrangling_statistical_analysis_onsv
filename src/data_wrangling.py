import pandas as pd
import numpy as np

print("[Limpieza] Iniciando proceso de auditoría, Data Wrangling y extracción de 15 Variables Críticas...")

# 1. Cargar los datos generados por la Ingesta
df_siniestros = pd.read_csv('data/processed/01_siniestros.csv', dtype=str)
df_vehiculos = pd.read_csv('data/processed/01_vehiculos.csv', dtype=str)
df_personas = pd.read_csv('data/processed/01_personas.csv', dtype=str)

# 2. Auditoría y Limpieza de Variables Críticas
auditoria = {
    'filas_iniciales': {'Siniestros': len(df_siniestros), 'Vehículos': len(df_vehiculos), 'Personas': len(df_personas)},
    'duplicados': {},
    'variables_auditadas': [] # Aquí registraremos obligatoriamente las 15 variables
}

def auditar_y_limpiar_texto(df, tabla, columna, valor_reemplazo='NO ESPECIFICADO'):
    """Evalúa la variable, la registra en la auditoría (tenga o no errores) y la limpia."""
    if columna in df.columns:
        nulos = df[columna].isna().sum()
        accion = f"Reemplazado por '{valor_reemplazo}'" if nulos > 0 else "Perfecto (Limpio de origen)"
        
        auditoria['variables_auditadas'].append({
            'Tabla': tabla, 'Variable': columna, 'Corruptos/Nulos': nulos, 'Acción / Estado': accion
        })
        
        # Transformación a mayúsculas y eliminación de espacios en blanco
        if nulos > 0:
            df[columna] = df[columna].fillna(valor_reemplazo).str.upper().str.strip()
        else:
            df[columna] = df[columna].str.upper().str.strip()
    else:
        auditoria['variables_auditadas'].append({
            'Tabla': tabla, 'Variable': columna, 'Corruptos/Nulos': 'N/A', 'Acción / Estado': "¡Columna faltante en CSV!"
        })
    return df


# A. ELIMINACIÓN DE DUPLICADOS EXACTOS
for tabla, nombre_tabla in [(df_siniestros, 'Siniestros'), (df_vehiculos, 'Vehículos'), (df_personas, 'Personas')]:
    auditoria['duplicados'][nombre_tabla] = tabla.duplicated().sum()
    tabla.drop_duplicates(inplace=True)


# B. TRATAMIENTO DE LA TABLA: PERSONAS (Variables 1 al 6)
# V1: SEXO | V2: TIPO PERSONA | V3: ESTADO LICENCIA | V4: DOSAJE ETÍLICO | V5: GRAVEDAD
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'SEXO', 'IGNORADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'TIPO PERSONA', 'IGNORADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'ESTADO LICENCIA', 'NO ESPECIFICADO')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO', 'NO SE REALIZÓ')
df_personas = auditar_y_limpiar_texto(df_personas, 'Personas', 'GRAVEDAD', 'NO ESPECIFICADO')

# V6: EDAD (Auditoría especializada para números y outliers)
edad_nula_orig = df_personas['EDAD'].isna().sum()
df_personas['EDAD'] = pd.to_numeric(df_personas['EDAD'], errors='coerce') 
edad_corrupta = df_personas['EDAD'].isna().sum() - edad_nula_orig
outliers_edad = ((df_personas['EDAD'] < 0) | (df_personas['EDAD'] > 110)).sum()
total_problemas_edad = edad_nula_orig + edad_corrupta + outliers_edad

accion_edad = "Textos vacíos y outliers convertidos a NaN" if total_problemas_edad > 0 else "Perfecto (Limpio de origen)"
auditoria['variables_auditadas'].append({
    'Tabla': 'Personas', 'Variable': 'EDAD', 'Corruptos/Nulos': total_problemas_edad, 'Acción / Estado': accion_edad
})

# Neutralizar outliers
if outliers_edad > 0:
    df_personas.loc[(df_personas['EDAD'] < 0) | (df_personas['EDAD'] > 110), 'EDAD'] = np.nan

# --- Agrupaciones de Personas ---
conductores = df_personas[df_personas['TIPO PERSONA'] == 'CONDUCTOR']
agg_personas = conductores.groupby('CÓDIGO SINIESTRO').agg(
    cond_hombres=('SEXO', lambda x: (x == 'MASCULINO').sum()),
    cond_mujeres=('SEXO', lambda x: (x == 'FEMENINO').sum()),
    cond_licencia_irregular=('ESTADO LICENCIA', lambda x: x.isin(['NO TIENE', 'VENCIDO', 'RETENIDO', 'CANCELADO']).sum()),
    cond_estado_ebriedad=('RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO', lambda x: (x == 'POSITIVO').sum())
).reset_index()


# C. TRATAMIENTO DE LA TABLA: VEHÍCULOS (Variables 7 al 9)
# V7: VEHÍCULO | V8: ESTADO SOAT | V9: ESTADO CITV
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'VEHÍCULO', 'IGNORADO')
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'ESTADO SOAT', 'NO ESPECIFICADO')
df_vehiculos = auditar_y_limpiar_texto(df_vehiculos, 'Vehículos', 'ESTADO CITV', 'NO ESPECIFICADO')

# --- Agrupaciones de Vehículos ---
agg_vehiculos = df_vehiculos.groupby('CÓDIGO SINIESTRO').agg(
    total_vehiculos=('CÓDIGO VEHICULO', 'count'),
    motos_involucradas=('VEHÍCULO', lambda x: x.str.contains('MOTO').sum()),
    vehiculos_sin_soat=('ESTADO SOAT', lambda x: x.isin(['NO TIENE', 'VENCIDO']).sum()),
    vehiculos_sin_citv=('ESTADO CITV', lambda x: x.isin(['NO TIENE', 'VENCIDO']).sum())
).reset_index()


# D. TRATAMIENTO DE LA TABLA: SINIESTROS (Variables 10 al 15)
# Limpieza base obligatoria
df_siniestros = df_siniestros.dropna(subset=['CÓDIGO SINIESTRO'])

# V10: CLASE SINIESTRO | V11: CAUSA | V12: CLIMA | V13: ZONA | V14: TIPO DE VÍA
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CLASE SINIESTRO', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CAUSA FACTOR PRINCIPAL', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'CONDICIÓN CLIMÁTICA', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'ZONA', 'NO ESPECIFICADO')
df_siniestros = auditar_y_limpiar_texto(df_siniestros, 'Siniestros', 'TIPO DE VÍA', 'NO ESPECIFICADO')

# V15: HORA SINIESTRO (Auditoría especializada)
nulos_hora = df_siniestros['HORA SINIESTRO'].isna().sum()
hora_dt = pd.to_datetime(df_siniestros['HORA SINIESTRO'], format='%H:%M', errors='coerce')
corruptos_hora = hora_dt.isna().sum() - nulos_hora

accion_hora = "Horas vacías extraídas como -1" if (nulos_hora + corruptos_hora) > 0 else "Perfecto (Limpio de origen)"
auditoria['variables_auditadas'].append({
    'Tabla': 'Siniestros', 'Variable': 'HORA SINIESTRO', 'Corruptos/Nulos': nulos_hora + corruptos_hora, 'Acción / Estado': accion_hora
})
df_siniestros['HORA_ENTERA'] = hora_dt.dt.hour.fillna(-1).astype(int)

# Tratamiento de Víctimas (Obligatorio para la estadística final)
for col in ['CANTIDAD DE FALLECIDOS', 'CANTIDAD DE LESIONADOS']:
    df_siniestros[col] = pd.to_numeric(df_siniestros[col], errors='coerce').fillna(0).astype(int)

# LLAVE ESPACIAL COMPUESTA
for col in ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']:
    df_siniestros[col] = df_siniestros[col].fillna('SIN_DATO').str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_siniestros['KEY_UBICACION'] = df_siniestros['DEPARTAMENTO'] + '-' + df_siniestros['PROVINCIA'] + '-' + df_siniestros['DISTRITO']


# E. MERGES RELACIONALES (DATASET MAESTRO)
df_master = pd.merge(df_siniestros, agg_personas, on='CÓDIGO SINIESTRO', how='left')
df_master = pd.merge(df_master, agg_vehiculos, on='CÓDIGO SINIESTRO', how='left')

# Rellenar nulos post-merge
vars_agregadas = ['cond_hombres', 'cond_mujeres', 'cond_licencia_irregular', 'cond_estado_ebriedad', 
                  'total_vehiculos', 'motos_involucradas', 'vehiculos_sin_soat', 'vehiculos_sin_citv']

df_master[vars_agregadas] = df_master[vars_agregadas].fillna(0).astype(int)

df_master.to_csv('data/processed/02_dataset_analitico.csv', index=False)


# F. REPORTE DE GOBIERNO DE DATOS Y CALIDAD
print("\n" + "="*95)
print(" REPORTE DE GOBIERNO DE DATOS: AUDITORÍA DE 15 VARIABLES CRÍTICAS")
print("="*95)

print("\n 1.FILAS INICIALES Y DUPLICADOS ELIMINADOS:")
for tabla, inicial in auditoria['filas_iniciales'].items():
    dups = auditoria['duplicados'][tabla]
    print(f"   - {tabla:<12}: {inicial:>7} filas crudas | {dups:>5} duplicados depurados")

print("\n 2.AUDITORÍA DE LAS 15 VARIABLES ESTRATÉGICAS:")
print(f"   {'TABLA':<12} | {'VARIABLE EVALUADA':<35} | {'ERRORES':<8} | {'ESTADO Y ACCIÓN'}")
print("   " + "-"*92)
for i, reg in enumerate(auditoria['variables_auditadas'], 1):
    # Formato visual para identificar rápidamente las que vinieron limpias
    icono = "✅" if reg['Corruptos/Nulos'] == 0 else "⚠️"
    print(f" {i:>2}. {reg['Tabla']:<11} | {reg['Variable']:<35} | {reg['Corruptos/Nulos']:<8} | {icono} {reg['Acción / Estado']}")

print(f"\n DATASET ANALÍTICO MAESTRO:")
print(f"   - Total de siniestros enriquecidos listos para análisis avanzado: {len(df_master)}")
print("="*95 + "\n")