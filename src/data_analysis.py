import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("[EDA] Iniciando Análisis Estadístico, Relaciones Latentes y Mapas Geoespaciales Per Cápita...")

# 1. Cargar el Dataset Maestro
df = pd.read_csv('data/processed/02_dataset_analitico.csv')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.max_open_warning': 0})


# DASHBOARD 1: CONTEXTO AMBIENTAL Y VIAL

print(" -> Generando Dashboard 1: Contexto Ambiental...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('EDA 1: Contexto Ambiental y Clasificación', fontsize=18, fontweight='bold', y=0.98)

sns.countplot(data=df, x='ZONA', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('1. Distribución por Zona', fontweight='bold')
sns.countplot(data=df, x='CONDICIÓN CLIMÁTICA', ax=axes[0, 1], palette='coolwarm')
axes[0, 1].set_title('2. Condición Climática', fontweight='bold')
sns.countplot(data=df, y='TIPO DE VÍA', ax=axes[1, 0], palette='pastel', order=df['TIPO DE VÍA'].value_counts().index)
axes[1, 0].set_title('3. Incidencia por Tipo de Vía', fontweight='bold')
sns.countplot(data=df, y='CLASE SINIESTRO', ax=axes[1, 1], palette='viridis', order=df['CLASE SINIESTRO'].value_counts().index)
axes[1, 1].set_title('4. Clasificación del Siniestro', fontweight='bold')
plt.tight_layout(); plt.savefig('results/plots/01_dashboard_contexto.png', dpi=300); plt.close()


# DASHBOARD 2 Y 3: CAUSAS, GRAVEDAD E INFRACCIONES

print(" -> Generando Dashboard 2 y 3...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('EDA 2: Temporalidad y Consecuencias', fontsize=18, fontweight='bold', y=0.98)
sns.histplot(data=df[df['HORA_ENTERA'] >= 0], x='HORA_ENTERA', bins=24, kde=True, ax=axes[0, 0], color='darkred')
axes[0, 0].set_title('5. Distribución Horaria', fontweight='bold')
sns.countplot(data=df, y='CAUSA FACTOR PRINCIPAL', ax=axes[0, 1], order=df['CAUSA FACTOR PRINCIPAL'].value_counts().nlargest(10).index, palette='Reds_r')
axes[0, 1].set_title('6. Top 10 Causas', fontweight='bold')
sns.boxplot(data=df, x='CANTIDAD DE FALLECIDOS', ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('7. Gravedad (Fallecidos)', fontweight='bold')
sns.boxplot(data=df, x='CANTIDAD DE LESIONADOS', ax=axes[1, 1], color='lightblue')
axes[1, 1].set_title('8. Gravedad (Lesionados)', fontweight='bold')
plt.tight_layout(); plt.savefig('results/plots/02_dashboard_causas.png', dpi=300); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('EDA 3: Perfil Humano y Vehicular', fontsize=18, fontweight='bold', y=1.05)
sns.barplot(data=pd.DataFrame({'M': ['Hombres', 'Mujeres', 'Vehículos'], 'V': [df['cond_hombres'].sum(), df['cond_mujeres'].sum(), df['total_vehiculos'].sum()]}), x='M', y='V', ax=axes[0], palette=['#1f77b4', '#e377c2', 'gray'])
axes[0].set_title('9 y 10. Demografía y Volumen', fontweight='bold')
irreg = pd.DataFrame({'F': ['Ebriedad', 'Licencia Irregular', 'Sin SOAT', 'Sin CITV', 'Motos'], 'S': [(df['cond_estado_ebriedad']>0).sum(), (df['cond_licencia_irregular']>0).sum(), (df['vehiculos_sin_soat']>0).sum(), (df['vehiculos_sin_citv']>0).sum(), (df['motos_involucradas']>0).sum()]}).sort_values('S', ascending=False)
sns.barplot(data=irreg, x='S', y='F', ax=axes[1], palette='magma')
axes[1].set_title('11 al 15. Infracciones y Riesgos', fontweight='bold')
plt.tight_layout(); plt.savefig('results/plots/03_dashboard_infracciones.png', dpi=300); plt.close()


# MATRIZ DE CORRELACIÓN EXPANDIDA (16 Variables)

print(" -> Generando Matriz de Correlación Completa (16 variables)...")
df_corr = df.copy()
for col in ['ZONA', 'CONDICIÓN CLIMÁTICA', 'TIPO DE VÍA', 'CLASE SINIESTRO', 'CAUSA FACTOR PRINCIPAL']:
    df_corr[col] = df_corr[col].astype('category').cat.codes

dieciseis_variables = [
    'CANTIDAD DE FALLECIDOS', 'CANTIDAD DE LESIONADOS', 'total_vehiculos', 
    'cond_hombres', 'cond_mujeres', 'cond_licencia_irregular', 'cond_estado_ebriedad', 
    'motos_involucradas', 'vehiculos_sin_soat', 'vehiculos_sin_citv', 'HORA_ENTERA',
    'ZONA', 'CONDICIÓN CLIMÁTICA', 'TIPO DE VÍA', 'CLASE SINIESTRO', 'total_pers'
]

plt.figure(figsize=(16, 12))
sns.heatmap(df_corr[dieciseis_variables].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-0.4, vmax=1)
plt.title('Matriz de Correlación Lineal (16 Variables Críticas)', fontweight='bold', fontsize=16)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('results/plots/04_matriz_correlacion.png', dpi=300); plt.close()


# ANÁLISIS BIVARIADO DE RELACIONES LATENTES

print(" -> Generando Análisis Bivariados de Relaciones Latentes...")

plt.figure(figsize=(10, 6))
df_distrital = df.groupby('KEY_UBICACION').agg({
    'CANTIDAD DE FALLECIDOS': 'sum', 
    'CÓDIGO SINIESTRO': 'count',  # Frecuencia total de siniestros
    'total_pers': 'first'
}).reset_index()

df_distrital.rename(columns={'CÓDIGO SINIESTRO': 'TOTAL_SINIESTROS'}, inplace=True)

sns.regplot(data=df_distrital, x='total_pers', y='CANTIDAD DE FALLECIDOS', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Relación Latente 1: Impacto del Tamaño Poblacional en la Mortalidad', fontweight='bold', fontsize=14)
plt.xlabel('Población Total del Distrito (Censo)')
plt.ylabel('Víctimas Fatales Acumuladas')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout(); plt.savefig('results/plots/05_bivariado_poblacion_fatalidad.png', dpi=300); plt.close()

plt.figure(figsize=(12, 6))
df_horas = df[df['HORA_ENTERA'] >= 0]
sns.kdeplot(data=df_horas[df_horas['cond_estado_ebriedad'] == 0], x='HORA_ENTERA', fill=True, label='Sobrios', color='blue', alpha=0.3)
sns.kdeplot(data=df_horas[df_horas['cond_estado_ebriedad'] > 0], x='HORA_ENTERA', fill=True, label='Ebriedad (Dosaje +)', color='red', alpha=0.5)
plt.title('Relación Latente 2: Concentración Horaria según Estado Etílico', fontweight='bold', fontsize=14)
plt.xlabel('Hora del Día (0-23)'); plt.legend(); plt.xticks(range(0, 24, 2))
plt.tight_layout(); plt.savefig('results/plots/06_bivariado_hora_ebriedad.png', dpi=300); plt.close()


# MODELADO GEOESPACIAL: CARTOGRAFÍA Y TASAS PER CÁPITA

print(" -> Procesando cartografía y métricas espaciales per cápita...")

mapa_peru = gpd.read_file('data/raw/Distritos_Peru_v1.geojson')
for col in ['NOMBDEP', 'NOMBPROV', 'NOMBDIST']:
    mapa_peru[col] = mapa_peru[col].fillna('').str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
mapa_peru['KEY_UBICACION'] = mapa_peru['NOMBDEP'] + '-' + mapa_peru['NOMBPROV'] + '-' + mapa_peru['NOMBDIST']

# Fusión Espacial
mapa_datos = mapa_peru.merge(df_distrital, on='KEY_UBICACION', how='left')
mapa_datos['CANTIDAD DE FALLECIDOS'] = mapa_datos['CANTIDAD DE FALLECIDOS'].fillna(0)
mapa_datos['TOTAL_SINIESTROS'] = mapa_datos['TOTAL_SINIESTROS'].fillna(0)

# CÁLCULO DE TASAS PER CÁPITA (Para ambos mapas)
mapa_datos['total_pers'] = mapa_datos['total_pers'].fillna(1)
mapa_datos['Tasa_Mortalidad_100k'] = (mapa_datos['CANTIDAD DE FALLECIDOS'] / mapa_datos['total_pers']) * 100000
mapa_datos['Tasa_Siniestros_100k'] = (mapa_datos['TOTAL_SINIESTROS'] / mapa_datos['total_pers']) * 100000
mapa_datos['centroide'] = mapa_datos.geometry.centroid


# SECCIÓN A: MAPAS DE TASA DE MORTALIDAD PER CÁPITA (Rojos)

print(" -> Exportando Mapas de Mortalidad Per Cápita...")
mapa_depto = mapa_datos[mapa_datos['NOMBDEP'] == 'LIMA']
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
mapa_depto.plot(column='Tasa_Mortalidad_100k', cmap='YlOrRd', linewidth=0.5, edgecolor='black', legend=True, 
                legend_kwds={'label': "Tasa de Mortalidad (x 100,000 hab.)", 'shrink': 0.6}, ax=ax)
ax.set_title('Tasa de Mortalidad Vial Per Cápita: Región Lima', fontsize=20, fontweight='bold')
ax.axis('off')
for idx, row in mapa_depto.iterrows():
    ax.annotate(text=row['NOMBDIST'], xy=(row['centroide'].x, row['centroide'].y), ha='center', va='center', fontsize=7, alpha=0.8)
plt.tight_layout(); plt.savefig('results/plots/07A_mapa_lima_departamento_percapita.png', dpi=300); plt.close()

mapa_metro_callao = mapa_datos[(mapa_datos['NOMBPROV'] == 'LIMA') | (mapa_datos['NOMBDEP'] == 'CALLAO')]
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
mapa_metro_callao.plot(column='Tasa_Mortalidad_100k', cmap='YlOrRd', linewidth=0.6, edgecolor='black', legend=True, 
                       legend_kwds={'label': "Tasa de Mortalidad (x 100,000 hab.)", 'shrink': 0.6}, ax=ax)
ax.set_title('Tasa de Mortalidad Vial Per Cápita: Lima Metropolitana y Callao', fontsize=20, fontweight='bold')
ax.axis('off')
for idx, row in mapa_metro_callao.iterrows():
    ax.annotate(text=row['NOMBDIST'], xy=(row['centroide'].x, row['centroide'].y), ha='center', va='center', fontsize=9, color='darkblue', weight='bold', alpha=0.9)
plt.tight_layout(); plt.savefig('results/plots/07B_mapa_lima_metropolitana_callao_percapita.png', dpi=300); plt.close()


# SECCIÓN B: MAPAS DE TASA DE SINIESTRALIDAD PER CÁPITA (Púrpuras)

print(" -> Exportando Mapas de Siniestralidad Per Cápita...")

# MAPA 1: Escala Regional - Tasa de Siniestros
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
mapa_depto.plot(column='Tasa_Siniestros_100k', cmap='PuBu', linewidth=0.5, edgecolor='black', legend=True, 
                legend_kwds={'label': "Tasa de Siniestros (x 100,000 hab.)", 'shrink': 0.6}, ax=ax)
ax.set_title('Tasa de Siniestralidad Vial Per Cápita: Región Lima', fontsize=20, fontweight='bold')
ax.axis('off')
for idx, row in mapa_depto.iterrows():
    ax.annotate(text=row['NOMBDIST'], xy=(row['centroide'].x, row['centroide'].y), ha='center', va='center', fontsize=7, alpha=0.8)
plt.figtext(0.15, 0.15, "Fuente: ONSV e INEI | Nota: Tasa calculada sobre siniestros con víctimas fatales", fontsize=10, color='gray')
plt.tight_layout(); plt.savefig('results/plots/08A_mapa_lima_departamento_siniestros_percapita.png', dpi=300); plt.close()

# MAPA 2: Escala Urbana - Tasa de Siniestros
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
mapa_metro_callao.plot(column='Tasa_Siniestros_100k', cmap='PuBu', linewidth=0.6, edgecolor='black', legend=True, 
                       legend_kwds={'label': "Tasa de Siniestros (x 100,000 hab.)", 'shrink': 0.6}, ax=ax)
ax.set_title('Tasa de Siniestralidad Vial Per Cápita: Lima Metropolitana y Callao', fontsize=20, fontweight='bold')
ax.axis('off')
for idx, row in mapa_metro_callao.iterrows():
    ax.annotate(text=row['NOMBDIST'], xy=(row['centroide'].x, row['centroide'].y), ha='center', va='center', fontsize=9, color='darkblue', weight='bold', alpha=0.9)
plt.figtext(0.15, 0.15, "Fuente: ONSV e INEI | Nota: Tasa calculada sobre siniestros con víctimas fatales", fontsize=10, color='gray')
plt.tight_layout(); plt.savefig('results/plots/08B_mapa_lima_metropolitana_callao_siniestros_percapita.png', dpi=300); plt.close()

print("\n=================================================================")
print(" EDA COMPLETO: Todos los 4 Mapas han sido normalizados a Tasas Per Cápita.")
print("=================================================================\n")