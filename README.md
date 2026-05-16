# Data Wrangling & Statistical Analysis for ONSV (Observatorio Nacional de Seguridad Vial)

Este proyecto realiza un análisis estadístico, preprocesamiento, estructuración, normalización y limpieza de los datos del Observatorio Nacional de Seguridad Vial, cuyos documentos poseen información de siniestros de tránsito, ocurridos a nivel nacional, 2008 - 2025 (Perú). Las cifras toman como fuente de información los Anuarios Estadísticos de la Policia Nacional del Perú. 


Portal de datos abiertos: [Observatorio Nacional de Seguridad Vial - ONSV](https://www.onsv.gob.pe/datosabiertos)

Mapa del Perú con límites distritales: [GeoJSON Perú](https://www.arcgis.com/home/item.html?id=6d183ac55a604ce1959458d1ac6f05d8)

Datos Georreferenciados : [Geo Perú](https://visor.geoperu.gob.pe/)

## 📁 Estructura del Repositorio

- `pipeline.sh`: Script para la ejecución secuencial de los procesos previos.
- `data_ingestion.py`: Script para la importación de los datos en formato csv y Json.
- `data_wrangling`: Script para el data wrangling process.
- `data_analysis`: Script para el análisis estadístico y visualización.

---