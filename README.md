# DATALAB-CHALLENGE

# 🌾 ¿Puede la IA anticipar los precios del agro?

**Pronóstico a 12 meses del Índice de Precios Agropecuarios (IPAP-BMC)**
---

## Tabla de contenido

- [Acerca del proyecto](#acerca-del-proyecto)
- [Contexto: el IPAP-BMC](#contexto-el-ipap-bmc)
- [Datos](#datos)
- [Enfoque](#enfoque)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Primeros pasos](#primeros-pasos)
- [Uso](#uso)
- [Reproducibilidad](#reproducibilidad)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Referencias](#referencias)
- [Equipo, licencia y contacto](#equipo-udea)

---

## Acerca del proyecto

Los precios agropecuarios cambian mucho y son difíciles de anticipar. Productores, comercializadores, agroindustria, entidades financieras y de política sectorial toman decisiones (qué y cuánto producir, cuándo comprar, cómo cubrirse del riesgo de precio) que mejorarían con una idea anticipada de hacia dónde va el índice.

Este repositorio responde al **Reto B** del BMC DataLab Challenge: construir un modelo de predicción a **12 meses** del IPAP-BMC, comparando enfoques estadísticos clásicos y de *machine learning*, evaluados frente a un *baseline* mediante métricas de error (**RMSE** y **MAPE**) y **validación temporal**.

Todo el trabajo (análisis, modelos, comparación y generación del tablero) vive en un único notebook: [`DATALAB_CHALLENGE.ipynb`](DATALAB_CHALLENGE.ipynb).

**Qué incluye**

- Estadística descriptiva y diagnóstico de la serie (estacionariedad, ACF/PACF, estacionalidad).
- Exploración de variables exógenas con correlación cruzada y causalidad de Granger.
- Ocho enfoques comparados: Naïve, ARIMA/ARIMAX, ETS, Prophet, Elastic Net, XGBoost, LSTM y GRU.
- Pronóstico de jul-2026 a jun-2027 con bandas de incertidumbre.
- Un tablero interactivo complementario (`tablero_ipap.html`) que se regenera con cada ejecución.

---

## Contexto: el IPAP-BMC

El **Índice de Precios Agropecuarios de la Bolsa Mercantil de Colombia (IPAP-BMC)** es un indicador de precios agropecuarios a nivel nacional, publicado por la Gerencia Corporativa de Analítica y Estudios Económicos de la BMC. Su metodología está documentada en [BMC (2022)](#referencias).

**Fuente de información: el registro de facturas.** A diferencia de otros indicadores del sector, que se basan en encuestas y entrevistas en plazas de mercado, el IPAP-BMC se construye con el registro de facturas de la BMC: transacciones de productos agropecuarios y agroindustriales realizadas fuera del escenario de negociación de la bolsa y registradas por firmas comisionistas. La base reúne unos 50 millones de facturas entre 2006 y 2022 (en promedio, 20.000 diarias), con más de 2.000 productos «padre». Su ventaja principal es la **inmediatez**: los datos están disponibles con menor rezago que los de las encuestas.

**Cómo se construye el índice** (resumen de la metodología oficial):

| Etapa | Descripción |
|---|---|
| 1. Selección de productos | Se usan las subclases CPC con las que el DANE calcula el IPP agropecuario y se conservan los productos con registros en más del 80 % del periodo analizado. Queda excluida la caña de azúcar por no cumplir esa condición |
| 2. Limpieza de facturas | Detección de atípicos por producto y mes con bandas intercuartílicas (Q1 − 3·RIQ, Q3 + 3·RIQ) sobre ventanas móviles de 360 días |
| 3. Series de tiempo | Agregación de facturas por fecha y segunda limpieza de atípicos con descomposición STL (LOESS): se marcan los residuos fuera de ±2 desviaciones estándar y se reemplazan por tendencia + componente periódico |
| 4. Imputación | En series incompletas se imputan valores con *Dynamic Time Warping* (DTW) para hallar la serie completa más similar, ajustada en nivel y tendencia |
| 5. Agregación | Índice de **Laspeyres** con las ponderaciones del IPP agropecuario del DANE (Encuesta Nacional Agropecuaria 2014). **Base: diciembre de 2014** |

El índice resultante se organiza por CPC en 16 grupos, 30 clases, 33 subclases y 142 productos. Los de mayor peso son ganado bovino (16,9 %), leche cruda de vaca (11,6 %), pollos y gallinas (9,0 %), huevo (7,4 %) y maíz (6,0 %), que juntos representan cerca de la mitad del índice.

**Validación del indicador.** La documentación reporta una alta correlación con el IPP agropecuario del DANE (0,99 en niveles y 0,92 en crecimiento anual), con el IPC de alimentos (0,88) y con el IPC de bienes no durables (0,82, ambos en variación anual), además de una relación no contemporánea con la tasa de cambio, lo que sugiere explorar rezagos de variables externas, tal como se hace aquí.

---

## Datos

| Archivo | Contenido | Cobertura |
|---|---|---|
| `Data/Datos.csv` | IPAP-BMC e IMP (mensual) | IMP desde ene-2010; IPAP desde ene-2011 hasta jun-2026 |
| `Data/Datos Tentativos.csv` | IPAP, TRM, IMP, TEMP y VOL | ene-2011 → jun-2026 (186 observaciones) |

Formato de los CSV: separador `;`, decimal `,` y fecha con el patrón `AAAAMmm` (por ejemplo, `2011M01`).

| Variable | Descripción |
|---|---|
| `IPAP` | Índice de Precios Agropecuarios BMC (**variable objetivo**) |
| `TRM` | Tasa representativa del mercado (tipo de cambio) |
| `IMP` | Valor de las importaciones mediante el índice de términos de intercambio |
| `TEMP` | Temperatura media del aire en el país |
| `VOL` | Volumen de transacciones de los 5 principales mercados que componen el IPAP-BMC |

**Comportamiento de la serie.** El IPAP presenta tendencia creciente y varianza heterogénea, por lo que **no es estacionario**. Su distribución es asimétrica positiva (0,63), platicúrtica (curtosis en exceso de −1,21) y difiere de una normal. Rango observado: 88,4 – 256,7; media 151,6; desviación estándar 56,7.

---

## Enfoque

**Validación hold-out en tres etapas** (común a todos los modelos):

1. **Entrenamiento** con la historia hasta jun-2025 (174 observaciones, por encima del mínimo sugerido de 48).
2. **Prueba** sobre los 12 meses siguientes (jul-2025 a jun-2026), sin incorporar sus valores reales; se calculan RMSE y MAPE.
3. **Reentrenamiento** con toda la serie y pronóstico final de jul-2026 a jun-2027.

**Cómo se elige la especificación de cada modelo** (sin usar la prueba, salvo donde se indica):

- **ETS, Elastic Net y XGBoost:** validación *hold-out* de 12 meses dentro del entrenamiento (ajuste hasta jun-2024 y validación jul-2024 a jun-2025). Se escoge la especificación de menor MAPE en esa ventana.
- **ARIMA/ARIMAX:** el orden (*p*, *q*) se elige por AIC sobre el entrenamiento. La decisión de incluir o no el IMP rezagado entre las dos finalistas se tomó con el MAPE de la prueba.
- **LSTM y GRU:** validación cruzada temporal (`TimeSeriesSplit`, 5 folds y hueco de 12 ventanas) con grilla gruesa y grilla fina, minimizando el RMSE.
- **Prophet:** los parámetros se afinaron por tanteo con el MAPE de la prueba, y además se reporta el *backtest* nativo del paquete.

Se comparan siempre contra un **baseline Naïve** (repetir el último valor observado): un modelo solo aporta valor si lo supera.

**Modelos**

| Familia | Modelo | Idea central |
|---|---|---|
| Baseline | Naïve | Repite el último valor observado |
| Clásico | ARIMA/ARIMAX | ARIMA(p,1,q) sobre log(IPAP) con deriva y exógena opcional (IMP rezagado 12 meses). Se evalúan las versiones con y sin IMP |
| Clásico | ETS | Suavizamiento exponencial; 36 especificaciones evaluadas por hold-out de 12 meses, también el promedio de las dos mejores |
| Aditivo | Prophet | Tendencia lineal por tramos (25 puntos de cambio candidatos, rango 0,8, flexibilidad 0,05); sin estacionalidad anual |
| ML | Elastic Net | Regresión penalizada, estrategia directa multi-horizonte (un modelo por mes pronosticado); 4 especificaciones de variables |
| ML | XGBoost | *Boosting* de árboles con la misma estrategia directa, pérdida de error absoluto y grilla conservadora; 4 especificaciones de variables |
| Deep learning | LSTM, GRU | Redes recurrentes con salida directa de 12 pasos y ensamble de 10 semillas |

**Pruebas estadísticas aplicadas**

| Prueba | Para qué se usa | Resultado |
|---|---|---|
| ADF y KPSS | Estacionariedad y orden de diferenciación | Serie no estacionaria en nivel; estacionaria con una diferencia (*d* = 1) |
| ACF / PACF y STL | Autocorrelación y estacionalidad | Órdenes bajos; sin patrón estacional anual claro, por lo que se descartan modelos estacionales |
| CCF con *prewhitening* | Rezagos en que una exógena antecede al IPAP | Solo IMP se considera relevante (rezagos 0, 6, 12 y 24; máximo 0,31 en el rezago 12) |
| Granger (rezagos 1-12) | ¿La exógena ayuda a predecir el IPAP? | IMP p = 0,011 (rezago 12) · TRM p = 0,041 (rezago 1) · TEMP p = 0,424 · VOL p = 0,191 |
| AIC / BIC | Orden (*p*, *q*) del ARIMA | AIC selecciona (0,1,2); BIC prefiere un modelo más simple, (0,1,0) |
| Ljung-Box | Residuos como ruido blanco | No se rechaza ausencia de autocorrelación (p = 0,82; 0,20; 0,34 en los rezagos 6, 12 y 24)¹ |
| Jarque-Bera | Normalidad de residuos | Se rechaza (colas pesadas); los intervalos son aproximados |

¹ Diagnóstico del ARIMAX con IMP estimado con el entrenamiento.

Los detalles de cada modelo (hiperparámetros, grillas, diagnósticos y gráficas) están documentados paso a paso en el notebook.

---

## Resultados

> **Resultados con datos hasta jun-2026.** Al agregar nuevos meses y reejecutar el notebook, las métricas y el pronóstico cambian; la versión vigente está en el notebook (secciones 3, 4 y 5) y en `tablero_ipap.html`.

Desempeño en la prueba hold-out (jul-2025 a jun-2026), ordenado por MAPE. Menor es mejor.

| Modelo | RMSE | MAPE | MAPE validación² |
|---|---:|---:|---:|
| **ARIMA(0,1,2) + deriva**³ | **5,25** | **1,74 %** | n. a. |
| Prophet | 6,64 | 2,36 % | n. a. |
| Elastic Net (compacto, sin IMP) | 10,68 | 3,81 % | 4,94 % |
| XGBoost (compacto, con IMP) | 11,78 | 4,32 % | 4,15 % |
| Naïve (baseline) | 14,10 | 4,90 % | n. a. |
| GRU⁴ | 18,79 | 6,17 % | n. a. |
| ETS(A,N,M) | 18,03 | 7,01 % | 1,23 % |
| LSTM⁴ | 23,88 | 7,56 % | n. a. |

² MAPE en la ventana de validación hold-out (jul-2024 a jun-2025), solo para los modelos elegidos con ese procedimiento.
³ El modelo seleccionado es la versión **sin** el regresor IMP; la versión con IMP rezagado (ARIMAX) obtuvo RMSE 7,83 y MAPE 2,84 % en la misma prueba.
⁴ Alta variabilidad entre ejecuciones (una sola corrida en prueba): resultados no significativos.

**Conclusiones principales**

- El mejor modelo (ARIMA sin IMP) reduce el RMSE ≈ 63 % y el MAPE ≈ 64 % frente al Naïve.
- Prophet, Elastic Net y XGBoost también superan al baseline en esta ventana; ETS, GRU y LSTM no.
- ETS tuvo el mejor MAPE en validación (1,23 %) pero uno de los peores en prueba (7,01 %): la elección entre 36 especificaciones con una sola ventana de 12 meses es muy sensible a esa ventana.
- Con solo 186 observaciones mensuales, los modelos de *machine learning* no superan a los clásicos (ARIMA y Prophet); con más datos o mayor periodicidad podrían mejorar.
- El IMP rezagado es significativo en el entrenamiento (y mejora el AIC en ≈ 9,5 puntos), pero no mejora el pronóstico fuera de muestra: sin IMP el MAPE es 1,74 % frente a 2,84 % con IMP.
- En la prueba el IPAP subió de forma sostenida y los modelos con deriva (ARIMA y Prophet) quedan cerca del valor real; los demás tienden a subestimar o a oscilar.

**Pronóstico final (jul-2026 → jun-2027).** El ARIMA(0,1,2) con deriva proyecta el índice de ≈ 253,7 a ≈ 269,6, con un intervalo del 90 % que se ensancha hasta ≈ 230–315 al final del horizonte. Los demás modelos terminan en jun-2027 entre ≈ 250 (Prophet) y ≈ 275 (XGBoost).

---

## Estructura del repositorio

```text
.
├── DATALAB_CHALLENGE.ipynb     # Notebook principal (análisis, modelos, comparación, tablero)
├── generar_tablero.py          # Función construir_tablero() que genera el HTML
├── Data/
│   ├── Datos.csv               # IPAP + IMP
│   └── Datos Tentativos.csv    # IPAP + TRM + IMP + TEMP + VOL
├── tablero_ipap.html           # Tablero interactivo (salida generada)
├── requirements.txt            # Dependencias
└── README.md
```

Secciones del notebook: **1)** estadística descriptiva · **2)** correlación y estacionalidad · **3)** modelos econométricos (ARIMA/ARIMAX, ETS, Prophet) · **4)** *machine learning* (Elastic Net, XGBoost, LSTM, GRU) · **5)** comparación y pronóstico final · **6)** tablero interactivo.

---

## Primeros pasos

### Requisitos

- Python 3.10 o superior
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`
- `statsmodels`, `prophet`
- `scikit-learn`, `xgboost`
- `tensorflow` (Keras)
- `jupyterlab`

### Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Uso

```bash
jupyter lab DATALAB_CHALLENGE.ipynb
```

Ejecuta el notebook completo (**Run All**). Tiempos aproximados (dependen del equipo): ETS y Elastic Net, menos de un minuto; XGBoost, unos 6 minutos; las redes LSTM y GRU tardan más por la búsqueda de hiperparámetros con validación cruzada. Al terminar se genera `tablero_ipap.html` en la misma carpeta.

**Actualización mensual**

1. Agrega el nuevo mes a `Data/Datos.csv` y `Data/Datos Tentativos.csv`.
2. Ejecuta el notebook completo: las fechas de entrenamiento, prueba (últimos 12 meses) y pronóstico (12 meses siguientes) se desplazan solas.
3. Abre `tablero_ipap.html` (requiere `generar_tablero.py` en la misma carpeta).

Para sumar un modelo nuevo, añade una entrada al diccionario `modelos` de la sección 6: aparece como una pestaña más y entra en la comparación. Cada entrada puede incluir las claves opcionales `validacion` (cómo se eligió el modelo), `usa_prueba` (si la prueba intervino en la elección) y `salvedades` (aclaraciones propias), que el tablero muestra en la pestaña del modelo y en el Resumen.

---

## Reproducibilidad

- Semilla global `SEED = 13` (simulación de intervalos de ETS, submuestreo de XGBoost y TensorFlow).
- Los intervalos de ETS con componentes multiplicativos se obtienen por simulación y pueden variar unas décimas entre corridas.
- LSTM y GRU son sensibles a la inicialización; por eso el pronóstico final es la mediana de 10 redes y la métrica de prueba (una sola corrida) debe leerse con cautela.
- Todas las particiones son temporales y el código verifica que no haya fuga de información hacia la prueba (en el ARIMAX, el `IMP_l12` de la prueba usa solo valores ya observados; en Elastic Net y XGBoost, ningún objetivo de la prueba entra al entrenamiento).

---

## Limitaciones y trabajo futuro

**Limitaciones**

- **Muestra pequeña** (186 observaciones mensuales), que restringe a los modelos de ML y de *deep learning*.
- **Una sola ventana de prueba y una sola de validación**, en un periodo de alza sostenida: el ranking podría cambiar en otra ventana (ETS es el ejemplo: primero en validación y entre los últimos en prueba).
- **Decisiones tomadas con la prueba:** los parámetros de Prophet y la inclusión o no del IMP en el ARIMA se eligieron mirando el MAPE de los últimos 12 meses, por lo que el error reportado puede ser algo optimista.
- **Intervalos aproximados:** residuos no normales en el ARIMA (el intervalo supone normalidad en logaritmos); bandas empíricas en Elastic Net y XGBoost construidas con solo 12 errores relativos, con el mismo ancho relativo en todos los horizontes; en LSTM/GRU la banda solo refleja la dispersión entre semillas.
- **Exógenas limitadas:** IMP aporta asociación predictiva (no causal) y solo con rezago de 12 meses. Otras candidatas (commodities internacionales, petróleo, IPC de alimentos, IPP-DANE, ENSO/IDEAM) no se incorporaron por disponibilidad y rezago de publicación.

**Trabajo futuro**

- Validación *walk-forward* con múltiples orígenes para todos los modelos y comparación formal de precisión (Diebold-Mariano, MASE).
- Series con mayor periodicidad o más largas; la propia metodología del IPAP-BMC anticipa que su fuente de datos permitiría indicadores semanales o quincenales.
- Combinación de pronósticos (*ensembles*) entre los mejores modelos.
- Prueba piloto de 6 a 12 meses para medir la capacidad predictiva real.
- Tablero de consulta con todos los modelos y su pronóstico para apoyar decisiones de gestión de riesgo, producción y apoyos financieros.

---

## Referencias

- Bolsa Mercantil de Colombia, Gerencia Corporativa de Analítica y Estudios Económicos. (2022, julio). *Índice de Precios Agropecuarios IPAP-BMC: Metodología*. Bolsa Mercantil de Colombia.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3.ª ed.). OTexts.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5.ª ed.). Wiley.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. J. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics, 6*(1), 3–33.
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician, 72*(1), 37–45.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B, 67*(2), 301–320.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD*, 785–794.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP*.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica, 37*(3), 424–438.

---

## Equipo UdeA

- **Equipo:** Miguel Ángel Patiño | Maria Camila Ramirez | Luisa Fernanda Rendón
