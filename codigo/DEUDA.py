# %%
#
# DEUDA
#

#---
# LIBRERÍAS
#---
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox      # NUEVA: autocorrelación de residuos
from statsmodels.stats.stattools import jarque_bera          # NUEVA: normalidad de residuos


# ---
# BASE DE DATOS
# ---
#%%

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datos"

FILE_P = DATA_DIR / "DEUDA.xlsx"


df = pd.read_excel(FILE_P, header=None, skiprows=3)
#%%
# ---
# ORGANIZACIÓN DE LA BASE
# ---

#%%
# OJO: en el Excel la columna 3 es "Deuda total (COP millones)".
#      Deuda externa está en la columna 2 (COP millones) o en la 5 (USD millones).
datos_deuda = df.iloc[:, 3].dropna().values

# CORREGIDO: el Excel empieza en junio de 2001 (no en enero de 2002) y termina en junio de 2024
fechas = pd.date_range(start='2001-06-01', periods=len(datos_deuda), freq='MS') # MS = Month Start

df_deuda = pd.DataFrame({
    'Fecha': fechas,
    'deuda_externa': datos_deuda
})

df_deuda['deuda_externa'] = pd.to_numeric(df_deuda['deuda_externa'], errors='coerce')

# %%
## ----
# BOX-JENKINS: ANÁLISIS 
# ---

#%%
plt.figure(figsize=(12,6))
plt.plot(df_deuda['Fecha'], df_deuda['deuda_externa'], color="green", linewidth=2)
plt.title("Deuda de Colombia (2001-2024)", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Valor de la deuda")
plt.grid(True)
plt.show()

#%%
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(df_deuda['deuda_externa'], lags=36, ax=ax[0])
plot_pacf(df_deuda['deuda_externa'], lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()


# ---
# ESTABILIZACIÓN DE VARIANZA
# ---

#%%
stat, pval, f_stat, f_pval = het_arch(df_deuda['deuda_externa'], nlags=12)

print('--- Prueba de Heterocedasticidad ARCH (Serie Original) ---')
print(f'Estadístico ARCH (LM): {stat:.4f}')
print(f'p-valor (LM): {pval:.4f}')
print(f'Estadístico F: {f_stat:.4f}')
print(f'p-valor (F): {f_pval:.4f}')

if pval < 0.05:
    print("\nConclusión: Se RECHAZA la hipótesis nula. Hay evidencia de Heterocedasticidad (la varianza NO es constante).")
    print("-> Se justifica plenamente aplicar el LOGARITMO para estabilizar la varianza.")
else:
    print("\nConclusión: NO se rechaza la hipótesis nula. No hay evidencia suficiente de Heterocedasticidad.")

#%%
serie_log = np.log(df_deuda['deuda_externa']) #TRANFORMACIÓN
serie_log.index = fechas  # NUEVO: la serie lleva las fechas, así el pronóstico sale con fechas

plt.figure(figsize=(12,6))
plt.plot(serie_log, color="purple", linewidth=1.5)
plt.title("Serie con Logaritmo ", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Log de la deuda")
plt.grid(True)
plt.show()

# %%
stat, pval, f_stat, f_pval = het_arch(serie_log, nlags=12)

print('--- Prueba de Heterocedasticidad ARCH (Serie en Logaritmo) ---')
print(f'Estadístico ARCH (LM): {stat:.4f}')
print(f'p-valor (LM): {pval:.4f}')
print(f'Estadístico F: {f_stat:.4f}')
print(f'p-valor (F): {f_pval:.4f}')

if pval < 0.05:
    print("\nConclusión: Se RECHAZA la hipótesis nula. Hay evidencia de Heterocedasticidad (la varianza NO es constante).")
else:
    print("\nConclusión: NO se rechaza la hipótesis nula. No hay evidencia suficiente de Heterocedasticidad.")

# %%
# ---
# ESTACIONARIEDAD
# ---
#%%
resultado_adf = adfuller(serie_log, regression='ct', autolag='AIC')

print('--- Resultados de la Prueba ADF ---')
print(f'Estadístico ADF: {resultado_adf[0]:.4f}')
print(f'p-valor: {resultado_adf[1]:.4f}')
print(f'Número de rezagos usados: {resultado_adf[2]}')
print(f'Observaciones usadas: {resultado_adf[3]}')

print('\n--- Valores Críticos ---')
for key, value in resultado_adf[4].items():
    print(f'  {key}: {value:.4f}')

# p-valor > 0.05 -> la serie en log tiene raíz unitaria -> hay que diferenciar


#%%
# CORREGIDO: antes se hacía .diff().diff(1), que son DOS diferencias regulares (d=2),
# no una estacional. Con UNA diferencia (d=1) la serie ya es estacionaria.
serie_diff = serie_log.diff().dropna() #TRANSFORMACIÓN: Diferenciación regular (d=1)

plt.figure(figsize=(12,6))
plt.plot(serie_diff, color="purple", linewidth=2)
plt.title("Deuda - Primera diferencia del logaritmo (d=1)", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Variación mensual (log)")
plt.grid(True)
plt.axhline(0, color='red', linestyle='--')
plt.show()

#%%
# regression='c' porque la serie diferenciada tiene media distinta de cero (crecimiento promedio)
resultado_adf_d1 = adfuller(serie_diff, regression='c', autolag='AIC')

print('--- Resultados de la Prueba ADF (Primera diferencia) ---')
print(f'Estadístico ADF: {resultado_adf_d1[0]:.4f}')
print(f'p-valor: {resultado_adf_d1[1]:.4f}')
print(f'Número de rezagos usados: {resultado_adf_d1[2]}')

print('\n--- Valores Críticos ---')
for key, value in resultado_adf_d1[4].items():
    print(f'  {key}: {value:.4f}')

if resultado_adf_d1[1] < 0.05:
    print("\nConclusión: Se RECHAZA la raíz unitaria. La serie diferenciada es ESTACIONARIA -> d = 1")


# %%
# ---
# 1. IDENTIFICACIÓN
# ---
#%%
# Se miran la ACF (sugiere el orden q del MA) y la PACF (sugiere el orden p del AR)
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(serie_diff, lags=36, ax=ax[0])
plot_pacf(serie_diff, lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()

# Los rezagos 1 y 2 salen en el límite de las bandas y no hay picos en 12, 24, 36
# (no hay estacionalidad). Candidatos: ARIMA(1,1,0), (0,1,1), (2,1,0), (1,1,1)


# %%
# ---
# 2. ESTIMACIÓN
# ---
#%%
# trend='t' agrega la deriva (el crecimiento promedio de la deuda cada mes)
modelos = [(1,1,0), (0,1,1), (2,1,0), (1,1,1),
           (1,1,2), (2,1,2),
           (0,1,3), (1,1,3), (2,1,3), (3,1,3)]

print('--- Comparación de modelos (menor AIC / BIC es mejor) ---')
for orden in modelos:
    modelo = ARIMA(serie_log, order=orden, trend='t').fit()
    significativos = (modelo.pvalues.drop('sigma2') < 0.05).all()   # True si TODOS los coeficientes tienen p < 0.05
    print(f'ARIMA{orden}   AIC = {modelo.aic:.2f}   BIC = {modelo.bic:.2f}   Todos significativos: {significativos}')

#%%
# Para ver los coeficientes de un modelo con MA(3), por ejemplo el ARIMA(2,1,3):
print(ARIMA(serie_log, order=(2,1,3), trend='t').fit().summary())

#%%
# Modelo elegido: ARIMA(2,1,0) con deriva -> menor AIC entre los modelos con TODOS
# los coeficientes significativos (su BIC es casi igual al del ARIMA(1,1,0)).
# Los modelos con MA(3) tienen BIC más alto y coeficientes NO significativos
# (sobreparametrización), por eso se descartan
# Modelo elegido: ARIMA(2,1,0) con deriva -> menor AIC y todos los coeficientes significativos

modelo_final = ARIMA(serie_log, order=(2,1,0), trend='t').fit()
print(modelo_final.summary())

# En la tabla, revisar la columna P>|z|: todos los coeficientes deben tener p-valor < 0.05


# %%
# ---
# 3. DIAGNÓSTICO (VALIDACIÓN DE LOS RESIDUOS)
# ---
#%%
# Se quita el primer residuo: con d=1 el primer dato no tiene diferencia y su residuo no es real
residuos = modelo_final.resid[1:]

plt.figure(figsize=(12,4))
plt.plot(residuos, color="gray")
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuos del modelo ARIMA(2,1,0)", fontsize=14)
plt.grid(True)
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(residuos, lags=36, ax=ax[0])
plot_pacf(residuos, lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.hist(residuos, bins=30, color="steelblue", edgecolor="black")
plt.title("Histograma de los residuos")
plt.show()

#%%
# a) Ruido blanco: Ljung-Box. H0: los residuos NO están autocorrelacionados
ljung = acorr_ljungbox(residuos, lags=[12, 24])
print('--- Prueba de Ljung-Box ---')
print(ljung)
if (ljung['lb_pvalue'] > 0.05).all():
    print("Conclusión: NO se rechaza H0. Los residuos son ruido blanco.\n")
else:
    print("Conclusión: Se rechaza H0. Queda autocorrelación en los residuos.\n")

# b) Normalidad: Jarque-Bera. H0: los residuos son normales
jb_stat, jb_pval, asimetria, curtosis = jarque_bera(residuos)
print('--- Prueba de Jarque-Bera ---')
print(f'Estadístico JB: {jb_stat:.4f}   p-valor: {jb_pval:.4f}')
if jb_pval > 0.05:
    print("Conclusión: Los residuos son normales.\n")
else:
    print("Conclusión: Los residuos NO son normales (colas pesadas). Los intervalos del pronóstico deben leerse con cuidado.\n")

# c) Varianza constante: ARCH. H0: no hay heterocedasticidad
stat, pval, f_stat, f_pval = het_arch(residuos, nlags=12)
print('--- Prueba ARCH sobre los residuos ---')
print(f'Estadístico ARCH (LM): {stat:.4f}   p-valor: {pval:.4f}')
if pval > 0.05:
    print("Conclusión: Varianza constante en los residuos.")
else:
    print("Conclusión: Hay heterocedasticidad en los residuos.")


# %%
# ---
# 4. PRONÓSTICO
# ---
#%%
# a) Evaluación: se esconden los últimos 12 meses, se estima con el resto y se compara
entrenamiento = serie_log[:-12]
prueba = serie_log[-12:]

modelo_prueba = ARIMA(entrenamiento, order=(2,1,0), trend='t').fit()
pron_prueba = modelo_prueba.forecast(steps=12)

real = np.exp(prueba)            # se devuelve el logaritmo a pesos con exp()
estimado = np.exp(pron_prueba)
mape = np.mean(np.abs((real - estimado) / real)) * 100
print(f'MAPE en los últimos 12 meses: {mape:.2f}%')

plt.figure(figsize=(12,6))
plt.plot(np.exp(serie_log[-48:]), color="green", label="Real")
plt.plot(estimado, color="orange", linestyle="--", label="Pronóstico (prueba)")
plt.title("Evaluación del modelo: últimos 12 meses", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

#%%
# b) Pronóstico hacia adelante: 12 meses con el modelo final (toda la muestra)
pronostico = modelo_final.get_forecast(steps=12)

pron_media = np.exp(pronostico.predicted_mean)
pron_ic = np.exp(pronostico.conf_int(alpha=0.05))   # intervalo de confianza del 95%

tabla_pronostico = pd.DataFrame({
    'Pronóstico': pron_media,
    'Límite inferior': pron_ic.iloc[:, 0],
    'Límite superior': pron_ic.iloc[:, 1]
})
print(tabla_pronostico.round(0).astype(int))

plt.figure(figsize=(12,6))
plt.plot(np.exp(serie_log), color="green", label="Deuda observada")
plt.plot(pron_media, color="red", label="Pronóstico")
plt.fill_between(pron_ic.index, pron_ic.iloc[:, 0], pron_ic.iloc[:, 1],
                 color="red", alpha=0.2, label="IC 95%")
plt.title("Pronóstico de la Deuda - ARIMA(2,1,0) con deriva", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Valor de la deuda")
plt.legend()
plt.grid(True)
plt.show()


# %%