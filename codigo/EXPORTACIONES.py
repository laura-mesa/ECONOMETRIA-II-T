#
# EXPO FLORES
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


# ---
# BASE DE DATOS
# ---
#%%

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datos"

FILE_P = DATA_DIR / "EXPORTACIONES.xlsx"

#%%

df = df.rename(columns={df.columns[0]: "Fecha", df.columns[6]: "Flores"})

df["Fecha"] = pd.to_datetime(df["Fecha"], format="%b-%y", errors="coerce")
df["Flores"] = pd.to_numeric(df["Flores"], errors="coerce")

df = df.dropna(subset=["Fecha", "Flores"])

serie = df.set_index("Fecha")["Flores"]


# ----
# BOX-JENKINS: ANÁLISIS 
# ---

#%%
plt.figure(figsize=(12,6))
plt.plot(serie, color="green", linewidth=2)
plt.title("Exportaciones de flores en Colombia (2008-2026)", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Valor exportado")
plt.grid(True)
plt.show()

#%%
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(serie, lags=36, ax=ax[0])
plot_pacf(serie, lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()


# ---
# ESTABILIZACIÓN DE VARIANZA
# ---

#%%
stat, pval, f_stat, f_pval = het_arch(serie, nlags=12)

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
serie_log = np.log(serie) #TRANFORMACIÓN

plt.figure(figsize=(12,6))
plt.plot(serie_log, color="purple", linewidth=1.5)
plt.title("Serie con Logaritmo ", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Cambio logarítmico")
plt.grid(True)
plt.show()


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


#%%
serie_diff_estacional = serie_log.diff().diff(12).dropna() #TRANSFORMACIÓN: Diferenciación regular y estacional (d=1, D=1)

plt.figure(figsize=(12,6))
plt.plot(serie_diff_estacional, color="purple", linewidth=2)
plt.title("Exportaciones de flores - Diferenciación Regular y Estacional (d=1, D=1)", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Cambio en el valor exportado")
plt.grid(True)
plt.axhline(0, color='red', linestyle='--')
plt.show()

#%%
resultado_adf_est = adfuller(serie_diff_estacional, regression='n', autolag='AIC')

print('--- Resultados de la Prueba ADF (Serie Regular + Estacional) ---')
print(f'Estadístico ADF: {resultado_adf_est[0]:.4f}')
print(f'p-valor: {resultado_adf_est[1]:.4f}')
print(f'Número de rezagos usados: {resultado_adf_est[2]}')

print('\n--- Valores Críticos ---')
for key, value in resultado_adf_est[4].items():
    print(f'  {key}: {value:.4f}')


# ---
# BOX-JENKINS: IDENTIFICACIÓN
# ---

#%%
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(serie_diff_estacional, lags=36, ax=ax[0])
plot_pacf(serie_diff_estacional, lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()


# ---
# BOX-JENKINS: ESTIMACIÓN
# ---

#%%
modelo1 = ARIMA(serie_diff_estacional, order=(1,1,1))
resultado1 = modelo1.fit()
print(resultado1.summary())

#%%
modelo2 = ARIMA(serie_diff_estacional, order=(1,1,2))
resultado2 = modelo2.fit()
print(resultado2.summary())

#%%
modelo3 = ARIMA(serie_diff_estacional, order=(2,1,2))
resultado3 = modelo3.fit()
print(resultado3.summary())

#%%
modelo3 = ARIMA(serie_diff_estacional, order=(3,1,2))
resultado3 = modelo3.fit()
print(resultado3.summary())
