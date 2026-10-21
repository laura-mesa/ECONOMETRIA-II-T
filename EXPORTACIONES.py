#
#EXPO FLORES
#

#%%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

#%%
df = pd.read_excel(r"C:\Users\lalas\OneDrive - Universidad Nacional de Colombia\Universidad\9no sem\Econometria II\Repositorio\ECONOMETRIA-II-T\EXPORTACIONES.xlsx", header=0)            # si es el .txt de NOAA: pd.read_csv("oni.txt", sep=r"\s+")

# La columna A tiene las fechas (mes-año) y la G las exportaciones de flores
df = df.rename(columns={df.columns[0]: "Fecha", df.columns[6]: "Flores"})

# Convertir la columna de fechas a tipo datetime
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%b-%y", errors="coerce")

# Convertir la columna de flores a numérica (forzando errores a NaN)
df["Flores"] = pd.to_numeric(df["Flores"], errors="coerce")

# Eliminar filas vacías o con NaN
df = df.dropna(subset=["Fecha", "Flores"])

# Crear la serie de tiempo
serie = df.set_index("Fecha")["Flores"]

#%%
# Grafica
plt.figure(figsize=(12,6))
plt.plot(serie, color="green", linewidth=2)
plt.title("Exportaciones de flores en Colombia (2008-2026)", fontsize=14)
plt.xlabel("Año")
plt.ylabel("Valor exportado")
plt.grid(True)
plt.show()

#%%
# ACF y PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(serie, lags=36, ax=ax[0])
plot_pacf(serie, lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()

#%%

# Ajuste de modelos ARIMA
modelo1= ARIMA(serie, order=(1,1,1))
resultado1 = modelo1.fit()
print(resultado1.summary())

modelo2= ARIMA(serie, order=(1,1,2))
resultado2 = modelo2.fit()
print(resultado2.summary())

modelo3= ARIMA(serie, order=(2,1,2))
resultado3 = modelo3.fit()
print(resultado3.summary())
