import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


#
#ONI
#
df = pd.read_excel(r"C:\Users\lalas\OneDrive - Universidad Nacional de Colombia\Universidad\9no sem\Econometria II\Repositorio\ECONOMETRIA-II-T\ONI.xlsx", header=0)

cols = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"] #columnas
df["Year"] = pd.to_numeric(df["Year"], errors="coerce") #Convertir los años en números
df = df.dropna(subset=["Year"]).copy() #Borarr filas con años nulos
print(df["Year"].min(), df["Year"].max(), len(df))
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

largo = df.melt(id_vars="Year", value_vars=cols, var_name="trim", value_name="oni")
largo["mes"] = largo["trim"].map({c: i+1 for i, c in enumerate(cols)}) # Mapear meses y crear fecha
largo["fecha"] = pd.to_datetime(dict(year=largo["Year"], month=largo["mes"], day=1))

oni = largo.set_index("fecha").sort_index()["oni"]

oni = oni.replace(-99.9, np.nan).dropna() # NOAA usa -99.9 para meses sin dato
print(f"Observaciones: {len(oni)}  ({oni.index.min():%Y-%m} a {oni.index.max():%Y-%m})")

# Gráfico de la serie
plt.figure(figsize=(12, 4))
oni.plot()
plt.axhline(0, color="k", lw=0.8)
plt.title("Índice ONI (1950-2026)")
plt.ylabel("ONI (°C)")
plt.show()

# ACF y PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(oni, lags=36, ax=ax[0])
plot_pacf(oni, lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()


modelo1= ARIMA(oni, order=(1,1,1))
resultado1 = modelo1.fit()
print(resultado1.summary())

modelo2= ARIMA(oni, order=(1,1,2))
resultado2 = modelo2.fit()
print(resultado2.summary())

modelo3= ARIMA(oni, order=(2,1,2))
resultado3 = modelo3.fit()
print(resultado3.summary())