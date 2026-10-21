# ---
# LIBRERÍAS
# ---

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm


# ---
# BASE DE DATOS
# ---

#%%
ruta_script = Path(__file__).resolve()
BASE_DIR = ruta_script.parent.parent
FILE_P = BASE_DIR / "datos" / "PIB00.xlsx" # <-- Asegúrate que el nombre sea EXACTO

print(f"Buscando archivo en: {FILE_P}")

# ---
# ORGANIZACIÓN DE LA BASE
# ---

#%%

datos_raw = df.iloc[14, 4:].values

# Filtramos para quedarnos solo con los datos que son números (ignorando textos o vacíos)
datos = [d for d in datos_raw if pd.notna(d)]

print(f"\nSe encontraron {len(datos)} datos.")
print(f"Primeros 4 datos: {datos[:4]}")
print(f"Últimos 4 datos: {datos[-4:]}")

# Creamos las fechas (2000-I, 2000-II, etc.)
año_inicio = 2000
trimestres_romanos = ['I', 'II', 'III', 'IV']

fechas = []
for i in range(len(datos)):
    año = año_inicio + (i // 4)
    trimestre = trimestres_romanos[i % 4]
    fechas.append(f"{año}-{trimestre}")

# Creamos el DataFrame final
df_limpio = pd.DataFrame({
    'Fecha': fechas,
    'PIB_Agricultura': datos
})

# Aseguramos que la columna sea numérica
df_limpio['PIB_Agricultura'] = pd.to_numeric(df_limpio['PIB_Agricultura'], errors='coerce')

print("\n--- PRIMEROS 5 DATOS ---")
print(df_limpio.head())
print("\n--- ÚLTIMOS 5 DATOS ---")
print(df_limpio.tail())
print(f"\nTotal de observaciones: {len(df_limpio)}")

# ---
# ANÁLISIS 
# ---

#%%

plt.figure(figsize=(14, 6))
plt.plot(
    df_limpio['Fecha'],
    df_limpio['PIB_Agricultura'],
    marker='.',
    linestyle='-',
    color='green'
)

plt.title(
    'PIB Agricultura (Series desestacionalizadas, precios constantes)',
    fontsize=14
)

plt.xlabel('Trimestre', fontsize=12)
plt.ylabel('Valor (Miles de millones de pesos)', fontsize=12)

paso = 4
plt.xticks(df_limpio['Fecha'][::paso], rotation=45)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ---
# FILTRO HP Y CONSTRUCCIÓN DEL GAP
# ---

#%%

df_limpio['log_PIB'] = np.log(df_limpio['PIB_Agricultura'])

ciclo, tendencia = sm.tsa.filters.hpfilter(
    df_limpio['log_PIB'],
    lamb=1600
)

df_limpio['Tendencia_HP'] = tendencia
df_limpio['GAP'] = ciclo

print("Datos con GAP calculado:")
print(df_limpio[['Fecha', 'PIB_Agricultura', 'log_PIB', 'GAP']].head())


#%%

plt.figure(figsize=(14, 6))

plt.plot(
    df_limpio['Fecha'],
    df_limpio['log_PIB'],
    label='PIB Agrícola (Log)',
    color='blue',
    linewidth=1
)

plt.plot(
    df_limpio['Fecha'],
    df_limpio['Tendencia_HP'],
    label='Tendencia HP',
    color='red',
    linestyle='--',
    linewidth=2
)

plt.title
# %%

# Graficar el GAP (La variable que vas a modelar con el AR(1))
plt.figure(figsize=(14, 6))
plt.plot(df_limpio['Fecha'], df_limpio['GAP'], color='purple', marker='.', linestyle='-')
plt.axhline(0, color='black', linestyle='--', linewidth=1) # Línea en cero

plt.title('Brecha del Producto Agrícola (GAP)', fontsize=14)
plt.xlabel('Trimestre', fontsize=12)
plt.ylabel('Desviación porcentual (aprox.)', fontsize=12)
paso = 4
plt.xticks(df_limpio['Fecha'][::paso], rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# ACF y PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(df_limpio['GAP'], lags=36, ax=ax[0])
plot_pacf(df_limpio['GAP'], lags=36, ax=ax[1], method="ywm")
plt.tight_layout()
plt.show()

# %%
modelo1= ARIMA(df_limpio['GAP'], order=(1,0,0))
resultado1 = modelo1.fit()
print(resultado1.summary())

#%%
modelo2= ARIMA(df_limpio['GAP'], order=(1,0,1))
resultado2 = modelo2.fit()
print(resultado2.summary())
#%%
modelo3= ARIMA(df_limpio['GAP'], order=(1,1,1))
resultado3 = modelo3.fit()
print(resultado3.summary())
#%%
modelo4= ARIMA(df_limpio['GAP'], order=(2,0,0))
resultado4 = modelo4.fit()
print(resultado4.summary())

#%%
modelo5= ARIMA(df_limpio['GAP'], order=(1,1,0))
resultado5 = modelo5.fit()
print(resultado5.summary())

# %%
