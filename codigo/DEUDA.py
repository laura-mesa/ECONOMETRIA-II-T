# Librerias
# ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
)  # NUEVA: autocorrelación de residuos
from statsmodels.stats.stattools import jarque_bera  # NUEVA: normalidad de residuos
# %%


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Metodología Box & Jenkinks para la serie de tiempo
#  de la deuda fiscal en Colombia (2001 - 2004 (jun))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Indice
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 0. Limpieza datos
# 1. Identificación del modelo
# 2. Estimación de parámetros
# 3. Verificación de supuestos
# 4. Uso del modelo (pronóstico)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 0. Limpieza de datos
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datos"

FILE_P = DATA_DIR / "DEUDA.xlsx"


df = pd.read_excel(FILE_P, header=None, skiprows=3)


# Organización de la base
datos_deuda = df.iloc[:, 3].dropna().values

fechas = pd.date_range(
    start="2001-06-01", periods=len(datos_deuda), freq="MS"
)  # MS = Month Start

df_deuda = pd.DataFrame({"Fecha": fechas, "deuda_externa": datos_deuda})

df_deuda["deuda_externa"] = pd.to_numeric(df_deuda["deuda_externa"], errors="coerce")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Identificación modelo
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1.a.Estacionariedad en la serie original

# a.1)Análisis de la varianza y tendencia estocástica de forma gráfica:
# ~~~~~~~~~~~~~~

# plot Serie de la deuda de Colombia (2001-2024)
plt.figure(figsize=(12, 6))
plt.plot(
    df_deuda["Fecha"],
    df_deuda["deuda_externa"] / 1_000_000,
    color="#2a78d6",
    linewidth=2,
)
plt.title("Deuda de Colombia, 2001-2024", fontsize=15, loc="left", fontweight="bold")
plt.xlabel("Año")
plt.ylabel("Billones de pesos (COP)")
plt.grid(axis="y", alpha=0.3)
plt.box(False)
plt.show()


# FAC y FACP de la serie original
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(
    df_deuda["deuda_externa"],
    lags=48,
    ax=ax[0],
    alpha=None,
    title="",
    marker="",
    vlines_kwargs={"colors": "green"},
)
plot_pacf(
    df_deuda["deuda_externa"],
    lags=48,
    ax=ax[1],
    alpha=None,
    title="",
    marker="",
    vlines_kwargs={"colors": "green"},
    method="ywm",
)
limite = 1.96 / np.sqrt(len(df_deuda["deuda_externa"]))  # bandas de confianza del 95%
for eje in ax:
    eje.axhline(limite, color="darkgray", linestyle=":")
    eje.axhline(-limite, color="darkgray", linestyle=":")
    eje.set_xticks(range(0, 49, 3))
    eje.set_ylim(-1.1, 1.1)  # eje Y de -1 a 1 (con un pequeño margen)
ax[1].set_ylabel("Autocorrelación parcial")
ax[0].set_ylabel("Autocorrelación")
plt.tight_layout()
plt.show()


# La FAC decae lento a 0: por lo tanto se piensa que la serie no es estacionaria,
# por otra parte la varianza se observa con un crecimiento leve


# a.2). Pruebas formales para conocimiento de los datos y transformaciones
# ~~~~~~~~~~~~~~~~~~

# Prueba ARCH para heterocedasticidad
stat, pval, f_stat, f_pval = het_arch(df_deuda["deuda_externa"], nlags=12)

print("--- Prueba de Heterocedasticidad ARCH (Serie Original) ---")
print(f"Estadístico ARCH (LM): {stat:.4f}")
print(f"p-valor (LM): {pval:.4f}")
print(f"Estadístico F: {f_stat:.4f}")
print(f"p-valor (F): {f_pval:.4f}")

if pval < 0.05:
    print(
        "\nConclusión: Se RECHAZA la hipótesis nula. Hay evidencia de Heterocedasticidad (la varianza NO es constante)."
    )
    print(
        "-> Se justifica plenamente aplicar el LOGARITMO para estabilizar la varianza."
    )
else:
    print(
        "\nConclusión: NO se rechaza la hipótesis nula. No hay evidencia suficiente de Heterocedasticidad."
    )


# transformación logaritmica de la serie
serie_log = np.log(df_deuda["deuda_externa"])  # TRANFORMACIÓN
serie_log.index = fechas

# Gráfica de la serie en logaritmo
plt.figure(figsize=(12, 6))
plt.plot(serie_log, color="purple", linewidth=2)
plt.title("Serie con Logaritmo", fontsize=15, loc="left", fontweight="bold")
plt.xlabel("Año")
plt.ylabel("Log de la deuda")
plt.grid(axis="y", alpha=0.3)
plt.box(False)
plt.show()


# Se reitera la verificación de heterocedasticidad con ARCH en serie logarítmica
stat, pval, f_stat, f_pval = het_arch(serie_log, nlags=12)

print("--- Prueba de Heterocedasticidad ARCH (Serie en Logaritmo) ---")
print(f"Estadístico ARCH (LM): {stat:.4f}")
print(f"p-valor (LM): {pval:.4f}")
print(f"Estadístico F: {f_stat:.4f}")
print(f"p-valor (F): {f_pval:.4f}")

if pval < 0.05:
    print(
        "\nConclusión: Se RECHAZA la hipótesis nula. Hay evidencia de Heterocedasticidad (la varianza NO es constante)."
    )
else:
    print(
        "\nConclusión: NO se rechaza la hipótesis nula. No hay evidencia suficiente de Heterocedasticidad."
    )
# Por ahora la transformación logarítmica no
# permite volver la varianza constante aún


# comprobación de la existencia de tendencia estocástica


resultado_adf = adfuller(serie_log, regression="ct", autolag="AIC")

print("--- Resultados de la Prueba ADF ---")
print(f"Estadístico ADF: {resultado_adf[0]:.4f}")
print(f"p-valor: {resultado_adf[1]:.4f}")
print(f"Número de rezagos usados: {resultado_adf[2]}")
print(f"Observaciones usadas: {resultado_adf[3]}")

print("\n--- Valores Críticos ---")
for key, value in resultado_adf[4].items():
    print(f"  {key}: {value:.4f}")
# p-valor > 0.05: la serie en log tiene raíz unitaria,
# hay evidencia para pensar que existe una tendencia estocástica
# por lo tanto, hay que aplicar el operador diferencia.


# Modelo de caminata aleatoria
# ~~~~~~~~~~~~~~~

serie_diff = serie_log.diff().dropna()

# Grafica de serie diferenciada I(1)
plt.figure(figsize=(12, 6))
plt.plot(serie_diff, color="grey", linewidth=2)
plt.title(
    "Deuda - Primera diferencia del logaritmo (d=1)",
    fontsize=15,
    loc="left",
    fontweight="bold",
)
plt.xlabel("Año")
plt.ylabel("Variación mensual (log)")
plt.grid(axis="y", alpha=0.3)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.box(False)
plt.show()

# Comprobación final en modelo de caminata aleatoria


# (A)DF sobre la serie diferenciada (d=1)

resultado_adf_d1 = adfuller(serie_diff, regression="c", autolag="AIC")
# regression='c' porque la serie diferenciada
#  tiene media distinta de cero (crecimiento promedio)
print("--- Resultados de la Prueba ADF (Primera diferencia) ---")
print(f"Estadístico ADF: {resultado_adf_d1[0]:.4f}")
print(f"p-valor: {resultado_adf_d1[1]:.4f}")
print(f"Número de rezagos usados: {resultado_adf_d1[2]}")

print("\n--- Valores Críticos ---")
for key, value in resultado_adf_d1[4].items():
    print(f"  {key}: {value:.4f}")

if resultado_adf_d1[1] < 0.05:
    print(
        "\nConclusión: Se RECHAZA la raíz unitaria. La serie diferenciada es ESTACIONARIA -> d = 1"
    )


# 1.b. Asociar p,q a un modelo ARMA
# ~~~~~~~~~~~~~~~~~~

# FAC y FACP de la serie log diferenciada (d=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(
    serie_diff,
    lags=48,
    ax=ax[0],
    alpha=None,
    title="",
    marker="",
    vlines_kwargs={"colors": "green"},
)
plot_pacf(
    serie_diff,
    lags=48,
    ax=ax[1],
    alpha=None,
    title="",
    marker="",
    vlines_kwargs={"colors": "green"},
    method="ywm",
)
limite = 1.96 / np.sqrt(len(serie_diff))  # bandas de confianza del 95%
for eje in ax:
    eje.axhline(limite, color="darkgray", linestyle=":")
    eje.axhline(-limite, color="darkgray", linestyle=":")
    eje.set_xticks(range(0, 49, 3))
    eje.set_ylim(-1.1, 1.1)  # eje Y de -1 a 1 (con un pequeño margen)
ax[1].set_ylabel("Autocorrelación parcial")
ax[0].set_ylabel("Autocorrelación")
plt.tight_layout()
plt.show()

# Los rezagos 1 y 2 salen en el límite de las bandas;
# Además no hay picos en 12, 24, 36 (no hay estacionalidad).

# Candidatos: ARIMA(1,1,0), (0,1,1), (2,1,0), (1,1,1)


# 1.b  Creación tabla de resumen de modelos
# ~~~~~~~~~~~~~~~~~~~~~~~~~


# Modelos a comparar. Todos llevan constante: trend='t' es la deriva (crecimiento promedio mensual)
modelos = [(1, 1, 0), (2, 1, 0), (1, 1, 2), (2, 1, 2), (2, 1, 3)]


# Se estiman los modelos y se guardan en un diccionario: {nombre del modelo: estimación}
estimaciones = {}
for orden in modelos:
    nombre = "ARIMA" + str(orden).replace(" ", "")
    estimaciones[nombre] = ARIMA(serie_log, order=orden, trend="t").fit()

nombres_modelos = list(estimaciones.keys())

# Parámetros que se reportan: (nombre en la tabla, nombre que les da statsmodels)
parametros = [
    ("constante", "x1"),
    ("a1", "ar.L1"),
    ("a2", "ar.L2"),
    ("b1", "ma.L1"),
    ("b2", "ma.L2"),
    ("b3", "ma.L3"),
]

# Inicialmente la tabla de modelos es una lista de python
tabla_modelos = []

# Se itera sobre los keys y values del diccionario "estimaciones"
for nombre, resultado in estimaciones.items():
    # nombre: es la variable que itera en el nombre de los modelos
    # resultado: es la variable que itera en la estimación de los modelos
    params = resultado.params  # Estimaciones de los parámetros del modelo
    errores = resultado.bse  # Errores estándar de las estimaciones
    pvalores = resultado.pvalues  # p-valores (para las estrellas)

    # Residuos sin el primero (con d=1 el primer residuo no es real)
    residuos = resultado.resid[1:]

    fila = {"Modelo": nombre}

    # Estimación, error estándar y p-valor de cada parámetro.
    # Si el modelo no tiene ese parámetro, queda vacío (np.nan)
    for etiqueta, nombre_param in parametros:
        fila[etiqueta] = params.get(nombre_param, np.nan)
        fila["se_" + etiqueta] = errores.get(nombre_param, np.nan)
        fila["p_" + etiqueta] = pvalores.get(nombre_param, np.nan)

    # Criterios de información y p-valores de las pruebas sobre los residuos
    fila["AIC"] = resultado.aic
    fila["BIC"] = resultado.bic
    fila["Ljung-Box"] = acorr_ljungbox(residuos, lags=[12])["lb_pvalue"].iloc[
        0
    ]  # H0: ruido blanco
    fila["ARCH"] = het_arch(residuos, nlags=12)[1]  # H0: varianza constante
    fila["Jarque-Bera"] = jarque_bera(residuos)[1]  # H0: normalidad

    # Se llena una a una la lista, con los resultados de cada modelo (lista de diccionarios)
    tabla_modelos.append(fila)

# Transformar la lista en un pandas DataFrame (cada fila es un modelo)
tabla_modelos_te = pd.DataFrame(tabla_modelos).set_index("Modelo")

# Nota: Hasta acá ya tenemos una tabla funcional.


# Estrellas de significancia: *** 1%, ** 5%, * 10%
def estrellas(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


# Especificar los valores de las estimaciones en la tabla (con estrellas)
def formato_estimacion(valor, pvalor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"{valor:.{decimales}f}{estrellas(pvalor)}"


# Especificar los valores de los errores estándar en la tabla
def formato_error_estandar(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"({valor:.{decimales}f})"


# Acá se genera la tabla final de publicación.

# Primero la tabla final es una lista
tabla_publicacion = []

# Parámetros: una fila con la estimación y debajo otra con el error estándar
for etiqueta, _ in parametros:
    fila_estimacion = {"Parámetro": etiqueta}
    fila_error = {"Parámetro": ""}

    for nombre in nombres_modelos:
        modelo = tabla_modelos_te.loc[nombre]
        fila_estimacion[nombre] = formato_estimacion(
            modelo[etiqueta], modelo["p_" + etiqueta]
        )
        fila_error[nombre] = formato_error_estandar(modelo["se_" + etiqueta])

    tabla_publicacion.append(fila_estimacion)
    tabla_publicacion.append(fila_error)

# AIC y BIC (1 decimal) y p-valores de las pruebas (3 decimales)
for medida in ["AIC", "BIC", "Ljung-Box", "ARCH", "Jarque-Bera"]:
    fila = {"Parámetro": medida}
    decimales = 1 if medida in ["AIC", "BIC"] else 3

    for nombre in nombres_modelos:
        fila[nombre] = f"{tabla_modelos_te.loc[nombre, medida]:.{decimales}f}"

    tabla_publicacion.append(fila)

# Genera la tabla final como un Pandas DataFrame
tabla_modelos_te_publicacion = pd.DataFrame(tabla_publicacion)

# Imprimir la tabla final en formato de texto
print("\nTabla resumen de modelos estimados")
print(tabla_modelos_te_publicacion.to_string(index=False))
print("\nErrores estándar entre paréntesis.")
print(
    "*, ** y *** indican significancia estadística al 10 %, 5 % y 1 %, respectivamente."
)
print(
    "Ljung-Box (12 rezagos), ARCH (12 rezagos) y Jarque-Bera: p-valores. p > 0.05 indica que no se rechaza H0 al 5 %."
)

# %%
# # Exportar la tabla como imagen (PNG)

# Letra estilo publicación (STIX, parecida a la de LaTeX), viene incluida con matplotlib
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"


# Pone las estrellas como superíndice: "0.009***" -> "0.009$^{***}$"
def estrellas_imagen(texto):
    n = texto.count("*")
    if n == 0:
        return texto
    return texto.replace("*", "") + "$^{" + "*" * n + "}$"


datos = tabla_modelos_te_publicacion
filas_parametros = (
    len(parametros) * 2
)  # filas de parámetros (estimación + error estándar)
ancho = len(datos.columns) + 0.5  # ancho total de la tabla (1 unidad por columna)

fig, ax = plt.subplots(figsize=(16, 7))
ax.axis("off")  # sin ejes, solo la tabla
ax.set_xlim(0, ancho)
ax.set_ylim(len(datos) + 1, -2)  # eje Y invertido: la fila 0 queda arriba

# Título
ax.text(
    ancho / 2,
    -1.5,
    "Modelos ARIMA estimados para la deuda de Colombia",
    ha="center",
    fontsize=15,
)

# Fila de nombres de los modelos
for j, nombre in enumerate(nombres_modelos):
    ax.text(j + 1.5, 0, nombre, ha="center", va="center", fontsize=13)

# Contenido: primera columna a la izquierda, las demás centradas
for i, fila in datos.iterrows():
    y = i + 1
    for j, celda in enumerate(fila):
        if j == 0:
            ax.text(0, y, celda, ha="left", va="center", fontsize=12)
        else:
            ax.text(
                j + 0.5,
                y,
                estrellas_imagen(celda),
                ha="center",
                va="center",
                fontsize=12,
            )

# Líneas: gruesa arriba, delgadas en el medio, gruesa abajo
ax.plot([0, ancho], [-0.6, -0.6], color="black", linewidth=1.5)
ax.plot([0, ancho], [0.5, 0.5], color="black", linewidth=0.8)
ax.plot(
    [0, ancho],
    [filas_parametros + 0.5, filas_parametros + 0.5],
    color="black",
    linewidth=0.8,
)
ax.plot([0, ancho], [len(datos) + 0.5, len(datos) + 0.5], color="black", linewidth=1.5)

# Nota debajo de la tabla
nota = (
    "Nota: estimaciones por máxima verosimilitud con statsmodels ARIMA(); errores estándar entre paréntesis. "
    "La constante es la deriva del modelo.\n"
    "Para Ljung-Box (12 rezagos), ARCH (12 rezagos) y Jarque-Bera se reportan p-valores.\n"
    "$^{*}$p < 0.10; $^{**}$p < 0.05; $^{***}$p < 0.01. Una, dos y tres estrellas indican significancia estadística "
    "a los niveles del 10 %, 5 % y 1 %, respectivamente."
)
ax.text(0, len(datos) + 0.9, nota, ha="left", va="top", fontsize=9)

plt.show()

plt.rcdefaults()  # vuelve a la letra normal para los siguientes gráficos


# Análisis resultados de la tabla
# ~~~~~~~~~~~~~~~~


# Modelo elegido: ARIMA(2,1,0) con deriva
# ya que tiene el menor AIC entre los modelos con TODOS
# los coeficientes significativos (su BIC es casi igual al del ARIMA(1,1,0))

modelo_final = ARIMA(serie_log, order=(2, 1, 0), trend="t").fit()
print(modelo_final.summary())

# En la tabla, revisar la columna P>|z|: todos los coeficientes deben tener p-valor < 0.05


# %%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Estimación de parámetros
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Tabla de resumen ARIMA(2,1,0)
texto = str(modelo_final.summary())  # el resumen convertido en texto

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")  # sin ejes, solo el texto
ax.text(0, 1, texto, family="monospace", fontsize=9, va="top", transform=ax.transAxes)
plt.show()

# ▲ ln(D_t) = 0.009  -0.121  ▲ (D_t-2)  +0.123  ▲ (D_t-1)

# Al trabajar sobre diferenciales se observa en los coeficientes el impacto que al
# momento actual, repercute la aceleración del aumento de la deuda
# (en 1 y 2 meses antes el incremento de la deuda se asocia a esa
# repercusión porcentual, pero en aceleraciones de la variación de la deuda).


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Verificación de supuestos
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 3.1Gráficas del modelo
# ~~~~~~~~~~


# Gráfica de los residuos del modelo ARIMA(2,1,0) / sin incluir el 1er residuo
residuos = modelo_final.resid[1:]
plt.figure(figsize=(12, 4))
plt.plot(residuos, color="gray")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuos del modelo ARIMA(2,1,0)", fontsize=14)
plt.grid(True)

plt.figtext(
    0.125,
    0.01,
    "Nota: Se omite el primer residuo derivado del proceso de diferenciación (d=1) para una visualización más limpia.",
    ha="left",
    fontsize=9,
    color="black",
    fontfamily="STIXGeneral",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.show()


# FAC y FACP de los residuos del modelo ARIMA(2,1,0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(
    residuos,
    lags=48,
    ax=ax[0],
    alpha=None,
    title="",
    marker="",
    vlines_kwargs={"colors": "green"},
)
plot_pacf(
    residuos,
    lags=48,
    ax=ax[1],
    alpha=None,
    title="",
    marker="",
    vlines_kwargs={"colors": "green"},
    method="ywm",
)
limite = 1.96 / np.sqrt(len(residuos))  # bandas de confianza del 95%
for eje in ax:
    eje.axhline(limite, color="darkgray", linestyle=":")
    eje.axhline(-limite, color="darkgray", linestyle=":")
    eje.set_xticks(range(0, 49, 3))
    eje.set_ylim(-1.1, 1.1)  # eje Y de -1 a 1 (con un pequeño margen)
ax[1].set_ylabel("Autocorrelación parcial")
ax[0].set_ylabel("Autocorrelación")
plt.tight_layout()
plt.show()


# Histograma de los residuos del modelo ARIMA(2,1,0)
plt.figure(figsize=(8, 4))
plt.hist(residuos, bins=30, color="steelblue", edgecolor="black")
plt.title("Histograma de los residuos")
plt.show()


# 3.2 Pruebas formales del Modelo ARIMA (2,1,0)


# a)Ruido blanco: Ljung-Box. H0: los residuos NO están autocorrelacionados
ljung = acorr_ljungbox(residuos, lags=[12, 24])
print("--- Prueba de Ljung-Box ---")
print(ljung)
if (ljung["lb_pvalue"] > 0.05).all():
    print("Conclusión: NO se rechaza H0. Los residuos son ruido blanco.\n")
else:
    print("Conclusión: Se rechaza H0. Queda autocorrelación en los residuos.\n")


# b)Normalidad: Jarque-Bera. H0: los residuos son normales
jb_stat, jb_pval, asimetria, curtosis = jarque_bera(residuos)
print("--- Prueba de Jarque-Bera ---")
print(f"Estadístico JB: {jb_stat:.4f}   p-valor: {jb_pval:.4f}")
if jb_pval > 0.05:
    print("Conclusión: Los residuos son normales.\n")
else:
    print(
        "Conclusión: Los residuos NO son normales (colas pesadas). Los intervalos del pronóstico deben leerse con cuidado.\n"
    )


# c) Varianza constante: ARCH. H0: no hay heterocedasticidad
stat, pval, f_stat, f_pval = het_arch(residuos, nlags=12)
print("--- Prueba ARCH sobre los residuos ---")
print(f"Estadístico ARCH (LM): {stat:.4f}   p-valor: {pval:.4f}")
if pval > 0.05:
    print("Conclusión: Varianza constante en los residuos.")
else:
    print("Conclusión: Hay heterocedasticidad en los residuos.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Pronóstico
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Pronóstico preliminar de PRUEBA: como estima el MODELO ARIMA(2,1,0) el año transcurrido 2024.
# ~~~~~~~~~~~~~~~~~

entrenamiento = serie_log[:-12]
prueba = serie_log[-12:]

modelo_prueba = ARIMA(entrenamiento, order=(2, 1, 0), trend="t").fit()
pron_prueba = modelo_prueba.forecast(steps=12)

real = np.exp(prueba)  # se devuelve el logaritmo a pesos con exp()
estimado = np.exp(pron_prueba)
mape = np.mean(np.abs((real - estimado) / real)) * 100
print(f"MAPE en los últimos 12 meses: {mape:.2f}%")  # Justificar que es MAPE

# Gráfica de la evaluación del pronóstico
plt.figure(figsize=(12, 6))
plt.plot(np.exp(serie_log[-48:]), color="green", label="Real", linewidth=2)
plt.plot(estimado, color="orange", linestyle="--", label="Pronóstico (prueba)")
plt.title(
    "Evaluación del modelo: últimos 12 meses",
    fontsize=15,
    loc="left",
    fontweight="bold",
)
plt.xlabel("Año-mes ")
plt.ylabel("Log de la deuda")
plt.grid(axis="y", alpha=0.3)
plt.box(False)
plt.show()


# Pronóstico hacia adelante: 12 meses con el modelo final (toda la muestra) para el 2025
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pronostico = modelo_final.get_forecast(steps=12)

pron_media = np.exp(pronostico.predicted_mean)
pron_ic = np.exp(pronostico.conf_int(alpha=0.05))  # intervalo de confianza del 95%

tabla_pronostico = pd.DataFrame(
    {
        "Pronóstico": pron_media,
        "Límite inferior": pron_ic.iloc[:, 0],
        "Límite superior": pron_ic.iloc[:, 1],
    }
)
print(tabla_pronostico.round(0).astype(int))


# Gráfica del pronóstico de la deuda
plt.figure(figsize=(12, 6))
plt.plot(np.exp(serie_log), color="green", label="Deuda observada", linewidth=2)
plt.plot(pron_media, color="red", label="Pronóstico")
plt.fill_between(
    pron_ic.index,
    pron_ic.iloc[:, 0],
    pron_ic.iloc[:, 1],
    color="red",
    alpha=0.2,
    label="IC 95%",
)
plt.title(
    "Pronóstico de la Deuda - ARIMA(2,1,0)", fontsize=15, loc="left", fontweight="bold"
)
plt.xlabel("Año")
plt.ylabel("Valor de la deuda")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.box(False)
plt.show()
