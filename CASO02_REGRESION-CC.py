# Caso práctico 02_Regresion.ipynb

# Conexion a Google Colaborative
# ------------
# Se importa y utiliza drive para conectar nuestro entorno de Colab al drive de Google para cargar nuestro dataset.
from google.colab import drive
drive.mount('/gdrive')

# Pandas: Utilizada extensivamente para el análisis y manipulación de datos en forma de DataFrames en todo el modelo.
import pandas as pd
# Numpy: Útil para realizar cálculo científico, algebra lineal y operaciones eficientes en el arreglo (matrices).
import numpy as np
# Scipy.stats: Brinda herramientas estadísticas clásicas necesarias para evaluar modelos y distribuciones.
import scipy.stats as stats
# Matplotlib.pyplot: Fundamento para la creación de cualquier gráfica de uso general (barras, dispersión, líneas).
import matplotlib.pyplot as plt
# Seaborn: Montada encima de Matplotlib, se utiliza para hacer gráficos estadísticos (ej. violín, regplots) super amigables.
import seaborn as sns
# Random: Incorpora la generación de números pseudoaleatorios y selecciones al azar en caso de necesitar muestreos manuales.
import random
# Se configuran los parámetros por defecto para todos los gráficos de Seaborn en este notebook
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
# Se utiliza para ignorar los warnings o advertencias que pueden aparecer durante la ejecución del código
import warnings
warnings.filterwarnings('ignore')
# gdown es una utilidad para descargar archivos grandes desde Google Drive, aunque no se usa directamente después.
import gdown

# vivienda = pd.read_csv('/gdrive/MyDrive/2026_MINERIA_Parte_Practica/bostonvivienda.csv')
# Se carga el dataset de viviendas de Boston desde la ruta especificada en Google Drive
vivienda = pd.read_csv('/gdrive/MyDrive/Data/bostonvivienda.csv')

# Muestra las primeras 5 filas del DataFrame para una rápida inspección visual de los datos
vivienda.head()

# Devuelve las dimensiones del DataFrame (número de filas, número de columnas)
vivienda.shape

# Entendiendo las covariables y target!
# Convierte los nombres de las columnas en una lista para facilitar su visualización y manejo
vivienda.columns.to_list()

# Muestra un resumen de los tipos de datos de cada columna (ej. int64, float64, object)
vivienda.dtypes

"""### Regresión lineal"""
# ------------

# Verificacion de algunos supuestos de Regresión Lineal!

# Linealidad
# Se crea una matriz de gráficos de dispersión (pairplot) para visualizar la relación entre cada variable predictora y la variable objetivo 'medv'
p = sns.pairplot(vivienda, x_vars = ['crim','zn','indus','nox','rm','edad','dis','rad','impuesto','ptratio','negro','lstat',],
                 y_vars='medv', size=3, aspect=0.7)

# Se cumple el supuesto de Linealidad?
# (Este es un comentario para el analista, para que evalúe visualmente si la relación parece lineal)

# Multicolinealidad

#vivienda.corr()
# Se calcula la matriz de correlación de Spearman entre todas las variables numéricas del DataFrame
vivienda.corr(method = 'spearman') # method = 'spearman'

# El coeficiente de correlación de pearson nos mide la asociación o correlación lineal entre los datos,
# sin embargo es un coeficiente que requiere el supuesto de normalidad entre los datos para que sea preciso.

# Multicolinealidad
# Se calcula la matriz de correlación (por defecto, Pearson)
corr = vivienda.corr()
# Se crea una máscara para ocultar la parte superior derecha del mapa de calor (que es un espejo de la parte inferior izquierda)
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
# Se configura el tamaño de la figura del gráfico
f, ax = plt.subplots(figsize=(11, 9))
# Se define un mapa de colores divergente
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Se dibuja el mapa de calor (heatmap) con la matriz de correlación y la máscara
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Dividimos los datos en Covariables y Target!
# Se crea el DataFrame 'X' con todas las variables predictoras (todas las columnas menos 'medv')
X = vivienda.drop('medv',axis=1)
type(X)
# Se crea la Serie 'y' con la variable objetivo ('medv')
y = vivienda.medv
type(y)

# Modelo de Regresión Lineal!
# ------------

# Se importan las clases y funciones necesarias de scikit-learn para el modelado y la evaluación
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Particion Muestral de los datos!

# Creación de la data de train y la data de test
# Se importan las clases y funciones necesarias de scikit-learn para el modelado y la evaluación
from sklearn.model_selection import train_test_split
# Se dividen los datos 'X' e 'y' en conjuntos de entrenamiento y prueba (33% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=100) # Semilla aleatoria para reproducibilidad

# Se crea una instancia del modelo de Regresión Lineal
reg_multiple = LinearRegression()
# Se entrena el modelo con los datos de entrenamiento
reg_multiple.fit(X_train,y_train)

# Vemos los coeficientes o la importancia relativa  impacto de las variables sobre el objetivo!
# Se muestran los coeficientes de regresión para cada variable, ordenados de mayor a menor
print(pd.Series(reg_multiple.coef_, index = X_train.columns).sort_values(ascending = False)) # Mostrar los coeficientes

# La ordenada en el origen o intercepto
# Se muestra el término de intercepción (el valor de 'y' cuando todas las 'X' son cero)
reg_multiple.intercept_
#ese valor que me aparece es: es el precio base de la casa sin una influencia de ninguna variable.
#variable en miles, sería 33,980.13 dolares, precio base de la casa.

# Validamos nuestros resultados en el train y en el test!
# Se realizan predicciones sobre los conjuntos de entrenamiento y prueba
y_pred_train = reg_multiple.predict(X_train)
y_pred_test = reg_multiple.predict(X_test)

# Obtenemos las funciones de coste
# Se calcula y muestra el Error Cuadrático Medio (MSE) para ambos conjuntos
print("ECM: {}".format(mean_squared_error(y_train,y_pred_train)))
print("ECM: {}".format(mean_squared_error(y_test,y_pred_test)))
# Mientras menos es el error, mejor...

# Obtenemos las funciones de coste
# Se calcula y muestra la Raíz del Error Cuadrático Medio (RMSE) para ambos conjuntos
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train,y_pred_train))))
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test,y_pred_test))))
# 4.42 para arriba o para bajo, si el train va bien
# Train (entrenamiento) tiene que ser menor test (prueba)

# Obtenemos el valor R cuadrado o Coeficiente de Determinación!
# R2          = El % de la explicación de Y, dado por todas las variables del modelo!
# Se calcula y muestra el coeficiente R-cuadrado para ambos conjuntos
print("R^2: {}".format(r2_score(y_train,y_pred_train)))
print("R^2: {}".format(r2_score(y_test,y_pred_test)))

# Viendo la importancia de las variables y la contribucion de estas!
# ------------

# Se crea un DataFrame para visualizar la importancia de cada variable (basada en el valor absoluto de su coeficiente)
imp = pd.DataFrame({'Nombre_Variable':X_train.columns,
                    'Importancia':(np.absolute(reg_multiple.coef_))}).sort_values(
    'Importancia', ascending=False)
# Se muestra el DataFrame con barras para una mejor visualización de la importancia
imp.style.bar()

# Podemos tambien ver las contribuciones de las variables!
# ------------

# Se toma el primer registro del conjunto de prueba para analizar la contribución de sus variables
x0 = pd.DataFrame(X_test.iloc[0].rename('Valor_Variable'))
x0['Nombre_Variable'] = x0.index
# Se une con el DataFrame de importancia
x0 = pd.merge(x0, imp, on='Nombre_Variable')
# Se calcula la contribución de cada variable (Valor * Importancia)
x0['Contribucion'] = x0.Valor_Variable * x0.Importancia

# Se muestra la contribución de cada variable para ese registro, ordenada y con barras
x0.sort_values('Contribucion', ascending=False).style.bar(['Contribucion'])

# Podemos revisar la contribucion de cada variable sobre el objetivo!
# Se llama a una función 'waterfallplot' (no definida en el script) para visualizar las contribuciones en un gráfico de cascada
waterfallplot(X_test.head(1), x0.Contribucion, formatting='{:,.3f}', size=(20,5), sorted_value=True, threshold=0.05);

# Anexo!
import numpy as np
import pandas as pd
import graphviz, IPython
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import FuncFormatter
from sklearn.tree import export_graphviz

"""Función auxiliar para visualizar la contribución de cada variable sobre el objetivo!
def waterfallplot(sample, data, Title="", x_lab="", y_lab="",
		 formatting="{:,.1f}", green_color='#29EA38', red_color='#FB3C62', blue_color='#24CAFF',
		 sorted_value=False, threshold=None, other_label='other', net_label='net',
		 rotation_value=0, size=None):
"""
