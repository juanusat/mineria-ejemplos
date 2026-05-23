# Caso práctico 01_Preparacion de datos_Modelado_Prueba.ipynb

# Conexion a Google Colaborative
# ------------
# Importamos el módulo para manejar la conexión e integración con Google Drive
from google.colab import drive
# Montamos la partición de drive para lectura y escritura de archivos locales en la nube
drive.mount('/gdrive')

#Importar las librerías necesarias en Python.
# ------------
# Pandas: Principal herramienta de manipulación y análisis de datos tabulares (DataFrames)
import pandas as pd
# Warnings: Librería base para controlar y ocultar mensajes de advertencia que emiten algunas funciones
import warnings
# Ocultamos temporalmente los mensajes de alerta (warnings) para mantener una consola o salida en pantalla más limpia
warnings.filterwarnings("ignore")
# Numpy: Librería clave de cálculo científico, optimizada para soporte de vectores, matrices y funciones matemáticas
import numpy as np
# Scipy.stats: Complemento científico que contiene una gran cantidad de distribuciones de probabilidad y funciones estadísticas
from scipy import stats

# Cargamos el dataset desde la ruta especificada en Google Drive a un DataFrame de pandas
desarrll = pd.read_csv("/gdrive/MyDrive/Data/AdquisicionCreditoHipotecario.csv") # Ruta donde esta su set de datos!

# Verificamos las dimensiones (filas, columnas) del DataFrame cargado
desarrll.shape

# Mostramos las primeras 5 filas del DataFrame para una inspección inicial de los datos
desarrll.head()

# Obtenemos la lista de nombres de las columnas
desarrll.columns

# Renombramos las variables por buenas prácticas
# Se define una lista con nombres de columna más descriptivos y en inglés para estandarizar
Columnsnames = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
# Asignamos la nueva lista de nombres a las columnas del DataFrame
desarrll.columns = Columnsnames

# Generamos estadísticas descriptivas para todas las variables, incluyendo las categóricas
desarrll.describe(include='all') # Describir todas las variables.

# Revisamos los valores nulos o missings!
# Contamos la cantidad de valores nulos (NaN) en cada columna del DataFrame
desarrll.isnull().sum()

#separar las variables
# Creamos listas para separar los nombres de las columnas categóricas y numéricas
columnas_categoricas = ["Gender","Married","Education","Self_Employed","Property_Area","Dependents",'Credit_History','Loan_Status']
columnas_numericas   = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]

"""**Preparar los datos**"""
# ------------

# Usamos metodos de imputacion
# Importamos la clase SimpleImputer de scikit-learn para manejar valores faltantes
from sklearn.impute import SimpleImputer
# Generamos el imputador iterativo - Imputacion Univariada Numerica
# Creamos un imputador para variables numéricas que reemplazará los NaN con la mediana
imp_univ_num = SimpleImputer(missing_values=np.nan, strategy='median')

# Generamos el imputador iterativo - Imputacion Univariada Categorica
# Creamos un imputador para variables categóricas que reemplazará los NaN con el valor más frecuente (moda)
imp_univ_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Generamos los subset de variables categoricas - continuas!
# Creamos sub-DataFrames con solo las columnas numéricas y categóricas respectivamente
data_impt_num = desarrll[columnas_numericas]
data_impt_cat = desarrll[columnas_categoricas]

# Realizamos la imputación univariada en una nueva base de datos - Variables Numericas
imp_univ_num.fit(data_impt_num)     # 1° Ajusta o entrena el imputador con los datos numéricos
imputed_data_univ_num = pd.DataFrame(data=imp_univ_num.transform(data_impt_num),  # 2° Aplica la transformación y crea un nuevo DataFrame
                             columns=data_impt_num.columns,dtype='float')

# Realizamos la imputación univariada en una nueva base de datos - Variables Categoricas
imp_univ_cat.fit(data_impt_cat) # 1° Ajusta o entrena el imputador con los datos categóricos
imputed_data_univ_cat = pd.DataFrame(data=imp_univ_cat.transform(data_impt_cat), # 2° Aplica la transformación y crea un nuevo DataFrame
                             columns=data_impt_cat.columns,dtype='object')

# Consolidamos los subset!
# Unimos los dos DataFrames (numérico y categórico ya imputados) en uno solo
desarrll_imp = pd.concat([imputed_data_univ_num,imputed_data_univ_cat],axis=1)

# Mostramos las primeras filas del DataFrame resultante para verificar la unión
desarrll_imp.head()

# Comprobamos la completitud de los datos! y lo conseguimos!
# Verificamos que ya no queden valores nulos en ninguna columna
desarrll_imp.isnull().sum()

"""#### Recodificacion de los datos"""
# ------------

# LabelEncoder de los datos!
# Importamos LabelEncoder para convertir las etiquetas categóricas en números
from sklearn.preprocessing import LabelEncoder
# Preprocesamiento con LabelEncoderfrom
# Iteramos sobre cada columna categórica para aplicar la codificación
for c in columnas_categoricas:
    print(str(c))
    le = LabelEncoder() # Creamos una instancia del codificador
    le.fit(desarrll_imp[str(c)]) # Ajustamos el codificador a los valores únicos de la columna
    desarrll_imp[str(c)]=le.transform(desarrll_imp[str(c)]) # Transformamos la columna a sus equivalentes numéricos

# Mostramos las primeras 20 filas para ver el resultado de la codificación
desarrll_imp.head(20)

"""#### Tratamiento de Outliers"""
# ------------

# Datos atípicos
# Tratamiento de Outliers Univariados!

# Creamos una funcion para poder visualizar los percentiles  ---- Estadisticos de orden
def Cuantiles(lista):
    # Definimos una lista de percentiles de interés
    c = [0,1,5,10,20,30,40,50,60,70,80,90,92.5,95,97.5,99,100]
    # Calculamos los valores correspondientes a esos percentiles
    matrix = pd.concat([pd.DataFrame(c),pd.DataFrame(np.percentile(lista.dropna(),c))],axis = 1)
    matrix.columns = ["Cuantil","Valor_Variable"]
    return(matrix)

# Analizamos las variables numericas
# Variable
# Aplicamos la función a la columna 'ApplicantIncome' para analizar su distribución
Cuantiles(desarrll_imp["ApplicantIncome"]).transpose()
# Nos hacemos la pregunta, podríamos acotar la variable?

# Tema : Cuantiles
# Divides tus datos en 10 : Deciles
# Divides tus datos en 100 : Percentiles
# Divides tus datos en 4 : Cuartil
# La mediana es : El decil 5, el percentil 50 y el cuartil 2

## ApplicantIncome
# Calculamos el percentil 1 y 97.5 para identificar y acotar valores extremos (outliers)
cuantil_1 = np.percentile(desarrll_imp["ApplicantIncome"],1)
cuantil_97 = np.percentile(desarrll_imp["ApplicantIncome"],97.5)

# Mostramos el valor del percentil 1
cuantil_1

# Reemplazamos el valor minimo y maximo
# Topear las colas!
# Reemplazamos los valores por debajo del percentil 1 con el valor del percentil 1
desarrll_imp.loc[desarrll_imp["ApplicantIncome"]<cuantil_1,"ApplicantIncome"] = cuantil_1
# Reemplazamos los valores por encima del percentil 97.5 con el valor del percentil 97.5
desarrll_imp.loc[desarrll_imp["ApplicantIncome"]>cuantil_97,"ApplicantIncome"] = cuantil_97

"""**Modelamiento de los datos**"""
# ------------

# Creación de la data de train y la data de test ----- Entrenamiento y prueba
# train_test_split (X,y,%y,Estratificar?,Semilla aleatoria)
# Importamos la función para dividir el dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Realizamos la partición del dataset
X_train, X_test, y_train, y_test = train_test_split(desarrll_imp.drop('Loan_Status',axis=1), # X: todas las variables excepto el target
                                                    desarrll_imp.Loan_Status, # y: la variable target
                                                    test_size=0.33, # Proporción del dataset para el conjunto de prueba
                                                    stratify=desarrll_imp.Loan_Status, # Estratificar para mantener la proporción de clases en train y test
                                                    random_state=100) # Semilla para reproducibilidad

"""Modelos Supervisados: Clasificación con Arbol CART"""
# ------------

# Verificamos las dimensiones del conjunto de entrenamiento
X_train.shape

# Verificamos el tipo de dato del target de prueba
type(y_test)

# Llamar un algoritmo predictivos
# Importamos el clasificador de Árbol de Decisión
from sklearn.tree import DecisionTreeClassifier

#max_depth --> profundidad 3,4,5
#max_features --> maximo de variables a probar
#min_samples ---> mínimo de observaciones en nodo

#cart = DecisionTreeClassifier()


# Creamos una instancia del clasificador con hiperparámetros específicos para controlar su complejidad
cart = DecisionTreeClassifier(criterion='gini', # Métrica para medir la calidad de una división
                             max_depth= 4, # Profundidad máxima del árbol
                              max_features = 11, # Número máximo de características a considerar para una división
                              min_samples_leaf = 25) # Número mínimo de muestras requeridas en un nodo hoja


cart.fit(X_train, y_train) # Entrenamos el algoritmo con los datos de entrenamiento

# Predecir con el algoritmo entrenado para validar
y_pred_train=cart.predict(X_train) # Prediccion sobre el train
y_pred_test= cart.predict(X_test) # Prediccion sobre el test

# Comparar el valor pronosticado con el valor real

# Importamos el módulo de métricas de scikit-learn
from sklearn import metrics as metrics
# Matriz de confusion
# Calculamos y mostramos la matriz de confusión para el conjunto de entrenamiento
print("Matriz confusion: Train")
cm_train = metrics.confusion_matrix(y_train,y_pred_train)
print(cm_train)

# Calculamos y mostramos la matriz de confusión para el conjunto de prueba
print("Matriz confusion: Test")
cm_test = metrics.confusion_matrix(y_test,y_pred_test)
print(cm_test)

# Accuracy
# Calculamos y mostramos la exactitud (accuracy) para el conjunto de entrenamiento
print("Accuracy: Train")
accuracy_train=metrics.accuracy_score(y_train,y_pred_train)
print(accuracy_train)

# Calculamos y mostramos la exactitud (accuracy) para el conjunto de prueba
print("Accuracy: Test")
accuracy_test=metrics.accuracy_score(y_test,y_pred_test)
print(accuracy_test)

# Precision
# Calculamos y mostramos la precisión para el conjunto de entrenamiento
print("Precision: Train")
precision_train=metrics.precision_score(y_train,y_pred_train)
print(precision_train)

# Calculamos y mostramos la precisión para el conjunto de prueba
print("Precision: Test")
precision_test=metrics.precision_score(y_test,y_pred_test)
print(precision_test)

# Recall
# Calculamos y mostramos el recall (sensibilidad) para el conjunto de entrenamiento
print("Recall: Train")
recall_train=metrics.recall_score(y_train,y_pred_train)
print(recall_train)

# Calculamos y mostramos el recall (sensibilidad) para el conjunto de prueba
print("Recall: Test")
recall_test=metrics.recall_score(y_test,y_pred_test)
print(recall_test)

# Esta función no está definida en el script, por lo que no se puede ejecutar.
# Su propósito sería visualizar la estructura del árbol de decisión entrenado.
draw_tree(cart,X_train) # Función propia no documentada para visualizar el árbol de decisión



# ANEXO -- DIBUJAR ARBOL
# Definición de una función de ejemplo que simula la visualización del árbol.
def draw_tree(tree, df):
    return True # solo de ejemplo

"""Función auxiliar para visualizar la contribución de cada variable sobre el objetivo!
def waterfallplot(sample, data, Title="", x_lab="", y_lab="",
		 formatting="{:,.1f}", green_color='#29EA38', red_color='#FB3C62', blue_color='#24CAFF',
		 sorted_value=False, threshold=None, other_label='other', net_label='net',
		 rotation_value=0, size=None):
"""
