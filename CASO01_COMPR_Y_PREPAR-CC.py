# Caso práctico 01_Comprensión y preparación de datos.ipynb


# Conexion a Google Colaborative a drive!
# ------------
# Importamos la librería para interactuar con Google Colab y acceder a nuestros archivos en la nube
from google.colab import drive
# Montamos nuestra unidad de Google Drive para poder acceder a los datasets y guardar resultados
drive.mount('/gdrive')

## importación de librerías
# ------------
# Pandas: Proporciona estructuras de datos de alto rendimiento (DataFrames) ideal para manipulación y análisis de datos
import pandas as pd
# NumPy: Útil para realizar cálculos numéricos avanzados y crear/manipular matrices o vectores
import numpy as np
# Seaborn: Proporciona una interfaz de alto nivel para dibujar gráficos estadísticos atractivos y complejos
import seaborn as sns
# Matplotlib: La librería principal en Python para la creación de visualizaciones estáticas, animadas e interactivas
import matplotlib.pyplot as plt

"""
### **Exploración de datos**"""
# ------------

# Cargamos nuestro archivo CSV a un DataFrame de pandas para la exploración inicial.
df = pd.read_csv('/gdrive/MyDrive/Data/AdquisicionCreditoHipotecario.csv')

# Se inspecciona el tamaño total de nuestro dataset (.shape devuelve una tupla de filas y columnas)
df.shape

# Mostramos los primeros 5 registros de nuestro dataset (útil para validar rápidamente que los datos se importaron bien)
df.head()

"""#### Buenas prácticas"""
# ------------

# Obtenemos una lista con los nombres actuales de las columnas del DataFrame
df.columns

# Renombramos las variables por buenas prácticas
# Se define una lista con nombres de columna más descriptivos y en inglés para estandarizar
Columnsnames = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
# Asignamos la nueva lista de nombres a las columnas del DataFrame
df.columns = Columnsnames

# Verificamos que los nombres de las columnas se hayan actualizado correctamente
df.columns

"""#### Identificando variables importantes/ delimitantes"""
# ------------

df.dtypes

# Cambiamos el tipo de dato!
df.Credit_History = df.Credit_History.astype('object')

df.ApplicantIncome = df.ApplicantIncome.astype('float64')

# Los préstamos de menos de menos de 10 años tienen sentido?
# Revisemos especificaciones de negocio.
df.groupby('Loan_Amount_Term').size()

# Elegir los solo las caracteristicas deseadas!
df = df[df['Loan_Amount_Term']>=120]

# Revisamos el tamaño nuevo del set de datos
df.shape

# Revisamos la nueva distribucion de datos
df.groupby('Loan_Amount_Term').size()

# Guardamos el data frame generado
df.to_csv('/gdrive/MyDrive/Data/AdquisicionCreditoHipotecario_Prueba.csv') # Ruta a guardar el archivo generado!

"""#### Diferenciamos el tipo de variales"""

columnsNumeric = ['ApplicantIncome','CoapplicantIncome','LoanAmount',
                  'Loan_Amount_Term']
columnsString = ['Gender', 'Married', 'Dependents','Education','Self_Employed',
                 'Property_Area','Credit_History','Loan_Status']

"""### **Análisis univariado**

#### Variables numericas

Primeros descubrimientos
"""

df[columnsNumeric].describe()
# Algunas conclusiones:
# Existen valores perdidos. --> ¡Imputacion!
# Existen valores atípicos. Media y mediana muy distintas. --> ¿Transformaciones, Recodificaciones, Topes de variables?

[columnsNumeric]

# Notamos que algunas de las variables tienen valores nulos o missings
df[columnsNumeric].isnull().sum()

"""Apoyo del análisis visual"""

# El grafico de cajas es muy importante pues nos muestra , dispersion, forma y atipicos:
for x in columnsNumeric:
  plt.title(df[x].name)
  sns.boxplot(x=df[x], palette="Blues");
  plt.show()

# El grafico de cajas es muy importante pues nos muestra , dispersion, forma y atipicos; siempre respecto al target:
for x in columnsNumeric:
  plt.title(df[x].name)
  sns.boxplot(x=df[x], y = df.Loan_Status);
  plt.show()

"""### **Preparación de los datos**

Presencia de valores missing
"""

# Revision de valores misssings o nulos - Valores absolutos
df[columnsNumeric].isnull().sum()

df.LoanAmount.notnull().value_counts()

# Probamos distintas maneras de rellenar o imputar los valores perdidos
df['LoanAmount_2'] = df['LoanAmount'].fillna(0)  # Experiencia o criterio experto!
df['LoanAmount_3'] = df['LoanAmount'].fillna(df.LoanAmount.mean())   # Imputar por media!
df['LoanAmount_4'] = df['LoanAmount'].fillna(df.LoanAmount.median()) # Imputar por mediana!
df['LoanAmount_5'] = df['LoanAmount'].fillna(method = 'backfill')    # Imputar por vecindad!

# Mostramos las estadísticas principales de las nuevas variables
df[['LoanAmount','LoanAmount_2','LoanAmount_3','LoanAmount_4','LoanAmount_5']].describe()

# Revisamos como la imputacion de alguna manera cambia la distribucion de la variable
for x in ['LoanAmount','LoanAmount_2','LoanAmount_3','LoanAmount_4','LoanAmount_5']:
  Q03 = int(df[x].quantile(0.75))+100
  plt.title(df[x].name)
  plt.hist(df[x], bins= 100 ,range=(0,Q03))
  plt.show()

columnsNumeric

data.columns

#Retiramos variables numéricas
columnsNumeric.remove('LoanAmount')

#Adherimos las nuevas variables numéricas
columnsNumeric = columnsNumeric + ['LoanAmount_5']
# Podemos quedarnos con la 4 o 5!

columnsNumeric

"""#### Variables categóricas

Primeros descubrimientos
"""

# Generamos estadísticas descriptivas solo para las columnas categóricas (frecuencia, valor más común, etc.)
df[columnsString].describe(include='O')

columnsString = ['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area','Credit_History','Loan_Status']

df.head(20)

df[columnsString].dtypes

# Validar registros unicos
df[columnsString].nunique()

# Validando registros únicos
print(df.shape)
print(df[columnsString].nunique())

# Mostramos la frecuencia de variables cateóricas para encontras hallazgos
for x in columnsString:
    print(x)
    print(df.groupby(x).size())
    print("\n")

columnsString

"""Apoyo del análisis visual"""

#Columnas categoricas
for x in columnsString:
  plt.title(x)
  df.fillna("--NULL").groupby(x)[x].count().plot(kind = "bar")
  plt.show()

"""Presencia de valores missing"""

#Presencia de valores missing
df[columnsString].isnull().sum()

columnsString

df[columnsString].dtypes

# Missings o valores perdidos absolutos
df[columnsString].isnull().sum() #obseva

# Missings o valores perdidos relativos
df.isnull().sum()

#Como el porcentaje de nan no es muy grande, hacemos imputacion por moda o criterio experto y seguimos trabajando.
df['Married'] = df['Married'].fillna('Yes')
df['Dependents'] = df['Dependents'].fillna('0')
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['Credit_History'] = df['Credit_History'].fillna('1.0')
df['Gender'] = df['Gender'].fillna('Male')


# Completar las demas variables

"""Recodificamos variables importantes"""

df.groupby('Dependents').size()/df.shape[0]*100

# Siempre hacemos una nueva columna para probar y despues si todo esta bien la eliminamos
df['Dependents2'] = df['Dependents']

df.head(3)

df['Dependents2'] = df['Dependents2'].replace('3+','3')

df.groupby('Dependents2').size()/df.shape[0]*100