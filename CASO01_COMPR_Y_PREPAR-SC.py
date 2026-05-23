from google.colab import drive
drive.mount('/gdrive')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/gdrive/MyDrive/Data/AdquisicionCreditoHipotecario.csv')
df.shape
df.head()
df.columns

Columnsnames = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

df.columns = Columnsnames
df.columns
df.dtypes
df.Credit_History = df.Credit_History.astype('object')
df.ApplicantIncome = df.ApplicantIncome.astype('float64')

df.groupby('Loan_Amount_Term').size()
df = df[df['Loan_Amount_Term']>=120]
df.shape
df.groupby('Loan_Amount_Term').size()
df.to_csv('/gdrive/MyDrive/Data/AdquisicionCreditoHipotecario_Prueba.csv')

columnsNumeric = ['ApplicantIncome','CoapplicantIncome','LoanAmount',
                  'Loan_Amount_Term']
columnsString = ['Gender', 'Married', 'Dependents','Education','Self_Employed',
                 'Property_Area','Credit_History','Loan_Status']

df[columnsNumeric].describe()
[columnsNumeric]
df[columnsNumeric].isnull().sum()

for x in columnsNumeric:
  plt.title(df[x].name)
  sns.boxplot(x=df[x], palette="Blues");
  plt.show()

for x in columnsNumeric:
  plt.title(df[x].name)
  sns.boxplot(x=df[x], y = df.Loan_Status);
  plt.show()

df[columnsNumeric].isnull().sum()

df.LoanAmount.notnull().value_counts()

df['LoanAmount_2'] = df['LoanAmount'].fillna(0)
df['LoanAmount_3'] = df['LoanAmount'].fillna(df.LoanAmount.mean())
df['LoanAmount_4'] = df['LoanAmount'].fillna(df.LoanAmount.median())
df['LoanAmount_5'] = df['LoanAmount'].fillna(method = 'backfill')

df[['LoanAmount','LoanAmount_2','LoanAmount_3','LoanAmount_4','LoanAmount_5']].describe()

for x in ['LoanAmount','LoanAmount_2','LoanAmount_3','LoanAmount_4','LoanAmount_5']:
  Q03 = int(df[x].quantile(0.75))+100
  plt.title(df[x].name)
  plt.hist(df[x], bins= 100 ,range=(0,Q03))
  plt.show()

columnsNumeric

data.columns

columnsNumeric.remove('LoanAmount')

columnsNumeric = columnsNumeric + ['LoanAmount_5']

columnsNumeric

df[columnsString].describe(include='O')

columnsString = ['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area','Credit_History','Loan_Status']

df.head(20)

df[columnsString].dtypes

df[columnsString].nunique()

print(df.shape)
print(df[columnsString].nunique())

for x in columnsString:
    print(x)
    print(df.groupby(x).size())
    print("\n")

columnsString

for x in columnsString:
  plt.title(x)
  df.fillna("--NULL").groupby(x)[x].count().plot(kind = "bar")
  plt.show()

df[columnsString].isnull().sum()

columnsString

df[columnsString].dtypes

df[columnsString].isnull().sum()

df.isnull().sum()

df['Married'] = df['Married'].fillna('Yes')
df['Dependents'] = df['Dependents'].fillna('0')
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['Credit_History'] = df['Credit_History'].fillna('1.0')
df['Gender'] = df['Gender'].fillna('Male')

df.groupby('Dependents').size()/df.shape[0]*100

df['Dependents2'] = df['Dependents']

df.head(3)

df['Dependents2'] = df['Dependents2'].replace('3+','3')

df.groupby('Dependents2').size()/df.shape[0]*100