from google.colab import drive
drive.mount('/gdrive')
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
desarrll = pd.read_csv("/gdrive/MyDrive/Data/AdquisicionCreditoHipotecario.csv")
desarrll.shape
desarrll.head()
desarrll.columns
Columnsnames = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
desarrll.columns = Columnsnames
desarrll.describe(include='all')
desarrll.isnull().sum()
columnas_categoricas = ["Gender","Married","Education","Self_Employed","Property_Area","Dependents",'Credit_History','Loan_Status']
columnas_numericas   = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]

from sklearn.impute import SimpleImputer
imp_univ_num = SimpleImputer(missing_values=np.nan, strategy='median')
imp_univ_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_impt_num = desarrll[columnas_numericas]
data_impt_cat = desarrll[columnas_categoricas]
imp_univ_num.fit(data_impt_num)
imputed_data_univ_num = pd.DataFrame(data=imp_univ_num.transform(data_impt_num),
                             columns=data_impt_num.columns,dtype='float')
imp_univ_cat.fit(data_impt_cat)
imputed_data_univ_cat = pd.DataFrame(data=imp_univ_cat.transform(data_impt_cat),
                             columns=data_impt_cat.columns,dtype='object')
desarrll_imp = pd.concat([imputed_data_univ_num,imputed_data_univ_cat],axis=1)
desarrll_imp.head()
desarrll_imp.isnull().sum()

from sklearn.preprocessing import LabelEncoder
for c in columnas_categoricas:
    print(str(c))
    le = LabelEncoder()
    le.fit(desarrll_imp[str(c)])
    desarrll_imp[str(c)]=le.transform(desarrll_imp[str(c)])

desarrll_imp.head(20)
def Cuantiles(lista):
    c = [0,1,5,10,20,30,40,50,60,70,80,90,92.5,95,97.5,99,100]
    matrix = pd.concat([pd.DataFrame(c),pd.DataFrame(np.percentile(lista.dropna(),c))],axis = 1)
    matrix.columns = ["Cuantil","Valor_Variable"]
    return(matrix)
Cuantiles(desarrll_imp["ApplicantIncome"]).transpose()

cuantil_1 = np.percentile(desarrll_imp["ApplicantIncome"],1)
cuantil_97 = np.percentile(desarrll_imp["ApplicantIncome"],97.5)
cuantil_1
desarrll_imp.loc[desarrll_imp["ApplicantIncome"]<cuantil_1,"ApplicantIncome"] = cuantil_1
desarrll_imp.loc[desarrll_imp["ApplicantIncome"]>cuantil_97,"ApplicantIncome"] = cuantil_97

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(desarrll_imp.drop('Loan_Status',axis=1),
                                                    desarrll_imp.Loan_Status,
                                                    test_size=0.33,
                                                    stratify=desarrll_imp.Loan_Status,
                                                    random_state=100)
X_train.shape
type(y_test)
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier(criterion='gini',
                             max_depth= 4,
                              max_features = 11,
                              min_samples_leaf = 25)
cart.fit(X_train, y_train)
y_pred_train=cart.predict(X_train)
y_pred_test= cart.predict(X_test)
from sklearn import metrics as metrics
print("Matriz confusion: Train")
cm_train = metrics.confusion_matrix(y_train,y_pred_train)
print(cm_train)
print("Matriz confusion: Test")
cm_test = metrics.confusion_matrix(y_test,y_pred_test)
print(cm_test)
print("Accuracy: Train")
accuracy_train=metrics.accuracy_score(y_train,y_pred_train)
print(accuracy_train)
print("Accuracy: Test")
accuracy_test=metrics.accuracy_score(y_test,y_pred_test)
print(accuracy_test)
print("Precision: Train")
precision_train=metrics.precision_score(y_train,y_pred_train)
print(precision_train)
print("Precision: Test")
precision_test=metrics.precision_score(y_test,y_pred_test)
print(precision_test)
print("Recall: Train")
recall_train=metrics.recall_score(y_train,y_pred_train)
print(recall_train)
print("Recall: Test")
recall_test=metrics.recall_score(y_test,y_pred_test)
print(recall_test)
draw_tree(cart,X_train)