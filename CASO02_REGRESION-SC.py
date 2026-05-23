from google.colab import drive
drive.mount('/gdrive')
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
import warnings
warnings.filterwarnings('ignore')
import gdown
vivienda = pd.read_csv('/gdrive/MyDrive/Data/bostonvivienda.csv')
vivienda.head()
vivienda.shape
vivienda.columns.to_list()
vivienda.dtypes
p = sns.pairplot(vivienda, x_vars = ['crim','zn','indus','nox','rm','edad','dis','rad','impuesto','ptratio','negro','lstat',],
                 y_vars='medv', size=3, aspect=0.7)

vivienda.corr(method = 'spearman')

corr = vivienda.corr()
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
X = vivienda.drop('medv',axis=1)
type(X)
y = vivienda.medv
type(y)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=100)
reg_multiple = LinearRegression()
reg_multiple.fit(X_train,y_train)
print(pd.Series(reg_multiple.coef_, index = X_train.columns).sort_values(ascending = False))
reg_multiple.intercept_

y_pred_train = reg_multiple.predict(X_train)
y_pred_test = reg_multiple.predict(X_test)
print("ECM: {}".format(mean_squared_error(y_train,y_pred_train)))
print("ECM: {}".format(mean_squared_error(y_test,y_pred_test)))

print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train,y_pred_train))))
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test,y_pred_test))))

print("R^2: {}".format(r2_score(y_train,y_pred_train)))
print("R^2: {}".format(r2_score(y_test,y_pred_test)))

imp = pd.DataFrame({'Nombre_Variable':X_train.columns,
                    'Importancia':(np.absolute(reg_multiple.coef_))}).sort_values(
    'Importancia', ascending=False)
imp.style.bar()

x0 = pd.DataFrame(X_test.iloc[0].rename('Valor_Variable'))
x0['Nombre_Variable'] = x0.index
x0 = pd.merge(x0, imp, on='Nombre_Variable')
x0['Contribucion'] = x0.Valor_Variable * x0.Importancia
x0.sort_values('Contribucion', ascending=False).style.bar(['Contribucion'])