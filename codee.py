import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
df_train = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv', index_col=0)
df_train['schoolsup'].dtype
# output : dtype('bool')
df_train['school'] = df_train['school'].map({'GP':1,'MS':2})
df_train['class'] = df_train['class'].map({'por':1,'mat':2})
df_train['sex'] = df_train['sex'].map({'F':1,'M':2})
df_train['address'] = df_train['address'].map({'U':1,'R':2})
df_train['famsize'] = df_train['famsize'].map({'GT3':1,'LE3':2})
df_train['Pstatus'] = df_train['Pstatus'].map({'T':1,'A':2})
df_train['Mjob'] = df_train['Mjob'].map({'other':1,'services':2,'at_home':3,'teacher':4,'health':5})
df_train['Fjob'] = df_train['Fjob'].map({'other':1,'services':2,'teacher':3,'at_home':4,'health':5})
df_train['reason'] = df_train['reason'].map({'course':1,'reputation':2,'home':3,'other':4})
df_train['guardian'] = df_train['guardian'].map({'mother':1,'father':2,'other':3})
'''df_train['schoolsup'] = df_train['schoolsup'].replace('TRUE',1).replace('FALSE',2)
df_train['famsup'] = df_train['famsup'].replace('TRUE',1).replace('FALSE',2)
df_train['paid'] = df_train['paid'].replace('TRUE',1).replace('FALSE',2)
df_train['activities'] = df_train['activities'].replace('TRUE',1).replace('FALSE',2)
df_train['nursery'] = df_train['nursery'].replace('TRUE',1).replace('FALSE',2)
df_train['higher'] = df_train['higher'].replace('TRUE',1).replace('FALSE',2)
df_train['internet'] = df_train['internet'].replace('TRUE',1).replace('FALSE',2)
df_train['romantic'] = df_train['romantic'].replace('TRUE',1).replace('FALSE',2)'''

#df_train.head()
#df_test.info()
df_test['school'] = df_test['school'].map({'GP':1,'MS':2})
df_test['class'] = df_test['class'].map({'por':1,'mat':2})
df_test['sex'] = df_test['sex'].map({'F':1,'M':2})
df_test['address'] = df_test['address'].map({'U':1,'R':2})
df_test['famsize'] = df_test['famsize'].map({'GT3':1,'LE3':2})
df_test['Pstatus'] = df_test['Pstatus'].map({'T':1,'A':2})
df_test['Mjob'] = df_test['Mjob'].map({'other':1,'services':2,'at_home':3,'teacher':4,'health':5})
df_test['Fjob'] = df_test['Fjob'].map({'other':1,'services':2,'teacher':3,'at_home':4,'health':5})
df_test['reason'] = df_test['reason'].map({'course':1,'reputation':2,'home':3,'other':4})
df_test['guardian'] = df_test['guardian'].map({'mother':1,'father':2,'other':3})
'''df_test['schoolsup'] = df_test['schoolsup'].replace('TRUE',1).replace('FALSE',2)
df_test['famsup'] = df_test['famsup'].replace('TRUE',1).replace('FALSE',2)
df_test['paid'] = df_test['paid'].replace('TRUE',1).replace('FALSE',2)
df_test['activities'] = df_test['activities'].replace('TRUE',1).replace('FALSE',2)
df_test['nursery'] = df_test['nursery'].replace('TRUE',1).replace('FALSE',2)
df_test['higher'] = df_test['higher'].replace('TRUE',1).replace('FALSE',2)
df_test['internet'] = df_test['internet'].replace('TRUE',1).replace('FALSE',2)
df_test['romantic'] = df_test['romantic'].replace('TRUE',1).replace('FALSE',2)'''
#df_train.corr()['G3'].sort_values()
#df_train.isnull().sum()
#df_test.isnull().sum()
X_train = df_train.drop(['G3'], axis=1).values
y_train = df_train['G3'].values
X_test = df_test.values
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
reg = DecisionTreeRegressor(max_depth=30, criterion='mse')
reg.fit(X_train, y_train)
'''params = {'criterion':['mse', 'mae'], 'max_depth':[ 2, 3, 4, 5, 6, 7]}
gscv = GridSearchCV(reg, params, cv=5, scoring='mean_squared_error')
gscv.fit(X_train, y_train)'''

reg.score(X_train, y_train)
# output: 1.0
p = reg.predict(X_test)
p
# Output:
# array([17., 14., 10., 11.,  8., 16., 11.,  9., 13., 14., 12.,  8.,  9.,
#         8.,  0., 14., 10., 14., 17., 10., 11., 11.,  8., 14., 13.,  0.,
#        13.,  0., 13., 11., 15., 10., 10., 11., 11., 15., 10., 14., 11.,
#        11., 18., 10.,  9., 15., 17., 12., 18., 15., 12., 10., 15., 13.,
#        12.,  8., 14., 14.,  0., 18., 14.,  9., 14.,  9., 15., 14., 12.,
#        10.,  9.,  9., 13., 12., 17., 10., 11., 13., 10., 12., 11.,  6.,
#         9., 14., 15., 17.,  9., 13., 12., 16., 15., 13., 14.,  7.,  0.,
#        11., 15., 13., 14., 10., 11., 19., 10., 17.,  8., 12.,  0., 13.,
#        11., 12., 13., 12.,  9., 12., 13., 10., 12., 15.,  0.,  7., 15.,
#        11.,  0., 12.,  0.,  7.,  0., 13., 11., 12., 13., 11., 17., 11.,
#         6., 18., 11., 13., 13., 15., 11., 10.,  7., 19., 13., 15., 16.,
#        13., 12., 11., 10., 14., 13., 16., 11.,  7., 11.,  8., 11.,  8.,
#         9., 13., 13., 10.,  0., 10., 18.,  8., 15.,  8., 10., 18., 11.,
#        18., 17., 10., 11., 16., 18., 16., 11., 14., 12., 14., 16., 15.,
#        16., 13., 12., 11., 10., 13., 10.,  0., 15., 12., 13., 11., 15.,
#        11., 10., 12.,  9., 13., 15., 10., 10., 11., 19., 13., 18., 14.,
#        10., 12., 10., 14., 11., 18.,  0., 11., 12.,  9., 16., 10., 14.,
#         8., 10.,  8., 11., 15., 14.,  0., 15., 14., 12., 14.,  9., 15.,
#        10., 11., 13., 12., 16., 13., 13.,  5., 10.,  9.,  8.,  9., 13.,
#         0., 13.,  9., 15.,  9., 14., 18., 15.,  7.,  0., 11., 14., 16.,
#        17., 15., 15., 12.,  8., 10.,  8., 10., 10., 17., 11.,  7., 10.,
#        11., 10.,  7.,  9., 15.,  7.,  0.,  5.,  9.,  8., 12.,  0., 12.,
#        16.,  8., 18.,  8., 18., 15., 14., 13., 14., 18., 12., 10., 11.,
#        10., 14., 15., 15., 16., 14.,  8.,  8., 15., 16.,  9., 16., 13.,
        8.])
df_submit = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv', index_col=0)
df_submit['G3'] = p
df_submit.to_csv('submission1.csv')
