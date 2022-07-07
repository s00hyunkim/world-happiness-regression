# STEP 0: IMPORT ALL NECESSARY LIBRARIES
## libraried installed so far: numpy, pandas, sklearn, matplotlib
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np

# STEP 1: IMPORT DATASET
# 1-1. import csv file
data = pd.read_csv('https://raw.githubusercontent.com/s00hyunkim/world-happiness-regression/main/world-happiness-score-2020.csv')
# 1-2. check if the dataset has been imported successfully
# print('[SNIPPET OF THE DATASET]')
# print(data.head())
# print('')

# STEP 2: CLEAN & MANIPULATE DATA
# 2-1. remove columns that we don't need
data.drop(columns = ['Country name', 'Regional indicator', 'Standard error of ladder score',
'upperwhisker', 'lowerwhisker', 'Ladder score in Dystopia', 'Explained by: Log GDP per capita',
'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices',
'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Dystopia + residual'],
axis = 1, inplace = True)
# 2-2. rename columns
data.columns = ['Ladder', 'LogGDP', 'SocialSupport', 'HealthyLifeExpectancy', 'Freedom', 'Generosity', 'Corruption']
# 2-3. check if there are any null values
print('[# OF NULL VALUES OF EACH FEATURE]')
print(data.isnull().sum())
print('')
# 2-4. determine what % of each feature is null to figure out which null values to drop
print('[% OF NULL VALUES OF EACH FEATURE]')
for col in data.columns:
    print(col,  ': %.2f%%' %((data[col].isnull().sum() / data.shape[0]) * 100))
print('')
# 2-5. check how many non-null values each feature has
print('[# OF NON-NULL VALUES OF EACH FEATURE]')
print(data.info())
print('')
# 2-6. check correlation between each feature and target
print('[CORRELATION BETWEEN FEATURES AND SCORE]')
print(data.corr()['Ladder'])
print('')

# STEP 3: PLOT CORRELATION BETWEEN FEATURES AND HAPPINESS SCORE
data.plot.scatter(x= 'LogGDP', y='Ladder')
data.plot.scatter(x= 'SocialSupport', y='Ladder')
data.plot.scatter(x= 'HealthyLifeExpectancy', y='Ladder')
data.plot.scatter(x= 'Freedom', y='Ladder')
data.plot.scatter(x= 'Generosity', y='Ladder')
data.plot.scatter(x= 'Corruption', y='Ladder')

# STEP 4: SPLIT DATA
# 4-1. set X to be all features and y to be target
X = data.iloc[:,1:]
y = data.iloc[:, 0]
# 4-2. split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# STEP 5: BUILD REGRESSION MODELS
print('[WITH NEITHER NORMALIZATION NOR STANDARDIZATION]')

print('[STANDARD LINEAR REGRESSION]')
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('Standard linear regression R2 = ', r2, '\n')
coef = reg.coef_
print('Standard linear regression coefficients :')
i = 0; 
for col in X.columns:
  print(col, ":", coef[i])
  i = i + 1
print('')

print('[RIDGE REGRESSION]')
ridge_reg = Ridge().fit(X_train, y_train)
y_pred = ridge_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Ridge linear regression R2 = ", r2, "\n")
coef = reg.coef_
print("Ridge linear regression coefficients :")
i = 0; 
for col in X.columns:
  print(col, ":", coef[i])
  i = i + 1
print('')