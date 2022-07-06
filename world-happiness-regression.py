# STEP 0: IMPORT ALL NECESSARY LIBRARIES
## libraried installed so far: numpy, pandas, sklearn, matplotlib
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
# import itertools
# import copy

# import random
# random.seed(0)

# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# import seaborn as sns

# from statistics import mean

# from sklearn import datasets
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# from mlxtend.plotting import plot_decision_regions
# from mlxtend.preprocessing import shuffle_arrays_unison
# import seaborn as sns

# from tqdm.notebook import tqdm_notebook as tqdm

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error
# import operator
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import RidgeCV
# from sklearn.linear_model import LassoCV
# from sklearn.linear_model import SGDRegressor
# from sklearn.neighbors import KNeighborsRegressor

# from math import log2, sqrt, isnan
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from sklearn import tree
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import six
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.metrics import make_scorer
# from sklearn.tree import export_graphviz
# from six import StringIO  
# from IPython.display import Image  
# import pydotplus
# from tqdm.notebook import tqdm_notebook as tqdm
# from sklearn.model_selection import GridSearchCV

# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# from collections import Counter

# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt
# from keras.layers import Dropout
# from keras import regularizers
# from keras import backend as K
# from keras import callbacks


# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import RFE

# STEP 1: IMPORT DATASET
# 1-1. import csv file
data = pd.read_csv('world-happiness-score-2020.csv')
# 1-2. check if the dataset has been imported successfully
data.head()

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
data.isnull().sum()
# 2-4. determine what % of each feature is null to figure out which null values to drop
print('[% OF NULL VALUES OF EACH FEATURE')
for col in data.columns:
    print(col,  ': %.2f%%' %((data[col].isnull().sum() / data.shape[0]) * 100))
print('')
# 2-5. drop null values of the features that have the least percentage
#      to avoid dropping lots of data entries
data = data[data[''].notna()]
data = data[data[''].notna()]
data = data[data[''].notna()]
# 2-6. check how many non-null values each feature has
data.info()
# 2-7. for other null values, use euclidean distance & uniform weights
#      to fill the null values with the mean of k (= 40) nearest neighbours
imputer = KNNImputer(n_neighbors = 40, copy = False) # copy is set to False to make imputation happen in place
imputer.fit_transform(data)
# 2-8. check how many non-null values each feature has
#      (all feature should have the same # of non-null values at this step)
data.info()
# 2-9. check correlation between each feature and target
#      to see if we need to drop more before getting into modelling
data.corr()['Ladder']

# STEP 3: SPLIT DATA
# 3-1. set X to be all features and y to be target
X = data.iloc[:,1:]
y = data.iloc[:, 0]
# 3-2. split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# STEP 4: BUILD REGRESSION MODEL

