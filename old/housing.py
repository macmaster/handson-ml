# housing.py
# some california housing data analysis...
# 
# Author: Ronny Macmaster

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import custom

DATAPATH = "datasets/housing"
MODELPATH = "models"

def build_test_set():
    # stratified sampling.
    housing = pd.read_csv("%s/housing.csv" % DATAPATH)
    strat_col, strat_pivot = ("income_cat", 5.0)
    housing[strat_col] = np.ceil(housing["median_income"] / 1.5)
    housing[strat_col].where(housing[strat_col] < strat_pivot, strat_pivot, inplace=True)
    
    # set aside a training set
    train, test = custom.strat_part(housing, 0.2, strat_col)
    train.drop(strat_col, axis=1, inplace=True)
    test.drop(strat_col, axis=1, inplace=True)
    train.to_csv("%s/train.csv" % DATAPATH)
    test.to_csv("%s/test.csv" % DATAPATH)

def visualize():
    # read dataset
    train = pd.read_csv("%s/train.csv" % DATAPATH)
    test = pd.read_csv("%s/test.csv" % DATAPATH)

    # visualize the geo data
    print train.columns
    train.plot(x="longitude", y="latitude", kind="scatter", alpha=0.5,
        s=(train["population"] / 100), label="population", 
        cmap=plt.get_cmap("jet"), c="median_house_value", colorbar=True)
    plt.legend()
    plt.show()
    
    # correlation coeffs
    train["rooms_per_household"] = train["total_rooms"] / train["households"]
    train["bedrooms_per_room"] = train["total_bedrooms"] / train["total_rooms"]
    train["population_per_household"] = train["population"] / train["households"]
    print train.corr()["median_house_value"].sort_values(ascending=False)
    features = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    pd.scatter_matrix(train[features])
    plt.show()
    
    # focus on income
    train.plot(x="median_income", y="median_house_value", kind="scatter", alpha=0.4)
    plt.show()

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, bpr=False):
        self.bpr = bpr
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None): 
        rooms_per_household = train["total_rooms"] / train["households"]
        population_per_household = train["population"] / train["households"]
        bedrooms_per_room = train["total_bedrooms"] / train["total_rooms"]
        comb = np.c_[X, rooms_per_household, population_per_household]
        if self.bpr: return np.c_[comb, bedrooms_per_room] 
        else: return comb

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X):
        return self
    def transform(self, X):
        return X[self.feature_names].values

# read dataset
train = pd.read_csv("%s/train.csv" % DATAPATH)
test = pd.read_csv("%s/test.csv" % DATAPATH)
print train.info(), "\n", test.info()

label_col = "median_house_value"
X = train.drop(label_col, axis=1)
y = train[label_col]

# preprocess the data with a feature pipeline
num_features = X.select_dtypes(exclude=["object"])
cat_features = X.select_dtypes(include=["object"])

num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_features.columns)),
    ("attrib_adder", CombinedAttributesAdder()),
    ("imputer", Imputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_features.columns)),
    ("label_binarizer", LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),    
])

X = full_pipeline.fit_transform(X)
print X, "\n", X.shape

# # fit a linear regression model
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# 
# from sklearn.metrics import mean_squared_error
# lin_mse = mean_squared_error(y, lin_reg.predict(X))
# lin_rmse = np .sqrt(lin_mse)
# print "linear rmse: {0}".format(lin_rmse)
# 
# # fit a nonlinear tree model
# from sklearn.tree import DecisionTreeRegressor
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(X, y)
# 
# tree_mse = mean_squared_error(y, tree_reg.predict(X))
# tree_rmse = np.sqrt(tree_mse)
# print "tree rmse: {0}".format(tree_rmse)
# 
# # cross validation training
# from sklearn.model_selection import cross_val_score
# tree_scores = cross_val_score(tree_reg, X, y, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-tree_scores)
# lin_scores = cross_val_score(lin_reg, X, y, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# print "tree rmse cv scores: {0}".format(tree_rmse_scores)
# print "tree rmse cv mean: {0}".format(tree_rmse_scores.mean())
# print "linear rmse cv scores: {0}".format(lin_rmse_scores)
# print "linear rmse cv mean: {0}".format(lin_rmse_scores.mean())
# 
# # save the models for later use
# from sklearn.externals import joblib
# joblib.dump(lin_reg, "%s/housing_lin_reg.pkl" % MODELPATH)
# joblib.dump(lin_reg, "%s/housing_tree_reg.pkl" % MODELPATH)

# grid search a random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
grid = {
    "n_estimators" : [3, 10, 30],
    "max_features" : [2, 4, 6, 8],
}

grid_search = GridSearchCV(
    RandomForestRegressor(), grid, cv=10, scoring="neg_mean_squared_error")
grid_search.fit(X, y)
print grid_search.best_params_

# save the models for later use
from sklearn.externals import joblib
joblib.dump(grid_search.best_estimator_, "%s/housing_rf_reg.pkl" % MODELPATH)
