# Databricks notebook source
# MAGIC %md
# MAGIC ## Workshop Session 2.0 Train ML model
# MAGIC ### Section 2.2 Grid Search Parameter Tuning
# MAGIC 
# MAGIC ![](/files/images/plan8.png)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### Spark Grid Search
# MAGIC 
# MAGIC We leverage the Spark cluster compute to train over 100,000 models in a few hours in a naive grid search.
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/spark_grid_search.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### MLFlow
# MAGIC 
# MAGIC We also track all of our experiments in MLFlow for easy tracking of our progress.
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/mlflow-1.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Importing required libraries
import numpy as np
import mlflow

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from itertools import product


# COMMAND ----------

# DBTITLE 1,Importing Data
# Available FE datasets to run
# 1. for business locatoins: "workshop.starbucks_features_business"
# 2. for residential locations: "workshop.starbucks_features_residential"
# 3. for residential and business locations: "workshop.starbucks_features"

starbucks_features = spark.sql("select * from workshop.starbucks_features").toPandas().dropna(axis=1)

# COMMAND ----------

# DBTITLE 1,Scale and split data into test/train
scaler = StandardScaler()
data_z = scaler.fit_transform(starbucks_features.drop(['PB_KEY', 'review_count', 'stars'],  axis=1))

X_train, X_test, Y_train, Y_test = train_test_split(
  data_z,
  starbucks_features['stars'],
  test_size=0.2, random_state=42)

# COMMAND ----------

# DBTITLE 1,Evaluation Functions & Metrics
def mape(y_true, y_pred):
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mae(y_true, y_pred):
  return np.mean(np.abs(y_true - y_pred))

# COMMAND ----------

# DBTITLE 1,Making Model Parameters for Grid Search
# create hyper param grid for search

n_estimators_grid = range(10,50)
criterion_grid = ['mse', 'mae']
bootstrap_grid = [True]
max_samples_grid = [(i+1)*0.1 for i in range(9)]

# create all hyperparam sets
training_grid = spark.createDataFrame(
    data=product(n_estimators_grid, criterion_grid, bootstrap_grid, max_samples_grid),
    schema=['n_estimators','criterion','bootstrap','max_samples']
  )

display(training_grid)

# COMMAND ----------

# DBTITLE 1,Create worker training function
# training function to run on each row of hyper param grid
def train_par(params, x_train, x_test, y_train, y_test):
  params_dict = params.asDict()
  
  with mlflow.start_run():
    # create model
    rfr = RFR(**params_dict)
    rfr.fit(x_train, y_train)
    
    # evaluate
    preds = rfr.predict(x_test)
    mape_res = mape(y_test, preds)
    mae_res = mae(y_test, preds)
    
    mlflow.log_metric('mape', mape_res)
    mlflow.log_metric('mae', mae_res)

    # log params
    mlflow.log_params(params_dict)
    mlflow.log_param('model type', 'randomForrestRegressor')
    mlflow.log_param('scaling', 'standard')


# COMMAND ----------

# DBTITLE 1,Train models in parallel
# train models in parallel on spark workers
res = (
  training_grid
  .rdd
  .map(lambda row: train_par(row, X_train, X_test, Y_train, Y_test))
  .collect()
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC Lets look at the results. Find the best model according to MAPE
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/mlfow_best.png" style="width: 50%; height: auto;">
# MAGIC </div>
# MAGIC 
# MAGIC - max_samples = 0.3
# MAGIC - model = randomForestRegressor
# MAGIC - n_estimators = 20
# MAGIC - criterion = mse
# MAGIC - scaling = none

# COMMAND ----------


