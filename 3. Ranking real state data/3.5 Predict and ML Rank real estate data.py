# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Session 3.0 Ranking Real State Data
# MAGIC 
# MAGIC ###  3.5 Predict and ML Rank Real Estate Data
# MAGIC 
# MAGIC ![](/files/images/plan13.png)

# COMMAND ----------

import os
import warnings
import sys

import pandas as pd
import numpy as np
import random

from itertools import cycle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

import  pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import lit, when, rand, col

#import mlflow
#import mlflow.sklearn

# COMMAND ----------

# DBTITLE 1,Re-Train Best Model (Chosen from HyperParameters notebook)
starbucks_features = spark.sql("select * from workshop.starbucks_features")

params = {'n_estimators': 51, 'max_depth': 3, 'criterion': 'mse'}
# params = {'n_estimators': 28, 'max_samples': 0.2, 'criterion': 'mae'}

train = starbucks_features
train = train.drop('review_count', 'PB_KEY') 
train = train.na.fill(0).toPandas()

y = train['stars']   
X = train.drop(columns=['stars'])
Xtrain,Xcv,ytrain,ycv = train_test_split(X, y, test_size=0.30, random_state=42)

rfr = RandomForestRegressor(**params)
rfr.fit(Xtrain,ytrain)

def mape_calc(y_true, y_pred):
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = mape_calc(actual, pred)
    return mape, rmse, mae
  
yp = rfr.predict(Xcv)
mape, rmse, mae = eval_metrics(yp,ycv)
print(f"evaluation errors...")
print(f"   mape: {mape}")
print(f"   rmse: {rmse}")
print(f"   mae: {mae}")

# COMMAND ----------

# DBTITLE 1,Score real estate data
real_estate_features_with_PBKEY = spark.sql("select * from workshop.real_estate_features").toPandas()
real_estate_features = spark.sql("select * from workshop.real_estate_features").drop('PB_KEY').toPandas()

# predicting Star Rates for each Real Estate
real_estate_predicted_stars = rfr.predict(real_estate_features)

PBKEYs = real_estate_features_with_PBKEY['PB_KEY'].values
real_estate_predictions = list(zip(list(PBKEYs), list(real_estate_predicted_stars)))

ranked_real_state_list1 = sorted(real_estate_predictions, key = lambda x:x[1], reverse = True)
winners_l = ranked_real_state_list1[:3]

#print(ranked_real_state_list1) 
for row in winners_l:
  print(row[0], row[1])

# COMMAND ----------

[w[0] for w in winners_l]

# COMMAND ----------

# DBTITLE 1,Lets see the address of the top 3 winners
winners = (
  spark.table("workshop.real_estate_geocoded")
  .drop("error", "precisionCode", "address", "_city")
  .filter(F.col("PB_KEY").isin([w[0] for w in winners_l]))
).toPandas()

scores_df = pd.DataFrame(data=winners_l, columns=['PB_KEY','score'])
merger = (
  winners.merge(scores_df, on=['PB_KEY'])[['PB_KEY', 'score', 'formattedStreetAddress', 'formattedLocationAddress', 'X', 'Y', '_price']]
)
display(merger)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <img src="/files/images/winner_pic.png" style="width:50%; height:auto;">
# MAGIC 
# MAGIC <br/>
# MAGIC <br/>
# MAGIC 
# MAGIC <img src="/files/images/winner2.png" style="width:50%; height:auto;">

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional Exploration - Combine ML Star Predictions with Additional Revenue and Demographics Data
# MAGIC 
# MAGIC * We could say the work is done but at Blueprint we dont stop there.... Because, as data scientists, we like to make sure we made all the right decisions, in a production environment we would very likely built additional models to make the Decision of choosing the right location more informed and robust (for example, including Predicted Revenue, Predicted Cost of Operation, Predicted ROI, and other metrics)
# MAGIC * below a fast quick example of including additional metrics to get to a combined score

# COMMAND ----------

# DBTITLE 1,Grab Additional Real Estate Data, including Precisely ID, Formatted Address, Star Rating, Price, and Size 
#join with additional real state data
real_state_geocoded = spark.table("workshop.real_estate_geocoded").drop("error", "precisionCode", "address", "_city").toPandas()
ranked_real_state1 = pd.DataFrame(ranked_real_state_list1, columns = ["PB_KEY", "stars_score"])
#print(ranked_real_state1.dtypes)
#print(real_state_geocoded.dtypes)

ranked_real_state2 = pd.merge(ranked_real_state1, real_state_geocoded, on=["PB_KEY"], how = 'left')
#ranked_real_state2
#ranked_real_state1.head()
#print(ranked_real_state1.dtypes)
display(ranked_real_state2)

# COMMAND ----------

# DBTITLE 1,Join with Precisely ID datasets - For example, join Real Estate Geofenced Data with Point of Interest Data to get Additional Business Metrics
#load geofenced data
geofenced_raw_cols = ['PB_KEY', 'PBKEY']
prop_type_cols = ['PBKEY', 'PROP_TYPE']

geofenced_raw = spark.sql("select * from workshop.real_estate_geofenced")
#Address Fabric
US_AF_addresses =  spark.sql(f"select * from gv_usa_address_fabric_001")
#geocoded = spark.sql("select A.*, B.PROP_TYPE from workshop.geofenced_starbucks A LEFT JOIN gv_usa_address_fabric_001 B ON A.PBKEY = B.PBKEY")
geofenced = geofenced_raw.join(US_AF_addresses.select(prop_type_cols), on = ['PBKEY'], how = 'left')

#split into business and residential
geofenced_residential = geofenced.filter(col('PROP_TYPE') == 'R')
geofenced_business = geofenced.filter(col('PROP_TYPE') == 'B')

#grab POIs data
POI_USA = spark.sql(f"select * from usa_points_of_insterest_001")
#print(POI_USA.columns)
poi_sales_cols = ['PBKEY', 'SALES_VOLUME_LOCAL', 'SALES_VOLUME_US_DOLLARS']
poi_sales_cols_to_keep = ['SALES_VOLUME_LOCAL']
geofenced_business2 = geofenced_business.select(['PB_KEY', 'PBKEY'])
geofenced_business3 = geofenced_business2.join(POI_USA.select(*poi_sales_cols), on = ['PBKEY'], how = 'left')
geofenced_business4 = geofenced_business3.na.drop(subset=["SALES_VOLUME_LOCAL"])
print(geofenced_business4.count())
display(geofenced_business4)

# COMMAND ----------

# DBTITLE 1,Calculate Median of Sales Volume Local based on Geofence
temp3 = geofenced_business4.drop(*['PBKEY'])
table_name = "temp4"
temp3.createOrReplaceTempView(table_name)
comm_template = "percentile_approx(%s, 0.5) AS %s"

comm_list = []
for col_name in poi_sales_cols_to_keep:
    #comm_template = f"percentile_approx({col_name}, 0.5)"
    comm_str = comm_template % (col_name, f"median_{col_name}")
    comm_list.append(comm_str)
    
comm_str = ", ".join(comm_list)
#print(comm_str)
query_template = f"SELECT PB_KEY, %s FROM {table_name} GROUP BY PB_KEY"
query = query_template % (comm_str)
print(query)

eng_features4 = spark.sql(query)
print("local sales medians...")
print(eng_features4.columns)
display(eng_features4)

# COMMAND ----------

# DBTITLE 1,Create a combined Score
out_cols = ['PB_KEY', 'stars_score', '_price', 'median_SALES_VOLUME_LOCAL', '_size', '_unit', 'formattedStreetAddress', 'formattedLocationAddress',  'X', 'Y', '_state']
sales_local = eng_features4.toPandas()
ranked_real_state3 = pd.merge(ranked_real_state2, sales_local, on=["PB_KEY"], how = 'left')
ranked_real_state3 = ranked_real_state3[out_cols]
ranked_real_state3.head()

#Business can select weights for each metric
stars_weight = 2
price_weight = 2
sales_weight = 1
size_weight = 0.5

combined_score = ranked_real_state3[['PB_KEY', 'stars_score',	'_price',	'median_SALES_VOLUME_LOCAL', '_size']].copy()
combined_score['max_price'] =  max(ranked_real_state3['_price'])
combined_score['min_price'] =  min(ranked_real_state3['_price'])
combined_score['max_sales'] = max(ranked_real_state3['median_SALES_VOLUME_LOCAL'])
combined_score['min_sales'] = min(ranked_real_state3['median_SALES_VOLUME_LOCAL'])
combined_score['max_stars'] = max(ranked_real_state3['stars_score'])
combined_score['min_stars'] = min(ranked_real_state3['stars_score'])
combined_score['max_size'] = max(ranked_real_state3['_size'])
combined_score['min_size'] = min(ranked_real_state3['_size'])
combined_score['adj_price'] = (combined_score['_price'] - combined_score['min_price']) / (combined_score['max_price'] - combined_score['min_price'])
combined_score['adj_sales'] = (combined_score['median_SALES_VOLUME_LOCAL'] - combined_score['min_sales']) / (combined_score['max_sales'] - combined_score['min_sales'])
combined_score['adj_stars'] = (combined_score['stars_score'] - combined_score['min_stars']) / (combined_score['max_stars'] - combined_score['min_stars'])
combined_score['adj_size'] = (combined_score['_size'] - combined_score['min_size']) / (combined_score['max_size'] - combined_score['min_size'])
combined_score['score'] = stars_weight * combined_score['adj_stars'] + \
                          sales_weight * combined_score['adj_sales'] - \
                          price_weight * combined_score['adj_price'] + \
                          size_weight * combined_score['adj_size']

combined_score2 = combined_score[['PB_KEY', 'score', 'stars_score',	'_price',	'median_SALES_VOLUME_LOCAL', '_size']]
combined_score2 = combined_score2.sort_values( by = ['score'], ascending = False)
display(combined_score2)


# COMMAND ----------

 
