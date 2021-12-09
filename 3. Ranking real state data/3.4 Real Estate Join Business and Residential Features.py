# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Session 3.0 Ranking Real State Data
# MAGIC 
# MAGIC ###  3.4 Feature Engineer Real State Data - Join Business and Residential Features
# MAGIC 
# MAGIC ![](/files/images/plan12.png)

# COMMAND ----------

import databricks.koalas as ks
import pyspark.sql.functions as F

from sklearn.preprocessing import quantile_transform

from pyspark.sql.functions import lit, when, rand, col
 
#import dython
import math, random
 
import numpy as np
import pandas as pd
 
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql.functions import concat, concat_ws

#load business features
business_features_raw = spark.sql("select * from workshop.real_estate_features_business")
nrow_business = business_features_raw.count()
#load residential features
residential_features_raw = spark.sql("select * from workshop.real_estate_features_residential")
nrow_residential = residential_features_raw.count()
#check PB_KEYs
res_PBKs = residential_features_raw.select('PB_KEY').distinct().collect()
res_PBKs = [row[0] for row in res_PBKs]
bus_PBKs = business_features_raw.select('PB_KEY').distinct().collect()
bus_PBKs = [row[0] for row in bus_PBKs]
nstarbucks_residential = len(res_PBKs)
nstarbucks_business = len(bus_PBKs)
print(f"loaded {nrow_business} business rows with {nstarbucks_business} starbucks")
print(f"loaded {nrow_residential} residential rows with {nstarbucks_residential} starbucks")
res_only = sorted(set(res_PBKs) - set(bus_PBKs))
bus_only = sorted(set(bus_PBKs) - set(res_PBKs))
print(f"analyzing starbucks in both business and residential...")
print(f"starbucks in residential only: {res_only}")
print(f"starbucks in buss_only only: {bus_only}")
#dropping those starbucks that have only residential or only business data
print("keeping starbucks that have both residential and business information...")
res_common_rows = residential_features_raw.filter(~col('PB_KEY').isin(res_only))
bus_common_rows = business_features_raw.filter(~col('PB_KEY').isin(bus_only))
bus_common_rows = bus_common_rows.drop_duplicates(['PB_KEY'])
print(f"number of starbucks with BOTH residential and business...")
print(f" {bus_common_rows.count()} from business feature engineering")
print(f" {res_common_rows.count()} from residential feature engineerig")
#just in case we want to show example of which starbucks have only residential or only business
#bus_only_rows = business_features_raw.filter(col('PB_KEY').isin(bus_only))
#display(bus_only_rows)
#res_only_rows = residential_features_raw.filter(col('PB_KEY').isin(res_only))
#display(res_only_rows)
#features list
key_cols = ['PB_KEY']
res_cols = residential_features_raw.columns
bus_cols = business_features_raw.columns
res_feature_cols = sorted(set(res_cols) - set(key_cols))
bus_feature_cols = sorted(set(bus_cols) - set(key_cols))
print(f"total number of proximity features per Starbucks...")
print(f" {len(bus_feature_cols)} business features ")
print(f" {len(res_feature_cols)} residential features")
#print(key_cols)
#print(res_feature_cols)
#print(bus_feature_cols)
feature_cols = bus_feature_cols + res_feature_cols
#let's join
residential_features = res_common_rows
business_features = bus_common_rows
starbucks_features = business_features.join(residential_features, on = ['PB_KEY'], how = 'inner')
feature_cols2 = len(set(starbucks_features.columns) - set(['PB_KEY']))
#print("     ")
print(f"number of features: {feature_cols2} in dataframe {len(feature_cols)} in column list")
print(starbucks_features.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Lake

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS workshop.real_state_features")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS workshop.real_estate_features")
starbucks_features.write.format("delta").saveAsTable("workshop.real_estate_features")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from workshop.real_estate_features

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from workshop.real_estate_features

# COMMAND ----------


