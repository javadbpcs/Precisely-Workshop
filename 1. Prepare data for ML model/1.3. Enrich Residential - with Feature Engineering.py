# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Workshop Session 1.0 Prepare data for ML model
# MAGIC 
# MAGIC ### Section 1.3 Enrich geofenced business and residential locations differently
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/plan4a.png" style="width: 100%; height: auto;">
# MAGIC </div>
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC 
# MAGIC #### 1.3.1 Enrich Residential data
# MAGIC 
# MAGIC In this notebook, we will enrich the 8M+ addresses resulting from the geofence with relevant industry specific insights and trends, as well, as general core demographics data, using Precisely datasets availble for the Hackathon.
# MAGIC 
# MAGIC For example:
# MAGIC 
# MAGIC * We will identify some **Coffee, Food, and Snacks Consumer Spend and Consumer specific trends** and
# MAGIC * Add additional core demographics statistics such as, **Population Census**, **Household Income**, and **Audience Profiles** 
# MAGIC * The audience profiles dataset provides insights into specifics costumer preferences and behaviors.
# MAGIC * The consumer spend datasets allow to identify patterns and preferences, including present, past, and future consumer spends projections 
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/enrich_join.png" style="width: 50%; height: auto;">
# MAGIC </div>
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC #### 1.3.2 Aggregate enriched Geofenced locations into “proximity summary features”
# MAGIC 
# MAGIC * To create a **Precisely data** enriched dataset of proximity summary features for each of our **600 Starbucks locations**
# MAGIC 
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/collapse_features.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3.1 Residential Address Enrichment 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Available Datasets
# MAGIC In your Databricks Environment, we have made available as both files and Delta Lake tables (read-only) the following datasets:
# MAGIC 
# MAGIC * Address Fabric
# MAGIC * Audience Profile
# MAGIC * Consumer Expenditure
# MAGIC * Family Households
# MAGIC * Household Assets
# MAGIC * Household Income
# MAGIC * Population
# MAGIC * Online Reviews
# MAGIC * Streets and Traffic Information
# MAGIC * Suites, Appartments, and Lot sub-divisions.
# MAGIC * Notebooks with documentation and locations of these datasets can be found in your Team-<no>/Datasets folder.
# MAGIC 
# MAGIC You also have the option to Bring Your Own Data (BYOD).

# COMMAND ----------

# DBTITLE 1,Load Precisely Data
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

#Address Fabric
US_AF_addresses =  spark.sql(f"select * from gv_usa_address_fabric_001")

#Points of Interest
US_POI = spark.sql(f"select * from usa_points_of_insterest_001")

#Audience Profile
US_AP_all_regions = spark.sql(f"select * from gv_usa_audience_profile_all_regions_001")
US_AP_all_regions = US_AP_all_regions.withColumnRenamed("Latitude", "AP_geoid_latitude").withColumnRenamed("Longitude", "AP_geoid_longitude")  
US_AP_all_regions = US_AP_all_regions.withColumnRenamed("popby", "AP_popby")
US_AP_all_regions = US_AP_all_regions.drop('Model_Profile_ND')

#Consumer Expenditure
US_SAUS_CEP1_CY = spark.sql(f"select * from gv_usa_consumer_exp_saus_CEP1_001")
US_SAUS_CEP1_CY = US_SAUS_CEP1_CY.drop("type")

#Household Family 
US_SS_HH_family_2k = spark.sql(f"select * from gv_usa_family_household_census_2k_001")
US_SS_HH_family_2k = US_SS_HH_family_2k.drop("type")
US_SS_HH_family_2k = US_SS_HH_family_2k.withColumnRenamed("hh2k", "family_hh2k")

#Household Income
US_S_HH_income_2k  = spark.sql(f"select * from  gv_usa_household_income_census_2K_001")
US_S_HH_income_2k  = US_S_HH_income_2k.drop("type")

#Population (census 2010)
US_DS1_population_BY  = spark.sql(f"select * from  gv_usa_population_census_BY_001")
US_DS1_population_BY = US_DS1_population_BY.drop("type")
#US_DS1_population_BY = US_DS1_population_BY.withColumnRenamed("popby", "census_BY_popby")

#Population (census 2000)
US_DS1_population_2K  = spark.sql(f"select * from  gv_usa_population_census_2K_001")
US_DS1_population_2K = US_DS1_population_2K.drop("type")

## Identify Key columns
  
#Address Fabric
key_cols = ["PBKEY", "GEOID", "CODE"]
latitude_cols = ["Latitude", "Longitude"]   
AF_lat_cols = ['LAT', 'LON'] 
AF_non_feature_cols = ['PARENT', 'TYPE']
AF_address_cols = ['ADD_NUMBER', 'STREETNAME', 'UNIT_DES', 'UNIT_NUM', 'CITY', 'STATE', 'ZIPCODE', 'PLUS4']
AF_features = ['CITY', 'STATE', 'ZIPCODE', 'PLUS4']

#Audience Profile
AP_key_cols = ["AP_CODE"]  #AP_key_cols = ["code"]
AP_features = US_AP_all_regions.columns
AP_features.remove("code")
# removing db_source_file from features as it is the name of the original precisely source file 
# For example: North East Audience Profiles would appear NE_US_AudienceProfile_2021_06.txt which would be just the name of a file not a demographics feature
AP_features.remove("db_source_file")
AP_features.remove("AP_geoid_latitude")	
AP_features.remove("AP_geoid_longitude")	

#Consumer Expense
cep_key_cols = ["PBKEY", "CODE"]
cep_features = US_SAUS_CEP1_CY.columns
for k in cep_key_cols:
  cep_features.remove(k)
  
#Family Household
family_key_cols = ["PBKEY", "CODE"]
family_features = US_SS_HH_family_2k.columns
for k in family_key_cols:
  family_features.remove(k)
  
#Household Income  
income_key_cols =['PBKEY', 'CODE']
income_features = US_S_HH_income_2k.columns 
for k in income_key_cols:
  income_features.remove(k)
  
#Population (census 2010)
population_key_cols = ['PBKEY', 'CODE']
population_features = US_DS1_population_BY.columns
for k in population_key_cols:
  population_features.remove(k)
feature_cols = AF_features + AP_features + cep_features + family_features + income_features + population_features
print(feature_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC * Geofencing produced 8M+ addresses

# COMMAND ----------

# DBTITLE 1,Load Residential Geofenced Data - GEOID & Census Blocks
# test batched processing approach if needed
geo_fence_raw1 =  spark.sql("select * from workshop.geofenced_starbucks_residential")
geo_fence_raw2 = geo_fence_raw1.na.drop(subset=["PB_KEY"])
pbkey_batches = spark.sql("select distinct PB_KEY, batch_id from team30.geocoded_locations_w_batch")
geo_fence_raw3 = geo_fence_raw2.join(pbkey_batches, on = ['PB_KEY'], how = 'left')
geo_fence_raw4 = geo_fence_raw3.drop_duplicates()
geo_fence_raw4.createOrReplaceTempView("geo_fence_raw4")
#print(geo_fence_raw4.count())
#display(geo_fence_raw4)

#Split into batches (or Not)
#nbatches = 3
#batch_ids = geo_fence_raw4.select("batch_id").distinct().collect()
#batch_ids = [row[0] for row in batch_ids]
#to_include = batch_ids[0:nbatches]
#geo_fence_raw5 = geo_fence_raw4.filter(col('batch_id').isin(to_include)) 
#geo_fence_raw5.createOrReplaceTempView("geo_fence_raw5")
#print(to_include)
#print(geo_fence_raw5.count())

#add GEOID to geofenced data
#GEOID is needed to join with Audience Profile dataset
geo_fence_raw6 = spark.sql(
"""
SELECT 
  PB_KEY,
  B.PROP_TYPE,
  B.PBKEY,
  ZIPCODE,
  GEOID
FROM geo_fence_raw4 A
LEFT JOIN gv_usa_address_fabric_001 B 
ON A.PBKEY = B.PBKEY
""")

geo_fence_raw7 = geo_fence_raw6.na.drop(subset=["PROP_TYPE"])
print(geo_fence_raw7.count())

#sanity check only residential and only the colummns we need
geo_fence_raw = geo_fence_raw7
key_goefenced_cols =  ['PB_KEY', 'PROP_TYPE', 'PBKEY', 'ZIPCODE', 'GEOID']
res_addresses = geo_fence_raw.select(*key_goefenced_cols)
#print(res_addresses.count())

res_addresses = res_addresses.drop_duplicates(key_goefenced_cols)
print(res_addresses.count())
#display(res_addresses)

geo_fence_raw7.select('PROP_TYPE').drop_duplicates().show()
display(geo_fence_raw7)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Enrich Residential Data using "Small" demographics joins

# COMMAND ----------

# DBTITLE 1,Audience Profiles
verbose = True
demo1 = res_addresses.join(US_AP_all_regions, res_addresses.GEOID == US_AP_all_regions.code, how='left')
#Please Note: AP_CODE is different from CODE, AP_CODE corresponds to GEOID (the difference is 12 digits vs 15 digits code)
# NOTE: do not join on A.AP_CODE == B.CODE
# for example: this won't work demo3 = demo1.join(demo2, on = ["PBKEY", "CODE"]) because AP_CODE != CODE
#print(demo1.columns)
demo1 = demo1.withColumnRenamed("code", "AP_CODE")

if(verbose): 
  display(demo1)

# COMMAND ----------

# DBTITLE 1,Family Households

demo2 = res_addresses.join(US_SS_HH_family_2k, res_addresses.PBKEY == US_SS_HH_family_2k.PBKEY, how='left')
if(verbose):
  display(demo2)

# COMMAND ----------

# DBTITLE 1,Household Income
#Household Income
demo3 = res_addresses.join(US_S_HH_income_2k, res_addresses.PBKEY == US_S_HH_income_2k.PBKEY, how='left')
if(verbose):
  display(demo3)

# COMMAND ----------

# DBTITLE 1,Consumer Expenditure
#Consumer Expenditure
demo4 =res_addresses.join(US_SAUS_CEP1_CY, on = ["PBKEY"], how = 'left')
if(verbose):
  display(demo4)

# COMMAND ----------

# DBTITLE 1,Population Census
#per Population (census 2010)
demo5 =res_addresses.join(US_DS1_population_BY, on = ["PBKEY"], how = 'left')
if(verbose):
  display(demo5)

# COMMAND ----------

#sanity check repeated columns 
cnt_cols = pd.DataFrame(demo5.columns, columns = ["col_name"])
cnt_cols["cnt"] = 1
cnt_cols2 = cnt_cols.groupby(["col_name"])[["cnt"]].sum().reset_index().sort_values(["cnt"], ascending = False)
cnt_cols2.head()

# COMMAND ----------

# DBTITLE 1,Drop Some Random Residential Features (To quickly explore the feature space)
random.seed(42)
#we could use two strategies: same_number or same_perc
#same_perc --> results in more columns for wider tables
#same_number --> results in same columns for each table
#here we choose the smaller of the two i.e. same number
drop_strategy = 'same_number' 
if(drop_strategy == 'same_number'):
  perc_to_drop = None
  fixed_sample_size = 5
else:
  perc_to_drop = 0.9
  fixed_sample_per_table = None
  
AF_TABLE_ALIAS = 'AF'
POI_TABLE_ALIAS = 'POI'
AP_TABLE_ALIAS = 'AP'
CEP_TABLE_ALIAS = 'CEP1_CY'
HH_FAMILY_TABLE_ALIAS = 'HH_family_2k'
HH_INCOME_TABLE_ALIAS = 'HH_income_2k'
POP_TABLE_ALIAS = 'DS1_population_2k'

#Address Fabric
#AF_features = ['CITY', 'STATE', 'ZIPCODE', 'PLUS4']
#Audience Profile
#AP_features = US_AP_all_regions.columns
print(f"total number of Audience Profile features: {len(AP_features)}")

ap_cols_to_table_map = dict(zip(AP_features, len(AP_features) * [AP_TABLE_ALIAS] ))
if(perc_to_drop is not None):
  ap_to_drop = int(perc_to_drop * len(AP_features))
  ap_to_keep = len(AP_features) - ap_to_drop
else:
  ap_to_keep = fixed_sample_size
ap_cols_to_keep = random.sample(AP_features, ap_to_keep)
print(f"keeping {len(ap_cols_to_keep)}")
print(ap_cols_to_keep)

#Consumer Expense
print(f"total number of Consumer Expense features: {len(cep_features)}")
cep_cols_to_table_map = dict(zip(cep_features, len(cep_features) * [CEP_TABLE_ALIAS] ))
if(perc_to_drop is not None):
  cep_to_drop = int(perc_to_drop * len(cep_features))
  cep_to_keep = len(cep_features) - cep_to_drop
else:
  cep_to_keep = fixed_sample_size
cep_cols_to_keep = random.sample(cep_features, cep_to_keep)
print(f"keeping {len(cep_cols_to_keep)}")
print(cep_cols_to_keep)

#Family Household
print(f"total number of Family Household features: {len(family_features)}")
family_cols_to_table_map = dict(zip(family_features, len(family_features) * [HH_FAMILY_TABLE_ALIAS] ))
if(perc_to_drop is not None):
  family_to_drop = int(perc_to_drop * len(family_features))
  family_to_keep = len(family_features) - family_to_drop
else:
  family_to_keep = fixed_sample_size
family_cols_to_keep = random.sample(family_features, family_to_keep)
print(f"keeping {len(family_cols_to_keep)}")
print(family_cols_to_keep)

#Household Income  
print(f"total number of Household Income features: {len(income_features)}")  
income_cols_to_table_map = dict(zip(income_features, len(income_features) * [HH_INCOME_TABLE_ALIAS] ))
if(perc_to_drop is not None):
  income_to_drop = int(perc_to_drop * len(income_features))
  income_to_keep = len(income_features) - income_to_drop
else:
  income_to_keep = fixed_sample_size
income_cols_to_keep = random.sample(income_features, income_to_keep)
print(f"keeping {len(income_cols_to_keep)}")
print(income_cols_to_keep)

#Population (census 2010)
print(f"total number of Population features: {len(population_features)}")  
pop_cols_to_table_map = dict(zip(population_features, len(population_features) * [POP_TABLE_ALIAS] ))
if(perc_to_drop is not None):
  population_to_drop = int(perc_to_drop * len(population_features))
  population_to_keep = len(population_features) - population_to_drop
else:
  population_to_keep = fixed_sample_size
population_cols_to_keep = random.sample(population_features, population_to_keep)
print(f"keeping {len(population_cols_to_keep)}")
print(population_cols_to_keep)

#total number of features
#feature_cols = AF_features + AP_features + cep_features + family_features + income_features + population_features
keep_feature_cols = AF_features + ap_cols_to_keep + cep_cols_to_keep + family_cols_to_keep + income_cols_to_keep + population_cols_to_keep
print(f"total number features: {len(feature_cols)}")  
print(f"keeping {len(keep_feature_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC From a total of 730 Precisely Features we chosen 29 residential features which we assume to be our "core" representative features - for the time being.
# MAGIC 
# MAGIC In addition to these 29 features add more targeted features in the next paragraphs (these 29 are not the only ones)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's have some fun with some Feature Engineering
# MAGIC 
# MAGIC 
# MAGIC * first, we select some relevant features
# MAGIC * then, we aggregate on those features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Census Blocks
# MAGIC 
# MAGIC Census Blocks are the smallest level of geography you can get census data, such as total population by age, sex, and race, etc.
# MAGIC Block groups generally contain between 600 and 3,000 people, with an average usually around 1,500 people. The census block can be seen as a demographics "proxy" of a [Designated Market Area](https://www.precisely.com/data-guide/products/designated-market-areas-nielsen-dma-maps). 

# COMMAND ----------

# DBTITLE 1,Census Blocks per Starbucks Location
#define key cols
demo_key_cols = key_cols
geo_fence_key_cols = ["PB_KEY"]
features_key_cols = geo_fence_key_cols + demo_key_cols
print(features_key_cols)

#Add Basic Demographics (Population Data)
#per Population (census 2010)
pop_features_raw1 = demo5.select(*features_key_cols+population_cols_to_keep)
#display(pop_features_raw1)

#census blocks per location
temp = pop_features_raw1.select('PB_KEY', 'CODE').drop_duplicates()
census_block_cnt = temp.groupBy("PB_KEY").count()
census_block_cnt = census_block_cnt.withColumnRenamed("count", "code_cnt")
display(census_block_cnt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coffee, Food, and Snacks Consumer Spend and Consumer trends
# MAGIC 
# MAGIC ![](/files/images/starbucksfood10.png)
# MAGIC 
# MAGIC These columns contain the $spend on these products in the current year:
# MAGIC 
# MAGIC * ce010105043cy  2020 Tea
# MAGIC * ce010105044cy   2020 Coffee
# MAGIC * ce0101050441cy  2020 Roasted coffee
# MAGIC * ce0101050442cy  2020 Instant and freeze dried coffee
# MAGIC * ce0101010243cy              2020 Sweetrolls, coffee cakes, doughnuts
# MAGIC * ce010101022cy  2020 Cookies and crackersFLOA
# MAGIC * ce0101010221cy             2020 Cookies
# MAGIC * ce0101010222cy             2020 Crackers
# MAGIC * ce010101023cy  2020 Frozen and refrigerated bakery products
# MAGIC  
# MAGIC These columns contain the forecast demand on these products over the next 5 years:
# MAGIC  
# MAGIC * ce0101050435y               2025 Tea
# MAGIC * ce0101050445y 2025 Coffee
# MAGIC * ce01010504415y              2025 Roasted coffee
# MAGIC * ce01010504425y              2025 Instant and freeze dried coffee
# MAGIC * ce01010102445y              2025 Sweetrolls, coffee cakes, doughnuts
# MAGIC * ce01010102125y              2025 Cookies and crackers
# MAGIC * ce0101010225y 2025 Cookies
# MAGIC * ce01010102215y              2025 Crackers
# MAGIC * ce01010102225y              2025 Frozen and refrigerated bakery products

# COMMAND ----------

# DBTITLE 1,Grab Coffee Spend Columns and Join to Starbucks PBKEYs
coffee_spend_cols1 = [ 'ce010105043cy', \
'ce010105044cy',  \
'ce0101050441cy', \
'ce0101050442cy',  \
'ce0101010243cy', \
'ce010101022cy',  \
'ce0101010221cy', \
'ce0101010222cy',  \
'ce010101023cy', \
]
cep_cols = sorted(set(cep_cols_to_keep + coffee_spend_cols1))
print("coffee spend features...")
print(cep_cols)
pop_features_raw2 = demo4.select(*features_key_cols+cep_cols)
display(pop_features_raw2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Audience Profiles
# MAGIC 
# MAGIC Precicsely [Audience Profiles](https://docs.precisely.com/docs-gated/data/groundview-audience-profiles/2021/en-us/pdf/audience_profiles_usa_v2021_product_guide.pdf) dataset is based on actual mobile trace data from location-enabled applications (apps), which allow to identify targeted consumer profiles and behavioral characteristics at the "proximity" census block or neighborhood geographical area level.
# MAGIC 
# MAGIC for example:
# MAGIC * Number of People that visits Wendys
# MAGIC * Number of People that visits Burger King
# MAGIC * etc.
# MAGIC 
# MAGIC These [precisely datasets](https://www.precisely.com/press-release/precisely-launches-dynamic-demographics-data-offering-revealing-changing-profiles-over-time) can be both static and dynamic. 
# MAGIC ![](/files/images/preciselyDynamicDemographics.jpeg)

# COMMAND ----------

# DBTITLE 1,Audience Profiles per Geofenced Location
#Audience Profiles
#Note: Audience profiles AP_code != CODE
ap_features_key_cols = ['PB_KEY', 'PBKEY', 'GEOID']
#print(features_key_cols)
#print(ap_features_key_cols)
#print(ap_cols_to_keep)
pop_features_raw3 = demo1.select(*ap_features_key_cols+ap_cols_to_keep)
print("Audience Profiles features...")
print(pop_features_raw3.columns)
#display(pop_features_raw3)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### Aggregate Features per Geofenced Location
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/collapse_features.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Stats of Demographics per Geofenced Location
temp1 = pop_features_raw1.drop(*['PBKEY', 'GEOID', 'CODE'])
funcs = [ ('avg', F.avg), ('stddev', F.stddev), ('min', F.min),('max', F.max) ]
expr_cv = [f[1](F.col(c)).alias(f"{f[0]}_{c}") for f in funcs for c in population_cols_to_keep]

eng_features1_1 = temp1.groupby('PB_KEY').agg(*expr_cv)
print("population avgs...")
print(eng_features1_1.columns)

# COMMAND ----------

# DBTITLE 1,Median of Demographics per Geofenced Location
temp4 = pop_features_raw1.drop(*['PBKEY', 'GEOID', 'CODE'])
table_name = "temp4"
temp4.createOrReplaceTempView(table_name)
comm_template = "percentile_approx(%s, 0.5) AS %s"

comm_list = []
for col_name in population_cols_to_keep:
    #comm_template = f"percentile_approx({col_name}, 0.5)"
    comm_str = comm_template % (col_name, f"median_{col_name}")
    comm_list.append(comm_str)
    
comm_str = ", ".join(comm_list)
#print(comm_str)
query_template = f"SELECT PB_KEY, %s FROM {table_name} GROUP BY PB_KEY"
query = query_template % (comm_str)
print(query)

eng_features1_2 = spark.sql(query)
print("population medians...")
print(eng_features1_2.columns)
#display(temp5)

# COMMAND ----------

# DBTITLE 1,Stats for Consumer Spend per Location
temp2 = pop_features_raw2.drop(*['PBKEY', 'GEOID', 'CODE'])
funcs = [ ('avg', F.avg), ('stddev', F.stddev), ('min', F.min),('max', F.max) ]
expr_cv = [f[1](F.col(c)).alias(f"{f[0]}_{c}") for f in funcs for c in cep_cols]
eng_features2_1 = temp2.groupby('PB_KEY').agg(*expr_cv)
print("consumer spend avgs...")
print(eng_features1_1.columns)

# COMMAND ----------

# DBTITLE 1,Median of Consumer Expense per Geofenced Location
temp3 = pop_features_raw2.drop(*['PBKEY', 'GEOID', 'CODE'])
table_name = "temp4"
temp3.createOrReplaceTempView(table_name)
comm_template = "percentile_approx(%s, 0.5) AS %s"

comm_list = []
for col_name in cep_cols:
    #comm_template = f"percentile_approx({col_name}, 0.5)"
    comm_str = comm_template % (col_name, f"median_{col_name}")
    comm_list.append(comm_str)
    
comm_str = ", ".join(comm_list)
#print(comm_str)
query_template = f"SELECT PB_KEY, %s FROM {table_name} GROUP BY PB_KEY"
query = query_template % (comm_str)
print(query)

eng_features2_2 = spark.sql(query)
print("coffee spend medians...")
print(eng_features2_2.columns)
#display(temp5)

# COMMAND ----------

# DBTITLE 1,Stats of Audience Profiles
temp31 = pop_features_raw3.drop(*['PBKEY', 'GEOID', 'AP_CODE'])
funcs = [ ('avg', F.avg), ('stddev', F.stddev), ('min', F.min),('max', F.max) ]
expr_cv = [f[1](F.col(c)).alias(f"{f[0]}_{c}") for f in funcs for c in ap_cols_to_keep]
eng_features3_1 = temp31.groupby('PB_KEY').agg(*expr_cv)
print("Audience Profile avgs...")
print(eng_features3_1.columns)

# COMMAND ----------

# DBTITLE 1,Median of Audience Profiles
temp3 = pop_features_raw3.drop(*['PBKEY', 'GEOID', 'AP_CODE'])
table_name = "temp4"
temp3.createOrReplaceTempView(table_name)
comm_template = "percentile_approx(%s, 0.5) AS %s"

comm_list = []
for col_name in ap_cols_to_keep:
    #comm_template = f"percentile_approx({col_name}, 0.5)"
    comm_str = comm_template % (col_name, f"median_{col_name}")
    comm_list.append(comm_str)
    
comm_str = ", ".join(comm_list)
#print(comm_str)
query_template = f"SELECT PB_KEY, %s FROM {table_name} GROUP BY PB_KEY"
query = query_template % (comm_str)
print(query)

eng_features3_2 = spark.sql(query)
print("coffee spend medians...")
print(eng_features3_2.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join the engineered features by GeoFenced Location

# COMMAND ----------

eng_features = (
  eng_features1_1
  .join(eng_features1_2, on = ['PB_KEY'], how = 'left')
  .join(eng_features2_1, on = ['PB_KEY'], how = 'left')
  .join(eng_features2_2, on = ['PB_KEY'], how = 'left')
  .join(census_block_cnt, on = ['PB_KEY'], how = 'left')
  .join(eng_features3_1, on = ['PB_KEY'], how = 'left')
  .join(eng_features3_2, on = ['PB_KEY'], how = 'left')
)
print(eng_features.columns)

# COMMAND ----------

#sanity check repeated columns 
cnt_cols = pd.DataFrame(eng_features.columns, columns = ["col_name"])
cnt_cols["cnt"] = 1
cnt_cols2 = cnt_cols.groupby(["col_name"])[["cnt"]].sum().reset_index().sort_values(["cnt"], ascending = False)
cnt_cols2.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save to Delta Lake

# COMMAND ----------

print(f"saving residential {len(eng_features.columns)} engineered proximity features")
#spark.sql("DROP TABLE IF EXISTS team30.starbucks_residential_features")
#eng_features.write.format("delta").saveAsTable("workshop.starbucks_features_residential")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from team30.starbucks_residential_features

# COMMAND ----------


