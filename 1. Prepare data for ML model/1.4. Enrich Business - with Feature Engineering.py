# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Session 1.0 Prepare data for ML model
# MAGIC 
# MAGIC ### Section 1.4 Enrich geofenced business and residential locations differently
# MAGIC 
# MAGIC ![](/files/images/plan5.png)
# MAGIC 
# MAGIC #### 1.4.1 Enrich Business data
# MAGIC 
# MAGIC In this notebook, we will enrich the 1.5M+ addresses resulting from the geofence with relevant business information available in the Precisely datasets.
# MAGIC 
# MAGIC For example:
# MAGIC 
# MAGIC * Number of Related Businesses
# MAGIC * Number of Employees of Related Businesses
# MAGIC * Number of all the businesses in the neighborhood (any types)
# MAGIC * Number of Nearby Starbucks
# MAGIC * Average Sales Volume of related nearby businesses 
# MAGIC 
# MAGIC #### 1.4.2 Aggregate enriched Geofenced locations into “proximity summary features”
# MAGIC 
# MAGIC * To create a **Precisely data** enriched dataset of proximity summary features for each of our **600 Starbucks locations**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4.1 Enrich Business data

# COMMAND ----------

# DBTITLE 1,Joining with POI and Address Fabric

from pyspark.sql.functions import col, asc

# yelp_reviews for all starbucks stores with geofence info
all_starbucks_geofence = spark.sql('select * from workshop.geofenced_starbucks')
# Depending on the type of locations you are looking for around your location of interest, you can filter them.
# This can happen by using the column named "PROP_TYPE" ==> B: Business, R: Residential

# USA Address Fabric
US_AF_addresses =  spark.sql(f"select * from gv_usa_address_fabric_001")

# USA POI
POI_USA = spark.sql(f"select * from usa_points_of_insterest_001")

# joining geofence table with US_AF_addresses ==> this gives the location type
all_starbucks_geofence_join_US_AF = all_starbucks_geofence.join(US_AF_addresses, on = ["PBKEY"], how = 'inner')

# Just Business locatoins 
all_starbucks_geofence_join_US_AF_B_only = all_starbucks_geofence_join_US_AF.select('*').where(col('PROP_TYPE')=='B')

# # Just Residential locatoins 
# all_starbucks_geofence_join_US_AF_R_only = all_starbucks_geofence_join_US_AF.select('*').where(col('PROP_TYPE')=='R')

# joining with POI_USA
all_starbucks_geofence_join_US_AF_B_only_POI = all_starbucks_geofence_join_US_AF_B_only.join(POI_USA, on = ["PBKEY"], how = 'inner')
display(all_starbucks_geofence_join_US_AF_B_only_POI)


# COMMAND ----------

# DBTITLE 1,Filter to Starbucks data
# choose the type/category of businesses you are looking for from POI table
POI_USA_BUS_CAT = POI_USA.select('BRANDNAME', 'BUSINESS_LINE','SIC8_DESCRIPTION','GROUP_NAME', 'MAIN_CLASS', 'SUB_CLASS').distinct()
#display(POI_USA_BUS_CAT)

# Finding all the Starbucks in the US
POI_USA_BUS_STARBUCKS= POI_USA.select('*').where(col('BRANDNAME') == 'STARBUCKS').distinct()
#display(POI_USA_BUS_STARBUCKS)
#print(f"{POI_USA_BUS_STARBUCKS.count()} Starbucks")

#Finding Starbucks in geofence data
all_starbucks_geofence_join_US_AF_B_only_STARBUCKS = all_starbucks_geofence_join_US_AF_B_only.join(POI_USA_BUS_STARBUCKS, on = ["PBKEY"], how = 'inner')
display(all_starbucks_geofence_join_US_AF_B_only_STARBUCKS)
# all_starbucks_geofence_join_US_AF_B_only_STARBUCKS.count()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1.4.2 Aggregate enriched Geofenced locations into “proximity summary features”
# MAGIC 
# MAGIC <br/>
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/collapse_features.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Find related businesses
# Note: you have to look at the POI_USA_BUS_CAT table to find what column and value fit best for your needs
# Here we are looking for "EATING AND DRINKING PLACES"

from pyspark.sql.functions import col, asc
all_starbucks_geofence_join_US_AF_B_only_POI_filtered = all_starbucks_geofence_join_US_AF_B_only_POI.select('*').where(col('GROUP_NAME')=='EATING AND DRINKING PLACES')

# COMMAND ----------

# DBTITLE 1,Feature 1: Number of related businesses including Starbucks
all_starbucks_geofence_join_US_AF_B_only_POI_filtered.registerTempTable('all_starbucks_geofence_join_US_AF_B_only_POI_filtered')

number_of_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, count(PBKEY) as related_businesses_count from  all_starbucks_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")

display(number_of_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 2: Average number of employees of related businesses including Starbucks
number_of_employeeshere_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, avg(EMPLOYEE_HERE) as avg_EMPLOYEE_HERE from  all_starbucks_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")

display(number_of_employeeshere_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 3: Average number of employees count of related businesses including Starbucks
number_of_employeescount_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, avg(EMPLOYEE_COUNT) as avg_EMPLOYEE_COUNT from  all_starbucks_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")

display(number_of_employeescount_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 4: Number of all the businesses (any types)
all_starbucks_geofence_join_US_AF_B_only_POI.registerTempTable('all_starbucks_geofence_join_US_AF_B_only_POI')

number_of_all_businesses  = spark.sql(f"select PB_KEY, count(PBKEY) as all_businesses_count from  all_starbucks_geofence_join_US_AF_B_only_POI group by PB_KEY")

display(number_of_all_businesses)

# COMMAND ----------

# DBTITLE 1,Feature 5: Number of all nearby Starbucks
all_starbucks_geofence_join_US_AF_B_only_STARBUCKS.registerTempTable('all_starbucks_geofence_join_US_AF_B_only_STARBUCKS')

number_of_all_starbucks  = spark.sql(f"select PB_KEY, count(PBKEY) as starbucks_count from  all_starbucks_geofence_join_US_AF_B_only_STARBUCKS group by PB_KEY")

display(number_of_all_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 6: Average sales volume of related nearby businesses 
avg_sales_vol_USdollars_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, avg(SALES_VOLUME_US_DOLLARS) as avg_SALES_VOLUME_US_DOLLARS from  all_starbucks_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")

display(avg_sales_vol_USdollars_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Join Engineered Features
f1 = number_of_related_businesses_including_starbucks
f2 = number_of_employeeshere_related_businesses_including_starbucks
f3 = number_of_employeescount_related_businesses_including_starbucks
f4 = number_of_all_businesses
f5 = number_of_all_starbucks
f6 = avg_sales_vol_USdollars_related_businesses_including_starbucks

all_features = (
  f1
  .join(f2, on = ["PB_KEY"], how = 'left')
  .join(f3, on = ["PB_KEY"], how = 'left')
  .join(f4, on = ["PB_KEY"], how = 'left')
  .join(f5, on = ["PB_KEY"], how = 'left')
  .join(f6, on = ["PB_KEY"], how = 'left')
)

#add labels - this step is optional (can help explore business feature space before joining with residential data, and/or create separate models for business only)
online_reviews = sqlContext.sql('select * from provided_datasets.yelp_reviews_starbucks')
all_starbucks_online_reviews = online_reviews.select('PB_KEY', 'review_count', 'stars').distinct()

#final join (with labels)
all_starbucks_features_df = all_features.join(all_starbucks_online_reviews, on = ["PB_KEY"], how = 'inner')

#display(all_starbucks_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving to Delta Lake

# COMMAND ----------

# replacing null values with zeros 
all_starbucks_features_df = all_starbucks_features_df.na.fill(0)

# removing duplications on PB_KEY
all_starbucks_features_df = all_starbucks_features_df.dropDuplicates(["PB_KEY"])

# writting to Delta table
print(f"saving {len(all_starbucks_features_df.columns)} business features")
#spark.sql("DROP TABLE IF EXISTS workshop.starbucks_features_business")
#all_starbucks_features_answers_df.write.format("delta").saveAsTable("workshop.starbucks_features_business")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from workshop.starbucks_features_business

# COMMAND ----------


