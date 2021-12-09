# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Session 3.0 Ranking Real State Data
# MAGIC 3.2. Feature Engineer Real Estate Data - Enrich Real State Business
# MAGIC 
# MAGIC ![](/files/images/plan11.png)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <img style="height:1.75em; top:0.05em; transform:rotate(15deg)" src="/files/images/icon_note.webp"/>
# MAGIC Note: We are just transforming this new data set for real estate EXACTLY the same way we transformed our starbucks data set.
# MAGIC 
# MAGIC In this hackathon-style notebook we have copy/pasted all the transformations... but in a production environment it would be critical to functionalize the transformation code to be sure the exact same process happens to any data set.

# COMMAND ----------

# DBTITLE 1,Read Geofence data for all Starbucks locations
# yelp_reviews for all starbucks stores with geofence info
real_estate_geofence = spark.sql('select * from workshop.real_estate_geofenced')

# COMMAND ----------

# DBTITLE 1,Loading the secondary table(s)
# USA Address Fabric
US_AF_addresses =  spark.sql(f"select * from gv_usa_address_fabric_001")

# USA POI
POI_USA = spark.sql(f"select * from usa_points_of_insterest_001")

# COMMAND ----------

# DBTITLE 1,Join with USA Address Fabric table
# joining geofence table with US_AF_addresses ==> this gives the location type
real_estate_geofence_join_US_AF = real_estate_geofence.join(US_AF_addresses, on = ["PBKEY"], how = 'inner')

# COMMAND ----------

# DBTITLE 1,Filter based on the location type and do the secondary joins for related tables 
# Depending on the type of locations you are looking for around your location of interest, you can filter them.
# This can happen by using the column named "PROP_TYPE" ==> B: Business, R: Residential

from pyspark.sql.functions import col, asc

# Just Business locatoins 
real_estate_geofence_join_US_AF_B_only = real_estate_geofence_join_US_AF.select('*').where(col('PROP_TYPE')=='B')

# # Just Residential locatoins 
# all_starbucks_geofence_join_US_AF_R_only = all_starbucks_geofence_join_US_AF.select('*').where(col('PROP_TYPE')=='R')

# COMMAND ----------

# DBTITLE 1,For business type locations will join with USA POI
# joining with POI_USA
real_estate_geofence_join_US_AF_B_only_POI = real_estate_geofence_join_US_AF_B_only.join(POI_USA, on = ["PBKEY"], how = 'inner')

# COMMAND ----------

# DBTITLE 1,Filter based on the type of businesses you are looking for
# choose the type/category of businesses you are looking for from POI table
POI_USA_BUS_CAT = POI_USA.select('BRANDNAME', 'BUSINESS_LINE','SIC8_DESCRIPTION','GROUP_NAME', 'MAIN_CLASS', 'SUB_CLASS').distinct()
display(POI_USA_BUS_CAT)

# COMMAND ----------

# DBTITLE 1,Finding all the Starbucks in the US
POI_USA_BUS_STARBUCKS= POI_USA.select('*').where(col('BRANDNAME') == 'STARBUCKS').distinct()
display(POI_USA_BUS_STARBUCKS)
print(POI_USA_BUS_STARBUCKS.count())

# COMMAND ----------

# DBTITLE 1,Finding Starbucks in geofence data
real_estate_geofence_join_US_AF_B_only_STARBUCKS = real_estate_geofence_join_US_AF_B_only.join(POI_USA_BUS_STARBUCKS, on = ["PBKEY"], how = 'inner')
# all_starbucks_geofence_join_US_AF_B_only_STARBUCKS.count()

# COMMAND ----------

# DBTITLE 1,Filtering the POI table with related column & the desired/specific value for that column

# Note: you have to look at the POI_USA_BUS_CAT table to find what column and value fit best for your needs
# Here we are looking for "EATING AND DRINKING PLACES"

from pyspark.sql.functions import col, asc
real_estate_geofence_join_US_AF_B_only_POI_filtered = real_estate_geofence_join_US_AF_B_only_POI.select('*').where(col('GROUP_NAME')=='EATING AND DRINKING PLACES')

# COMMAND ----------

# DBTITLE 1,Feature 1: Number of related businesses including Starbucks
real_estate_geofence_join_US_AF_B_only_POI_filtered.registerTempTable('real_estate_geofence_join_US_AF_B_only_POI_filtered')
number_of_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, count(PBKEY) as related_businesses_count from  real_estate_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")
display(number_of_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 2: Average number of employees of related businesses including Starbucks
number_of_employeeshere_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, avg(EMPLOYEE_HERE) as avg_EMPLOYEE_HERE from  real_estate_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")
display(number_of_employeeshere_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 3: Average number of employees count of related businesses including Starbucks
number_of_employeescount_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, avg(EMPLOYEE_COUNT) as avg_EMPLOYEE_COUNT from  real_estate_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")
display(number_of_employeescount_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 4: Number of all the businesses (any types)
real_estate_geofence_join_US_AF_B_only_POI.registerTempTable('real_estate_geofence_join_US_AF_B_only_POI')
number_of_all_businesses  = spark.sql(f"select PB_KEY, count(PBKEY) as all_businesses_count from  real_estate_geofence_join_US_AF_B_only_POI group by PB_KEY")
display(number_of_all_businesses)

# COMMAND ----------

# DBTITLE 1,Feature 5: Number of all nearby Starbucks
real_estate_geofence_join_US_AF_B_only_STARBUCKS.registerTempTable('real_estate_geofence_join_US_AF_B_only_STARBUCKS')
number_of_all_starbucks  = spark.sql(f"select PB_KEY, count(PBKEY) as starbucks_count from  real_estate_geofence_join_US_AF_B_only_STARBUCKS group by PB_KEY")
display(number_of_all_starbucks)

# COMMAND ----------

# DBTITLE 1,Feature 6: Average sales volume of related nearby businesses 
avg_sales_vol_USdollars_related_businesses_including_starbucks  = spark.sql(f"select PB_KEY, avg(SALES_VOLUME_US_DOLLARS) as avg_SALES_VOLUME_US_DOLLARS from  real_estate_geofence_join_US_AF_B_only_POI_filtered group by PB_KEY")
display(avg_sales_vol_USdollars_related_businesses_including_starbucks)

# COMMAND ----------

# DBTITLE 1,Building Feature Vector
f1 = number_of_related_businesses_including_starbucks
f2 = number_of_employeeshere_related_businesses_including_starbucks
f3 = number_of_employeescount_related_businesses_including_starbucks
f4 = number_of_all_businesses
f5 = number_of_all_starbucks
f6 = avg_sales_vol_USdollars_related_businesses_including_starbucks

real_estate_business_features = (
  f1
  .join(f2, on = ["PB_KEY"], how = 'left')
  .join(f3, on = ["PB_KEY"], how = 'left')
  .join(f4, on = ["PB_KEY"], how = 'left')
  .join(f5, on = ["PB_KEY"], how = 'left')
  .join(f6, on = ["PB_KEY"], how = 'left')
)
display(real_estate_business_features)

# COMMAND ----------

display(real_estate_business_features)

# COMMAND ----------

# DBTITLE 1,Writing to Delta table for ML purpose 
# replacing null values with zeros 
real_estate_business_features = real_estate_business_features.na.fill(0)

# removing duplications on PB_KEY
real_estate_business_features = real_estate_business_features.dropDuplicates(["PB_KEY"])

# writting to Delta table
spark.sql("DROP TABLE IF EXISTS workshop.real_estate_business_features")
real_estate_business_features.write.format("delta").saveAsTable("workshop.real_estate_features_business")

# COMMAND ----------

# replacing null values with zeros 
real_estate_business_features = real_estate_business_features.na.fill(0)

# removing duplications on PB_KEY
real_estate_business_features = real_estate_business_features.dropDuplicates(["PB_KEY"])

# COMMAND ----------


