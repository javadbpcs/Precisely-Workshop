# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Session 1.0 Prepare data for ML model
# MAGIC 
# MAGIC ### Section 1.2 Split Geofenced data into Business and Residential

# COMMAND ----------

# MAGIC %md
# MAGIC ![](files/images/plan3a.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Before we start building ML features...
# MAGIC 
# MAGIC Our data has a natural separation on the geofenced locations: some are businesses and some are residential. By splitting this geofenced data set according to property type we not only normalize our features a bit, but we also allow multiple data scientists to build enrichments/aggregations in parallel. 
# MAGIC 
# MAGIC This step in our process is very valuable for making the best use of our team members and also allows us to focus a bit more on creating features specific to each property type. After all, residential enrichments available from Precisely data sets are very different than business enrichments

# COMMAND ----------

# DBTITLE 1,Enrich geofence data with property types
geofenced = (
  spark.sql("select * from workshop.geofenced_starbucks")
    .join(
      spark.sql("select PBKEY, PROP_TYPE from gv_usa_address_fabric_001"),
      on=['PBKEY'],
      how='left'
    )
)


display(geofenced)


# COMMAND ----------

# DBTITLE 1,Separate into two data sets in delta
spark.sql("DROP TABLE IF EXISTS workshop.geofenced_starbucks_residential")
spark.sql("DROP TABLE IF EXISTS workshop.geofenced_starbucks_business")

geofenced_res = geofenced.filter(col('PROP_TYPE') == 'R')
geofenced_res.write.format("delta").saveAsTable("workshop.geofenced_starbucks_residential")

geofenced_bus = geofenced.filter(col('PROP_TYPE') == 'B')
geofenced_bus = geofenced.write.format("delta").saveAsTable("workshop.geofenced_starbucks_business")

# COMMAND ----------

# DBTITLE 1,Check results
# MAGIC %sql
# MAGIC select 
# MAGIC   (select count(1) from workshop.geofenced_starbucks_residential) as n_res,
# MAGIC   (select count(1) from workshop.geofenced_starbucks_business) as n_bus
# MAGIC   

# COMMAND ----------


