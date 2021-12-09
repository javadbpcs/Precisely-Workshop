# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Session 1.0
# MAGIC 
# MAGIC 
# MAGIC ![](files/images/problem_statement-2.png)
# MAGIC 
# MAGIC ![](/files/images/plan1a.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC At a high level, to train our model we want data like this:
# MAGIC 
# MAGIC | starbucks_id | feature1 | feature2 | ... | stars |
# MAGIC |--------------|----------|----------|-----|-------|
# MAGIC | 001||| ... | 3.5 |
# MAGIC | 002||| ... | 4.0 |
# MAGIC | 003||| ... | 4.2 |
# MAGIC | 004||| ... | 2.4 |
# MAGIC | ...|...|...| ... | ... |
# MAGIC 
# MAGIC We use these features in the above table to describe the starbucks location in various ways like population, number of competitors nearby, etc.
# MAGIC 
# MAGIC We then train a model to predict `stars` (or in a more realistic setting `revenue`) based on the descriptive features associated with each location.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### To build a table with these types of features, we will use Location Intelligence and Data Enrichment for each starbucks location.
# MAGIC 
# MAGIC We start with Location Intelligence for each starbucks location with the Geo-* process to find all nearby locations.
# MAGIC 
# MAGIC <br/><br/>
# MAGIC 
# MAGIC <div style="display: flex;">
# MAGIC <div style="flex: 33.33%; padding=0.5px; text-align: center;">
# MAGIC   <img src="/files/images/geocode.png" style="width: 50%; height: auto; border: 2px solid black;">
# MAGIC   <p><ins> Geo-Code </ins></p>
# MAGIC   <p> given an address return the unique PBKEY and lat, long </p>
# MAGIC </div>
# MAGIC <div style="flex: 33.33%; padding=0.5px; text-align: center;">
# MAGIC   <img src="/files/images/georoute.png" style="width: 50%; height: auto; border: 2px solid black;">
# MAGIC   <p><ins> Geo-Route </ins></p>
# MAGIC   <p> given a PBKEY, return a polygon representing a 5 minute drive time from that location </p>
# MAGIC </div>
# MAGIC <div style="flex: 33.33%; padding=0.5px; text-align: center;">
# MAGIC   <img src="/files/images/geofence.png" style="width: 50%; height: auto; border: 2px solid black;">
# MAGIC   <p><ins> Geo-Fence </ins></p>
# MAGIC   <p> given a polygon, return all PBKEYs inside </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC 
# MAGIC <p> </p>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC Next, we can further enrich EACH of these nearby locations with tons of Precisely data sets available such as household income shown below.
# MAGIC 
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/enrich_join.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC We can then collapse the Geo-Fenced data at each starbucks location (thousands of rows per starbucks) with aggregations like `mean, min, max` to transform them into one feature row for each location as described above.
# MAGIC 
# MAGIC 
# MAGIC <div style="text-align: center;">
# MAGIC <img src="/files/images/collapse_features.png" style="width: 50%; height: auto;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC At this point we have built a dataset with features which describe the proximity around each point, greatly enhancing our understanding of each starbucks location with additional relevant information.
# MAGIC 
# MAGIC <div style="text-align:center;">
# MAGIC <img src="/files/images/visual_features.png" style="width:70%; height:auto;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Modeling Time
# MAGIC 
# MAGIC Once we have the features built we can start some simple POC modeling.
# MAGIC 
# MAGIC For the purpose of the 2 week hackathon-style challenge, we left a lot on the table here. Real implementations would spend more than 1-2 days training models. But since we are predicting star ratings as a substitute for more valuable metrics like revenue, we chose to keep the non-transferrable effort low.

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

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Almost Done!
# MAGIC 
# MAGIC From here the code is mostly written, we just need to re-use a few pieces of it.
# MAGIC 
# MAGIC 1. transform/enrich the real estate candidate data (with the same code that transforms/enriches Starbucks locations)
# MAGIC 2. load the best ML model and use it to predict the stars of each candidate location
# MAGIC 
# MAGIC 
# MAGIC The result we get is simply a ranked list of all properties we considered purchasing
# MAGIC <br/>
# MAGIC <div style="text-align:center">
# MAGIC <img src="/files/images/rankings.png" style="width:30%; height:auto;">
# MAGIC </div>
