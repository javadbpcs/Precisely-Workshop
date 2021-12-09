# Databricks notebook source
# MAGIC %md
# MAGIC ## Workshop Session 2.0 Train ML model
# MAGIC ### Section 2.1 Create Models, Train, Test, and Evaluate
# MAGIC 
# MAGIC ![](/files/images/plan7.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Recall that our goal to start modeling was feature data for each store that looks like this
# MAGIC 
# MAGIC | starbucks_id | feature1 | feature2 | ... | stars |
# MAGIC |--------------|----------|----------|-----|-------|
# MAGIC | 001||| ... | 3.5 |
# MAGIC | 002||| ... | 4.0 |
# MAGIC | 003||| ... | 4.2 |
# MAGIC | 004||| ... | 2.4 |
# MAGIC | ...|...|...| ... | ... |

# COMMAND ----------

# DBTITLE 1,Imports
from pyspark.ml.regression import DecisionTreeRegressor,GBTRegressor,RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import numpy as np
import matplotlib.pyplot as plt

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# DBTITLE 1,Grab Feature Data
# Available FE datasets to run
# 1. for business locatoins: "workshop.starbucks_features_business"
# 2. for residential locations: "workshop.starbucks_features_residential"
# 3. for residential and business locations: "workshop.starbucks_features"


starbucks_features = spark.sql("select * from workshop.starbucks_features").na.fill(0)

display(starbucks_features)

# COMMAND ----------

# DBTITLE 1,Prepare data for model
# Before Spark can accept the data, it needs to be in the form of two columns
# ("label","features")

# Dropping unnecessary columns
starbucks_features = starbucks_features.drop('review_count', 'PB_KEY')

# Extracting List of Feature Names to Make Assembler
starbucks_features_names = starbucks_features.drop('stars').columns
assembler = VectorAssembler(
  inputCols = starbucks_features_names,
              outputCol="features")

output = assembler.transform(starbucks_features)
final_data = output.select("features",'stars')

# COMMAND ----------

# DBTITLE 1,Splitting to Train and Test Sets
train_data,test_data = final_data.randomSplit([0.7,0.3])

# COMMAND ----------

# DBTITLE 1,Training Three Models
# Use mostly defaults for each model to make this comparison "fair"
dtr = DecisionTreeRegressor(labelCol='stars',featuresCol='features')
rfr = RandomForestRegressor(labelCol='stars',featuresCol='features')
gbr = GBTRegressor(labelCol='stars',featuresCol='features')

# Train the models for review rate
dtr_model = dtr.fit(train_data)
rfr_model = rfr.fit(train_data)
gbr_model = gbr.fit(train_data)

# COMMAND ----------

# DBTITLE 1,Making Predictions 
dtr_predictions = dtr_model.transform(test_data)
rfr_predictions = rfr_model.transform(test_data)
gbr_predictions = gbr_model.transform(test_data)

# COMMAND ----------

# DBTITLE 1,Evaluation of Models
rmse_evaluator = RegressionEvaluator(labelCol="stars", predictionCol="prediction", metricName="rmse")


# COMMAND ----------

dtr_rmse = rmse_evaluator.evaluate(dtr_predictions)
rfr_rmse = rmse_evaluator.evaluate(rfr_predictions)
gbr_rmse = rmse_evaluator.evaluate(gbr_predictions)

# COMMAND ----------

# DBTITLE 0,Comparison of Models
# MAGIC %md
# MAGIC # Comparing model results

# COMMAND ----------

# DBTITLE 1,RMSE
print("Here are the results!")
print('-'*80)
print('A single decision tree had a rmse of: {0:2.2f}'.format(dtr_rmse))
print('-'*80)
print('A random forest ensemble had a rmse  of: {0:2.2f}'.format(rfr_rmse))
print('-'*80)
print('A ensemble using GBT had a rmse  of: {0:2.2f}'.format(gbr_rmse))
print('-'*80)
# review_rate_mean = starbucks_features.select(avg("stars")).show()

# COMMAND ----------

# DBTITLE 1,MAPE
# MAPE
dtr_predictions.registerTempTable("dtr_predictions")
dtr_mape = spark.sql('select 100*avg(abs(stars - prediction)/stars) as MAPE_dtr from dtr_predictions')
dtr_mape.show()

rfr_predictions.registerTempTable("rfr_predictions")
rfr_mape = spark.sql('select 100*avg(abs(stars - prediction)/stars) as MAPE_rfr from rfr_predictions')
rfr_mape.show()

gbr_predictions.registerTempTable("gbr_predictions")
gbr_mape = spark.sql('select 100*avg(abs(stars - prediction)/stars) as MAPE_gbr from gbr_predictions')
gbr_mape.show()


# COMMAND ----------

dtr_mape = dtr_mape.toPandas()
rfr_mape = rfr_mape.toPandas()
gbr_mape = gbr_mape.toPandas()

rmse = [dtr_rmse, rfr_rmse, gbr_rmse]
mape = [dtr_mape.values[0][0],
        rfr_mape.values[0][0],
        gbr_mape.values[0][0]]

# COMMAND ----------

# DBTITLE 1,Plotting Comparisons
labels = ['Decision Tree', 'Random Forest', 'Gradient-Boosted Trees']
X = np.arange(len(labels)) 

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# Add some text for labels,x-axis tick labels
ax.set_xlabel('ML Model', fontsize = 15)
ax.set_xticks(X)
ax.set_xticklabels(labels)
ax.xaxis.set_label_coords(0.5, -0.15)

# Create another axes that shares the same x-axis as ax.
ax2 = ax.twinx() 

# bar plot
ax.bar(X + 0.00, rmse, color = 'r', label = 'rmse', width = 0.25)
ax2.bar(X + 0.3, mape, color = 'b', label = 'mape', width = 0.25)

# add y label
ax.set_ylabel('RMSE')
ax2.set_ylabel('MAPE')

# Add legend
ax.legend(loc='upper right')
ax2.legend(loc='upper left')

# Setting limits for the two y axis
ax.set_ylim(0, 1);
ax2.set_ylim(0, 22);
# plt.ylim([0,1])

# Adding title
plt.title('Performance Comparison of the Three Models')

# Display plot
plt.show()

# COMMAND ----------


