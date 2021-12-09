# Databricks notebook source
# MAGIC %md
# MAGIC # Welcome to the Hackathon Workshop!
# MAGIC ![](https://bpcs.com/wp-content/uploads/2021/08/Hackathon-4-FeatMast.jpg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Starbucks Locations
# MAGIC 
# MAGIC For this workshop we will be using a list of 600 Starbucks locations obtained from the X Challenge Business Reviews Dataset.
# MAGIC 
# MAGIC Each of these 600 locations in the X Challenge data set includes customer reviews and star ratings (0-5).
# MAGIC 
# MAGIC | Starbucks Location | Average Rating |
# MAGIC |--------------------|----------------|
# MAGIC | 123 Main St        |       3.5      |
# MAGIC | 100 South Broad    |       4.5      |
# MAGIC |    ...             |       ...      |

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/images/plan2a.png)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,View the X Challenge Starbucks Data set
# MAGIC %scala
# MAGIC //
# MAGIC // Example of Geocoding a sample of X challenge Starbucks locations
# MAGIC //
# MAGIC 
# MAGIC val input_example1 = spark.sql("""
# MAGIC SELECT
# MAGIC   business_id,
# MAGIC   name,
# MAGIC   address,
# MAGIC   city,
# MAGIC   postal_code,
# MAGIC   state,
# MAGIC   'USA' as country,
# MAGIC   stars
# MAGIC FROM 
# MAGIC   provided_datasets.x_challenge_business_enhanced
# MAGIC WHERE 
# MAGIC   name like '%Starbucks%' and PB_KEY is not NULL
# MAGIC """)
# MAGIC val input_sample1 = input_example1.sample(.05)
# MAGIC 
# MAGIC display(input_sample1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Geocode the Starbucks Locations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Pre-requisite: Install Precisely Geocoding Libraries
# MAGIC 
# MAGIC ![](/files/images/precisely1.png)
# MAGIC 
# MAGIC This is a pre-requisite for joining your data to Precisely enriched datasets.
# MAGIC 
# MAGIC Detailed instructions and code are in the basic geocode installation notebook (`Geocoding/0.1 Basic Geocoding Installation and Operation`).
# MAGIC 
# MAGIC All we need to do is run the cell below provided by Precisely!
# MAGIC 
# MAGIC Then we can start using their geocode/georoute/geofence UDFs in scala.

# COMMAND ----------

# DBTITLE 1,Precisely Geocoding Libraries (Basic Geocoding Installation)
# MAGIC %scala
# MAGIC //Step 0.1 (Prep) - Set the Software and Data resource variables
# MAGIC 
# MAGIC // This step of the notebook has been added to make running the cells a bit easier.  You can add these literals in each cell below
# MAGIC // or leverage these global variables to simplify the setup in the cells below.
# MAGIC //
# MAGIC // Most Geocoding Reference Data is updated monthly.  Set this variable to ensure the proper treatment of updates
# MAGIC val DATA_VINTAGE = "2021.7"
# MAGIC 
# MAGIC // Set the shared directories where the SDKs, Resource directories, and Reference Data is located
# MAGIC val GeocodingRootDBFS = "/dbfs/blueprint-hackathon/geocoding"
# MAGIC val ResourcesLocationDBFS = s"$GeocodingRootDBFS/sdk/spectrum-bigdata-geocoding-5.0.0.4/resources/"
# MAGIC val DataLocationDBFS = s"$GeocodingRootDBFS/spd-data/$DATA_VINTAGE/*"
# MAGIC 
# MAGIC // Set the location for the Geocoding Preferences to control behavior of the engine in geocoding jobs
# MAGIC val PreferencesFileDBFS = "/blueprint-hackathon/geocoding/geocodePreferences.xml"
# MAGIC 
# MAGIC // Set the local directories where the Resource directories and Preferences are located
# MAGIC val ResourcesLocationLocal = s"$ResourcesLocationDBFS"
# MAGIC val PreferencesFileLocal = s"$PreferencesFileDBFS"
# MAGIC 
# MAGIC // Set the local directories where the geocoding Reference Data is located
# MAGIC val DataLocationLocal = List("/dbfs/blueprint-hackathon/geocoding/spd-data/2021.7/KLD072021.spd", "/dbfs/blueprint-hackathon/geocoding/spd-data/2021.7/KNT072021.spd")
# MAGIC val ExtractLocationLocal = "/precisely/data"
# MAGIC 
# MAGIC // Set the output fields that the Geocoder will append to the input records as they are processed 
# MAGIC val outputFields = List("formattedStreetAddress", "formattedLocationAddress", "X", "Y", "precisionCode", "PB_KEY")
# MAGIC 
# MAGIC //Step 0.2 (Prep) - Register the Geocoding & Address Validation Function
# MAGIC 
# MAGIC import com.pb.bigdata.geocoding.spark.api.GeocodeUDFBuilder
# MAGIC import org.apache.spark.sql.functions._
# MAGIC GeocodeUDFBuilder.singleCandidateUDFBuilder()
# MAGIC       .withResourcesLocation(ResourcesLocationLocal)
# MAGIC       .withDataLocations(DataLocationLocal:_*)
# MAGIC       .withExtractionLocation(ExtractLocationLocal)
# MAGIC       .withOutputFields(outputFields:_*)
# MAGIC       .withPreferencesFile("/dbfs"+PreferencesFileLocal)
# MAGIC       .withErrorField("error")
# MAGIC 	  .register("geocode", spark);
# MAGIC 
# MAGIC // Step 0.3 (Prep) - Register the Travel Boundary Generation Function (Routing)
# MAGIC 
# MAGIC import com.pb.routing.gra.boundary.GetTravelBoundaryRequest
# MAGIC import com.pb.bigdata.routing.spark.api.GRAInstanceBuilder
# MAGIC import com.mapinfo.midev.geometry.impl.Point
# MAGIC import com.mapinfo.midev.coordsys.CoordSysConstants
# MAGIC import com.mapinfo.midev.geometry.{DirectPosition, IFeatureGeometry, SpatialInfo}
# MAGIC import com.mapinfo.midev.unit.TimeUnit
# MAGIC import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import com.mapinfo.midev.persistence.json.GeoJSONUtilities
# MAGIC import com.mapinfo.midev.unit.{Length, LinearUnit, TimeUnit, VelocityUnit}
# MAGIC import org.apache.spark.sql.expressions.UserDefinedFunction
# MAGIC import com.pb.routing.enums.HistoricSpeedBucket
# MAGIC import com.pb.routing.gra.route.GetRouteRequest
# MAGIC 
# MAGIC object Holder extends java.io.Serializable {
# MAGIC 
# MAGIC   //lazy transient
# MAGIC  @transient lazy val graInstance = GRAInstanceBuilder()
# MAGIC   // .withDownloadManager(downloadManager)
# MAGIC   .addDataset("/dbfs/blueprint-hackathon/routing/RoutingReference/")
# MAGIC   .build()
# MAGIC 
# MAGIC val PTPUDF = udf((x1: Double, y1: Double, x2: Double, y2: Double) => {
# MAGIC   val startTimeMillis = System.currentTimeMillis()
# MAGIC   
# MAGIC   val strpoint = new Point(SpatialInfo.create(CoordSysConstants.longLatWGS84), new DirectPosition(x1, y1))
# MAGIC     val endpoint = new Point(SpatialInfo.create(CoordSysConstants.longLatWGS84), new DirectPosition(x2, y2))
# MAGIC 
# MAGIC     val request = new GetRouteRequest.Builder(strpoint, endpoint)
# MAGIC       .distanceUnit(LinearUnit.MI)
# MAGIC       .timeUnit(TimeUnit.MINUTE)
# MAGIC       .majorRoads(false)
# MAGIC       .historicSpeedBucket(HistoricSpeedBucket.NONE)
# MAGIC       .build()
# MAGIC 
# MAGIC     val response = graInstance.getRoute(request)
# MAGIC     val endTimeMillis = System.currentTimeMillis()
# MAGIC     val result1 = List(response.getDistance.getValue, response.getTime.getValue,
# MAGIC       (endTimeMillis - startTimeMillis))
# MAGIC     result1
# MAGIC })
# MAGIC   
# MAGIC val TravelBoundaryUDF = udf((x: Double, y: Double) => {
# MAGIC   val startTimeMillis = System.currentTimeMillis()
# MAGIC   
# MAGIC   val point = new Point(SpatialInfo.create(CoordSysConstants.longLatWGS84), new DirectPosition(x, y))
# MAGIC   
# MAGIC   val request = new GetTravelBoundaryRequest.Builder(point, Array(5.0), TimeUnit.MINUTE)
# MAGIC         .defaultAmbientSpeed(2.5)
# MAGIC         .majorRoads(false)
# MAGIC         .ambientSpeedUnit(VelocityUnit.MPH)
# MAGIC         .maxOffRoadDistance(0.1)
# MAGIC         .maxOffRoadDistanceUnit(LinearUnit.MI)
# MAGIC         .historicSpeedBucket(HistoricSpeedBucket.NONE)
# MAGIC         .build()
# MAGIC         val response = graInstance.getTravelBoundary(request)
# MAGIC         val largeIsoGeom = response.getTravelBoundaries.get(0).getGeometry
# MAGIC          GeoJSONUtilities.toGeoJSON(largeIsoGeom.asInstanceOf[IFeatureGeometry])
# MAGIC 
# MAGIC })  
# MAGIC }
# MAGIC 
# MAGIC //Step 0.4 (Prep) - Register the Geometry & Polygon Functions (Location Intelligence)
# MAGIC 
# MAGIC import org.apache.spark.SparkConf
# MAGIC import com.pb.bigdata.li.spark.api.SpatialImplicits._
# MAGIC import org.apache.spark.sql._
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
# MAGIC import com.mapinfo.midev.geometry.{DirectPosition, IFeatureGeometry, SpatialInfo}
# MAGIC import com.pb.bigdata.li.spark.api.table.TableBuilder
# MAGIC import com.mapinfo.midev.language.filter.{FilterSearch, GeometryFilter, GeometryOperator, SelectList}
# MAGIC import com.mapinfo.midev.persistence.json.GeoJSONUtilities
# MAGIC import spark.sqlContext.implicits._
# MAGIC import org.apache.spark.sql.types._
# MAGIC 
# MAGIC spark.sql("set spark.sql.legacy.allowUntypedScalaUDF = true")
# MAGIC 
# MAGIC def contains = (geoJsonPolygon: String) => {
# MAGIC   val table = TableBuilder.NativeTable("/dbfs/blueprint-hackathon/AddressFabric/", "USA_AddressFabric.tab").build()
# MAGIC   val selectList = new SelectList(List[String]("PBKEY"): _*)
# MAGIC 
# MAGIC   val polygon: IFeatureGeometry = GeoJSONUtilities.fromGeoJSON(geoJsonPolygon)
# MAGIC   
# MAGIC   // create the point-in-polygon filter
# MAGIC   val filter = new GeometryFilter("Obj", GeometryOperator.INTERSECTS, polygon)
# MAGIC   
# MAGIC   // create a search for all the specified columns and the specified filter
# MAGIC   val filterSearch = new FilterSearch(selectList, filter, null)
# MAGIC 
# MAGIC   table.search(filterSearch)
# MAGIC }

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # A few definitions 
# MAGIC 
# MAGIC Running the above cell gives us access to lots of cool tools built by Precisely. We will focus our use on three specific tools (UDFs):
# MAGIC 
# MAGIC 
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

# DBTITLE 1,Geocoding a small sample of X challenge Starbucks locations
# MAGIC %scala
# MAGIC 
# MAGIC // Validate, Geocode, and Add PreciselyID to Data
# MAGIC 
# MAGIC val input_df = input_sample1
# MAGIC // If you are running this for a second time you will have to delete the table
# MAGIC spark.sql("DROP TABLE IF EXISTS team30.geocoded_example_1")
# MAGIC //
# MAGIC // Standardize, validate, geocode the input addresses
# MAGIC val output_df = input_df.withColumn("geocode_result",
# MAGIC        callUDF("geocode",
# MAGIC           map(
# MAGIC           lit("mainAddressLine"), $"address",
# MAGIC           lit("areaName3"), $"city",
# MAGIC           lit("areaName1"), $"state",
# MAGIC           lit("postCode1"), $"postal_code",
# MAGIC           lit("country"), lit("USA")
# MAGIC         )
# MAGIC       )
# MAGIC     ).persist()   // to prevent re-calculation
# MAGIC     .select("*", "geocode_result.*").drop("geocode_result").filter($"PB_KEY".isNotNull)
# MAGIC 
# MAGIC 
# MAGIC // Create table in the metastore using DataFrame's schema and write data to it    
# MAGIC // output_df.write.format("delta").saveAsTable("team30.geocoded_example_1")
# MAGIC 
# MAGIC display(output_df)
# MAGIC 
# MAGIC ""
# MAGIC // For more information, see the input and output options for the Spectrum Geocoding for Big Data User Guide
# MAGIC // https://docs.precisely.com/docs/sftw/hadoop/landingpage/docs/geocoding/webhelp/index.html#Geocoding/source/geocoding/options_title.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Geo-Route & Geo-Fence all X challenge Starbucks locations

# COMMAND ----------

# DBTITLE 1,Load pre-geocoded list of locations 
# MAGIC %scala
# MAGIC // Select initial list of locations for training data 
# MAGIC // for this example: this would be the list of 600+ Starbucks locations
# MAGIC 
# MAGIC val geocoded_locations = spark.sql("""
# MAGIC SELECT
# MAGIC   business_id, name,
# MAGIC   PB_KEY, X, Y,
# MAGIC   formattedLocationAddress, formattedStreetAddress,
# MAGIC   city, state, postal_code,
# MAGIC   review_count, stars
# MAGIC FROM 
# MAGIC   provided_datasets.x_challenge_business_enhanced
# MAGIC WHERE 
# MAGIC   name like '%Starbucks%' and PB_KEY is not NULL
# MAGIC """)
# MAGIC //geocoded_locations.write.format("delta").saveAsTable("team30.geocoded_locations_001")
# MAGIC display(geocoded_locations)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Geo-routing and Geo-fencing the Starbucks locations
# MAGIC 
# MAGIC 
# MAGIC 1. Geo-Route - create a drive time polygon around each location
# MAGIC 2. Geo-Fence - create a list of all PBKEYs inside that polygon
# MAGIC 
# MAGIC We use a simple checkpointing batch pattern to ensure safe progress on long runs
# MAGIC 
# MAGIC <img style="height:1.75em; top:0.05em; transform:rotate(15deg)" src="/files/images/icon_note.webp"/>
# MAGIC Notice how simple the Precisely SDK is for these operations

# COMMAND ----------

# DBTITLE 1,Some helper functions to organize usage of Precisely's easy UDFs
# MAGIC %scala
# MAGIC 
# MAGIC def next_batch(next_bid: Int, source_table: String) : DataFrame = {
# MAGIC   // return the requested batch from the source table
# MAGIC   val next_batch = spark.table(source_table).filter(f"batch_id = $next_bid")
# MAGIC   return next_batch
# MAGIC }
# MAGIC 
# MAGIC def travel_boundary_by_batch(geocoded_input:DataFrame) : DataFrame = {
# MAGIC   // create a 5 min drive time travel boundary given a geocoded dataframe
# MAGIC   val polygonResult = geocoded_input
# MAGIC           .withColumn(
# MAGIC             "polygon",
# MAGIC             Holder.TravelBoundaryUDF(
# MAGIC               col("X").cast(DoubleType),
# MAGIC               col("Y").cast(DoubleType)
# MAGIC             )
# MAGIC           )
# MAGIC   return polygonResult
# MAGIC }
# MAGIC 
# MAGIC def geofence_by_batch(travel_boundary: DataFrame) : DataFrame = {
# MAGIC   // find all locations inside each polygon
# MAGIC   val liResult = travel_boundary.filter("polygon is not null")
# MAGIC                  .withSpatialSearchColumns(
# MAGIC                    Seq(col("polygon").cast(StringType)),
# MAGIC                    contains,
# MAGIC                    includeEmptySearchResults = false
# MAGIC                  )
# MAGIC                  .drop(col("Obj")).drop("MI_Style")
# MAGIC   return liResult
# MAGIC }
# MAGIC 
# MAGIC def append_to_table(data:DataFrame, output_table_name:String) {
# MAGIC   // append this batch of results to delta table
# MAGIC   if (data.rdd.isEmpty){
# MAGIC     println("append data frame is empty!")
# MAGIC   } else {
# MAGIC     if (spark.catalog.tableExists(output_table_name)) {
# MAGIC       // append
# MAGIC       data.write.format("delta")
# MAGIC           .mode("append")
# MAGIC           .option("mergeSchema",true)
# MAGIC           .saveAsTable(output_table_name)
# MAGIC 
# MAGIC     } else {
# MAGIC       // create
# MAGIC       data.write.format("delta")
# MAGIC           .option("overwriteSchema","true")
# MAGIC           .mode("overwrite")
# MAGIC           .saveAsTable(output_table_name)
# MAGIC     }
# MAGIC   }
# MAGIC }

# COMMAND ----------

# DBTITLE 1,Batch process all locations, writing batch results to delta as we go
# MAGIC %scala 
# MAGIC 
# MAGIC val geocode_table_name = "team30.geocoded_locations_w_batch"
# MAGIC val georoute_table_name = "team30.georouted_locations_w_batch"
# MAGIC val geofence_table_name = "team30.geofenced_locations_w_batch"
# MAGIC 
# MAGIC // clean up previous run
# MAGIC //spark.sql(f"DROP TABLE IF EXISTS $georoute_table_name")
# MAGIC //spark.sql(f"DROP TABLE IF EXISTS $geofence_table_name")
# MAGIC 
# MAGIC 
# MAGIC val nbatch = 25
# MAGIC println(f"total number of batches to process: $nbatch%d ")
# MAGIC 
# MAGIC 
# MAGIC // loop to persist periodically
# MAGIC for (currentBatch <- 0 to nbatch){
# MAGIC   println("===================================================")
# MAGIC   println(f"processing batch $currentBatch%d of $nbatch%d...")
# MAGIC 
# MAGIC   // get batch df
# MAGIC   val geocoded_batch = next_batch(currentBatch)
# MAGIC   val nrows = geocoded_batch.count()
# MAGIC   println(f"loaded $nrows%d rows")
# MAGIC 
# MAGIC   // get boundary polygon
# MAGIC   val geo_routed_batch = travel_boundary_by_batch(geocoded_batch)
# MAGIC   append_to_table(geo_routed_batch, georoute_table_name)
# MAGIC   println(f"travel boundary polygon $currentBatch%d done")
# MAGIC 
# MAGIC   // get geofence data
# MAGIC   val geo_fenced_batch = geofence_by_batch(geo_routed_batch)
# MAGIC   println(f"geoFence $currentBatch%d done")
# MAGIC 
# MAGIC   // write batch results
# MAGIC   append_to_table(geo_fenced_batch, geofence_table_name)
# MAGIC   println(f"batch number $currentBatch%d succesfully processed")
# MAGIC }

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Lets take a look at the results table of Geo-Fenced locations
# MAGIC 
# MAGIC <img style="height:1.75em; top:0.05em; transform:rotate(15deg)" src="/files/images/icon_note.webp"/>
# MAGIC **PreciselyIDs**
# MAGIC 
# MAGIC Precisely IDs are a unique, persistent index applied to all addresses in the geofence - is also referred to as the "PB_KEY" or "PBKEY" in many files and outputs. The terms are synonymous.
# MAGIC 
# MAGIC For the rest of this notebook:
# MAGIC * "PB_KEY" is the original Starbucks
# MAGIC * "PBKEY" is the result of the geofence (addresses inside the 5min drive polygon)

# COMMAND ----------

# DBTITLE 0,Lets see the results table of geofenced locations
# MAGIC %sql
# MAGIC select * from workshop.geofenced_starbucks

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # One Example of a Stabucks and its Polygon and Geofence Locations
# MAGIC 
# MAGIC ![](/files/images/Geofenced_Locaions_around_one_starbucks.png)

# COMMAND ----------

# DBTITLE 1,How many locations does each Starbucks have in its geofence?
# MAGIC %sql
# MAGIC 
# MAGIC select
# MAGIC   PB_KEY as starbucks,
# MAGIC   count(1) as locs_in_fence
# MAGIC from workshop.geofenced_starbucks
# MAGIC   group by PB_KEY
# MAGIC   order by locs_in_fence

# COMMAND ----------

# DBTITLE 1,Geofence Stats
# MAGIC %md
# MAGIC * starting from **600+** gecododed Starbucks
# MAGIC * We get a total of **10 Million+** records as a result of the geofence
# MAGIC * In the geocoded dataset:
# MAGIC   * **X** and **Y** columns are Precisely high precision *Latitude* and *Longitude* 
# MAGIC * In the geofenced dataset:
# MAGIC   * **PB_KEY** is the original Starbucks Location
# MAGIC   * **PBKEY** is the geofenced address (addresses inside 5 minute distance drive poygon)

# COMMAND ----------

# DBTITLE 1,Geo-coded Starbucks 
# MAGIC %sql
# MAGIC select * from workshop.geocoded_starbucks

# COMMAND ----------

# DBTITLE 1,Geofenced Starbucks
# MAGIC %sql
# MAGIC select * from workshop.geofenced_starbucks

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from workshop.geofenced_starbucks

# COMMAND ----------


