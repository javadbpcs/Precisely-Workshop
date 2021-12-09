// Databricks notebook source
// MAGIC %md
// MAGIC ## Workshop Session 3.0 Ranking Real Estate Data
// MAGIC ### Section 3.1 Geocode and Geofence Real Estate Data
// MAGIC 
// MAGIC ![](/files/images/plan9.png)

// COMMAND ----------

// DBTITLE 1,Instructions for Generating Geofenced Data
// MAGIC %md-sandbox
// MAGIC 
// MAGIC 
// MAGIC <img style="height:1.75em; top:0.05em; transform:rotate(15deg)" src="/files/images/icon_note.webp"/>
// MAGIC Please note: if you have not yet run notebook 1.1 on this cluster, then you can't use the geocode/georoute/geofence udfs until they are installed.
// MAGIC 
// MAGIC We will continue from here to just re-use the same code in the 1.x notebooks to transform the new dataset.

// COMMAND ----------

// DBTITLE 1,Import/redefine Geo-* UDFs
// MAGIC %scala
// MAGIC //Step 0.1 (Prep) - Set the Software and Data resource variables
// MAGIC 
// MAGIC // This step of the notebook has been added to make running the cells a bit easier.  You can add these literals in each cell below
// MAGIC // or leverage these global variables to simplify the setup in the cells below.
// MAGIC //
// MAGIC // Most Geocoding Reference Data is updated monthly.  Set this variable to ensure the proper treatment of updates
// MAGIC val DATA_VINTAGE = "2021.7"
// MAGIC 
// MAGIC // Set the shared directories where the SDKs, Resource directories, and Reference Data is located
// MAGIC val GeocodingRootDBFS = "/dbfs/blueprint-hackathon/geocoding"
// MAGIC val ResourcesLocationDBFS = s"$GeocodingRootDBFS/sdk/spectrum-bigdata-geocoding-5.0.0.4/resources/"
// MAGIC val DataLocationDBFS = s"$GeocodingRootDBFS/spd-data/$DATA_VINTAGE/*"
// MAGIC 
// MAGIC // Set the location for the Geocoding Preferences to control behavior of the engine in geocoding jobs
// MAGIC val PreferencesFileDBFS = "/blueprint-hackathon/geocoding/geocodePreferences.xml"
// MAGIC 
// MAGIC // Set the local directories where the Resource directories and Preferences are located
// MAGIC val ResourcesLocationLocal = s"$ResourcesLocationDBFS"
// MAGIC val PreferencesFileLocal = s"$PreferencesFileDBFS"
// MAGIC 
// MAGIC // Set the local directories where the geocoding Reference Data is located
// MAGIC val DataLocationLocal = List("/dbfs/blueprint-hackathon/geocoding/spd-data/2021.7/KLD072021.spd", "/dbfs/blueprint-hackathon/geocoding/spd-data/2021.7/KNT072021.spd")
// MAGIC val ExtractLocationLocal = "/precisely/data"
// MAGIC 
// MAGIC // Set the output fields that the Geocoder will append to the input records as they are processed 
// MAGIC val outputFields = List("formattedStreetAddress", "formattedLocationAddress", "X", "Y", "precisionCode", "PB_KEY")
// MAGIC 
// MAGIC //Step 0.2 (Prep) - Register the Geocoding & Address Validation Function
// MAGIC 
// MAGIC import com.pb.bigdata.geocoding.spark.api.GeocodeUDFBuilder
// MAGIC import org.apache.spark.sql.functions._
// MAGIC GeocodeUDFBuilder.singleCandidateUDFBuilder()
// MAGIC       .withResourcesLocation(ResourcesLocationLocal)
// MAGIC       .withDataLocations(DataLocationLocal:_*)
// MAGIC       .withExtractionLocation(ExtractLocationLocal)
// MAGIC       .withOutputFields(outputFields:_*)
// MAGIC       .withPreferencesFile("/dbfs"+PreferencesFileLocal)
// MAGIC       .withErrorField("error")
// MAGIC 	  .register("geocode", spark);
// MAGIC 
// MAGIC // Step 0.3 (Prep) - Register the Travel Boundary Generation Function (Routing)
// MAGIC 
// MAGIC import com.pb.routing.gra.boundary.GetTravelBoundaryRequest
// MAGIC import com.pb.bigdata.routing.spark.api.GRAInstanceBuilder
// MAGIC import com.mapinfo.midev.geometry.impl.Point
// MAGIC import com.mapinfo.midev.coordsys.CoordSysConstants
// MAGIC import com.mapinfo.midev.geometry.{DirectPosition, IFeatureGeometry, SpatialInfo}
// MAGIC import com.mapinfo.midev.unit.TimeUnit
// MAGIC import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import com.mapinfo.midev.persistence.json.GeoJSONUtilities
// MAGIC import com.mapinfo.midev.unit.{Length, LinearUnit, TimeUnit, VelocityUnit}
// MAGIC import org.apache.spark.sql.expressions.UserDefinedFunction
// MAGIC import com.pb.routing.enums.HistoricSpeedBucket
// MAGIC import com.pb.routing.gra.route.GetRouteRequest
// MAGIC 
// MAGIC object Holder extends java.io.Serializable {
// MAGIC 
// MAGIC   //lazy transient
// MAGIC  @transient lazy val graInstance = GRAInstanceBuilder()
// MAGIC   // .withDownloadManager(downloadManager)
// MAGIC   .addDataset("/dbfs/blueprint-hackathon/routing/RoutingReference/")
// MAGIC   .build()
// MAGIC 
// MAGIC val PTPUDF = udf((x1: Double, y1: Double, x2: Double, y2: Double) => {
// MAGIC   val startTimeMillis = System.currentTimeMillis()
// MAGIC   
// MAGIC   val strpoint = new Point(SpatialInfo.create(CoordSysConstants.longLatWGS84), new DirectPosition(x1, y1))
// MAGIC     val endpoint = new Point(SpatialInfo.create(CoordSysConstants.longLatWGS84), new DirectPosition(x2, y2))
// MAGIC 
// MAGIC     val request = new GetRouteRequest.Builder(strpoint, endpoint)
// MAGIC       .distanceUnit(LinearUnit.MI)
// MAGIC       .timeUnit(TimeUnit.MINUTE)
// MAGIC       .majorRoads(false)
// MAGIC       .historicSpeedBucket(HistoricSpeedBucket.NONE)
// MAGIC       .build()
// MAGIC 
// MAGIC     val response = graInstance.getRoute(request)
// MAGIC     val endTimeMillis = System.currentTimeMillis()
// MAGIC     val result1 = List(response.getDistance.getValue, response.getTime.getValue,
// MAGIC       (endTimeMillis - startTimeMillis))
// MAGIC     result1
// MAGIC })
// MAGIC   
// MAGIC val TravelBoundaryUDF = udf((x: Double, y: Double) => {
// MAGIC   val startTimeMillis = System.currentTimeMillis()
// MAGIC   
// MAGIC   val point = new Point(SpatialInfo.create(CoordSysConstants.longLatWGS84), new DirectPosition(x, y))
// MAGIC   
// MAGIC   val request = new GetTravelBoundaryRequest.Builder(point, Array(5.0), TimeUnit.MINUTE)
// MAGIC         .defaultAmbientSpeed(2.5)
// MAGIC         .majorRoads(false)
// MAGIC         .ambientSpeedUnit(VelocityUnit.MPH)
// MAGIC         .maxOffRoadDistance(0.1)
// MAGIC         .maxOffRoadDistanceUnit(LinearUnit.MI)
// MAGIC         .historicSpeedBucket(HistoricSpeedBucket.NONE)
// MAGIC         .build()
// MAGIC         val response = graInstance.getTravelBoundary(request)
// MAGIC         val largeIsoGeom = response.getTravelBoundaries.get(0).getGeometry
// MAGIC          GeoJSONUtilities.toGeoJSON(largeIsoGeom.asInstanceOf[IFeatureGeometry])
// MAGIC 
// MAGIC })  
// MAGIC }
// MAGIC 
// MAGIC //Step 0.4 (Prep) - Register the Geometry & Polygon Functions (Location Intelligence)
// MAGIC 
// MAGIC import org.apache.spark.SparkConf
// MAGIC import com.pb.bigdata.li.spark.api.SpatialImplicits._
// MAGIC import org.apache.spark.sql._
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
// MAGIC import com.mapinfo.midev.geometry.{DirectPosition, IFeatureGeometry, SpatialInfo}
// MAGIC import com.pb.bigdata.li.spark.api.table.TableBuilder
// MAGIC import com.mapinfo.midev.language.filter.{FilterSearch, GeometryFilter, GeometryOperator, SelectList}
// MAGIC import com.mapinfo.midev.persistence.json.GeoJSONUtilities
// MAGIC import spark.sqlContext.implicits._
// MAGIC import org.apache.spark.sql.types._
// MAGIC 
// MAGIC spark.sql("set spark.sql.legacy.allowUntypedScalaUDF = true")
// MAGIC 
// MAGIC def contains = (geoJsonPolygon: String) => {
// MAGIC   val table = TableBuilder.NativeTable("/dbfs/blueprint-hackathon/AddressFabric/", "USA_AddressFabric.tab").build()
// MAGIC   val selectList = new SelectList(List[String]("PBKEY"): _*)
// MAGIC 
// MAGIC   val polygon: IFeatureGeometry = GeoJSONUtilities.fromGeoJSON(geoJsonPolygon)
// MAGIC   
// MAGIC   // create the point-in-polygon filter
// MAGIC   val filter = new GeometryFilter("Obj", GeometryOperator.INTERSECTS, polygon)
// MAGIC   
// MAGIC   // create a search for all the specified columns and the specified filter
// MAGIC   val filterSearch = new FilterSearch(selectList, filter, null)
// MAGIC 
// MAGIC   table.search(filterSearch)
// MAGIC }

// COMMAND ----------

// DBTITLE 1,Step 1 - Upload & View the real estate input file
val input_df = spark.sql("""select * from real_estate_csv""")

display(input_df)

// COMMAND ----------

// DBTITLE 1,Step 2 - Validate, Geocode, and Add PreciselyID to Input File
// A few notes:
// - If you have parsed addresses then define each as detailed in the column headers of the file
// - If you have a single line address, just use the "mainAddressLine" for the entire line (the geocoder will parse it)  
// - The country is alway required (for the hackathon only US reference data is installed)
//
// If you are running this for a second time you wiull have to delete the table
spark.sql("DROP TABLE IF EXISTS workshop.geocoded")
//
// Standardiuze, validate, geocode the input addresses
val output_df = input_df.withColumn("geocode_result",
       callUDF("geocode",
          map(
         lit("mainAddressLine"), $"address",
         lit("areaName3"), $"_city",
         lit("areaName1"), $"_state",
//         lit("postCode1"), $"zipcode",
//          lit("country"), lit("USA")
        )
      )
    ).persist()   // we need this to prevent re-calculation
    .select("*", "geocode_result.*").drop("geocode_result").filter($"PB_KEY".isNotNull)
    // Create table in the metastore using DataFrame's schema and write data to it    
    //.write.format("delta").saveAsTable("workshop.real_estate_geocoded")
//
val geocoded_df = spark.table("workshop.real_estate_geocoded")
display(geocoded_df)


// COMMAND ----------

def travel_boundary(geocoded_input:DataFrame) : DataFrame = {
  // create a 5 min drive time travel boundary given a geocoded dataframe
  val polygonResult = geocoded_input
          .withColumn(
            "polygon",
            Holder.TravelBoundaryUDF(
              col("X").cast(DoubleType),
              col("Y").cast(DoubleType)
            )
          )
  return polygonResult
}

def geofence(travel_boundary: DataFrame) : DataFrame = {
  // find all locations inside each polygon
  val liResult = travel_boundary.filter("polygon is not null")
                 .withSpatialSearchColumns(
                   Seq(col("polygon").cast(StringType)),
                   contains,
                   includeEmptySearchResults = false
                 )
                 .drop(col("Obj")).drop("MI_Style")
  return liResult
}

// COMMAND ----------

// DBTITLE 1,Step 3 - Generate a Travel Boundary for Each Record
// This cell will take this input and compute the travel boundary and append it as a GeoJSON column to the output (see example in output below)
// The "X" below is the Latitude used as input, the "Y" is the Longitude
//
// Read in a dataset from the file system, or reference an earlier calculated dataframe

// If you are running this for a second time you wiull have to delete the table
spark.sql("DROP TABLE IF EXISTS workshop.real_estate_routed") 

// create geo-fence
val polygonResult = travel_boundary(geocoded_df)
//         spark.table("workshop.real_estate_geocoded")
//         .withColumn("polygon", Holder.TravelBoundaryUDF(col("X").cast(DoubleType), col("Y").cast(DoubleType)))

// Create table in the metastore using the output schema and write data to it    
polygonResult.write.format("delta").saveAsTable("workshop.real_estate_routed")

val routed_df = spark.table("workshop.real_estate_routed")
display(routed_df)

// COMMAND ----------

// DBTITLE 1,Step 4 - Identify All Addresses that Fit Inside the Travel Boundaries
// The input can be a file or the dataframe output from the previous step
// This cell will take the Geofence polygon (travel boundary) and provide an array of addresses that fit inside the polygon attached to each record
// Notice that the "polygon" field (below) is the name of the column in the input dataframe
//
// If you are running this for a second time you wiull have to delete the table
spark.sql("DROP TABLE IF EXISTS workshop.real_estate_geofenced")
//
val liResult = geofence(routed_df)
//               spark.table("workshop.real_estate_routed").filter("polygon is not null")
//              .withSpatialSearchColumns(Seq(col("polygon").cast(StringType)), contains, includeEmptySearchResults = false)
//             .persist()
//             .drop(col("Obj")).drop(col("polygon")).drop("MI_Style")
// The base result is a de-normalized dataframe, with each record from the input joined to each PBKEY (location) within the polygon
// Create table in the metastore using the output schema and write data to it    
liResult.write.format("delta").saveAsTable("workshop.real_estate_geofenced")
//
val located_df = spark.table("workshop.real_estate_geofenced")
display(located_df)

// COMMAND ----------


