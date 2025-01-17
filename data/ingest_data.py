# Databricks notebook source
# MAGIC %md ## Ingest Data from generated text files into DELTA LAKE

# COMMAND ----------

# MAGIC %md ### Create a table with a surrogate key (autogenerated)
# MAGIC
# MAGIC By adding a key this will help DELTA further optimize how this data is compacted and partitioned

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE SCHEMA IF NOT EXISTS xomics_mri.evographnet

# COMMAND ----------

# MAGIC %sql DROP TABLE xomics_mri.evographnet.mri_bronze

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE xomics_mri.evographnet.mri_bronze (
# MAGIC   id BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   value STRING, 
# MAGIC   values_a ARRAY<double>,
# MAGIC   source_file_path STRING,
# MAGIC   description STRING,
# MAGIC   annotations STRING
# MAGIC ) --USING DELTA LOCATION '/dbfs/ml/evographnet/aggregated-delta-table'

# COMMAND ----------

# %sql 
# CREATE EXTERNAL TABLE xomics_mri.evographnet.mri_bronze_ext (
#   id BIGINT GENERATED ALWAYS AS IDENTITY,
#   value STRING, 
#   values_a ARRAY<double>,
#   source_file_path STRING,
#   filename STRING,
#   annotations STRING
# ) LOCATION 'dbfs:/ml/blogs/evographnet/mri_bronze_ext'

# COMMAND ----------

# MAGIC %sql -- DROP TABLE xomics_mri.evographnet.mri_bronze_ext

# COMMAND ----------

# MAGIC %sql DESCRIBE DETAIL xomics_mri.evographnet.mri_bronze

# COMMAND ----------

# MAGIC %sql -- ALTER TABLE xomics_mri.evographnet.mri_bronze ADD COLUMNS (annotations STRING)

# COMMAND ----------

# MAGIC %md ### Read the text source data and transform it into a Spark DataFrame

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, input_file_name, regexp_extract, regexp_replace, element_at, size, lit

# Read multiple text files into a single DataFrame
df = spark.read.text("/ml/gan/data/hls/dpm_dl/evographnet/data/*.txt")

# Add a new column containing the full path to the underlying file 
# and another containing only the filename (without extension)
filename_pattern = "[\/]([^\/]+)$"
df_with_filename = df.withColumn("source_file_path", input_file_name()).withColumn("filename", regexp_extract("source_file_path", filename_pattern, 1)).withColumn("description", regexp_replace("filename", ".txt", ""))#.withColumn("split_array", split("description", "\.")).withColumn("array_size", size("split_array")).withColumn("annotations", element_at("split_array", "array_size"))

display(df_with_filename)


# COMMAND ----------



# COMMAND ----------

display(df_last_segment)

# COMMAND ----------

cccccbkbrbrbeb

# COMMAND ----------

df_last_segment = df_last_segment.withColumn("scan_type", element_at("split_array", 1))
df_last_segment = df_last_segment.withColumn("hemisphere", element_at("split_array", 2))
display(df_last_segment)


# COMMAND ----------


# Transform the DataFrame to split the strings into arrays of floats
df_transformed = df_last_segment.withColumn("values_a", split(df_with_filename["value"], " ").cast("array<double>"))

df_transformed = df_transformed.select("value", "values_a", "source_file_path", "filename", "scan_type", "hemisphere", "annotations")

display(df_transformed)

# Transform the DataFrame to split the strings into arrays of floats
# df_transformed = df.withColumn("value", split(df["value"], " ").cast("array<double>"))


# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Write this data to our newly created DELTA table

# COMMAND ----------

from pyspark.sql import functions as F
# Save initial DataFrame to Delta Lake
# df_transformed.write.format("delta").saveAsTable("my_evographnet_data"))
# df_transformed.write.format("delta").save('/dbfs/ml/evographnet/aggregated-delta-table')
# df_transformed.format("delta").
df_transformed_key = df_transformed.withColumn("id", F.monotonically_increasing_id()).withColumnRenamed("filename", "description")
df_transformed_key = df_transformed_key.select("id", "value", "values_a", "source_file_path", "description", "scan_type", "hemisphere", "annotations")
display(df_transformed_key)


# COMMAND ----------

df_bronze = df_transformed_key.select("id", "value", "values_a", "source_file_path", "description", "scan_type", "hemisphere", "annotations")
# df_bronze.write.format("delta").mode("overwrite").save("dbfs:/ml/blogs/evographnet/mri_bronze_ext")
display(df_bronze)

# COMMAND ----------

# from pyspark.sql.functions import lit
# df_bronze_id = df_bronze.withColumn("id", lit("").cast("int"))
# df_bronze_id = df_bronze_id.select("id", "value", "values_a", "source_file_path", "description", "annotations")
# display(df_bronze_id)


# COMMAND ----------

# DBTITLE 0,e
df_bronze.write.format("delta").saveAsTable("xomics_mri.evographnet.mri_bronze_ext")

# COMMAND ----------

# MAGIC %sql SELECT * FROM xomics_mri.evographnet.mri_bronze_ext

# COMMAND ----------

# df_bronze_id.createOrReplaceTempView("EVO_BRONZE")

# COMMAND ----------

# %sql SELECT * FROM EVO_BRONZE

# COMMAND ----------

# %sql CREATE TABLE xomics_mri.evographnet.mri_bronze_ext SELECT * FROM EVO_BRONZE

# COMMAND ----------

# MAGIC %md ### Analyze this DELTA table to prepare for optimization

# COMMAND ----------

# MAGIC %sql 
# MAGIC ANALYZE TABLE xomics_mri.evographnet.mri_bronze_ext COMPUTE STATISTICS

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE xomics_mri.evographnet.mri_bronze_ext

# COMMAND ----------

# MAGIC %md ### Test performance of queries on our new DELTA table

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM xomics_mri.evographnet.mri_bronze_ext

# COMMAND ----------

# %sql OPTIMIZE xomics_mri.evographnet.mri_bronze_ext ZORDER BY (id, description)

# COMMAND ----------

# %sql 
# SELECT * FROM xomics_mri.evographnet.mri_bronze_ext
