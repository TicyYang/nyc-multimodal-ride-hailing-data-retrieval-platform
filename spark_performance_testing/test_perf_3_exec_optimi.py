print("3 executor, 6c, 4g")
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time
import datetime
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("MySparkSession") \
                            .config("spark.memory.offHeap.enabled", "true") \
                            .config("spark.memory.offHeap.size", "2g") \
                            .config("spark.master", "yarn") \
                            .config("spark.scheduler.mode", "FAIR") \
                            .config("spark.driver.cores", "1") \
                            .config("spark.driver.memory", "512m") \
                            .config("spark.executor.instances", "3") \
                            .config("spark.executor.cores", "6") \
                            .config("spark.executor.memory", "4g") \
                            .config("spark.memory.fraction", "0.8") \
                            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=20") \
                            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=20") \
                            .config("spark.checkpoint.compress", "true") \
                            .getOrCreate()


df = spark.read.parquet('/dataset/ts/TS6-201902_202306.parquet')
df = df.drop('is_holiday', 'TempTime', '__index_level_0__')
df = df.repartition(54)

mapping = {'yellow': 1, 
           'lyft': 2, 
           'uber': 3}

for key, value in mapping.items():
    df = df.withColumn("Name", when(df["Name"] == key, value).otherwise(df["Name"]))


for c in ['Name', 'year', 'month', 'day', 'hour', 'PULocationID']:
    df = df.withColumn(c, col(c).cast('integer'))


feature_variables = df.drop('count', 'countN')

inputcols = feature_variables.columns

assembler = VectorAssembler(inputCols=inputcols, outputCol="features")

df_val = df.filter((col("year") == 2023) & (col("month") > 3))


df_test = df.filter(
    (col("year") == 2022) |
    ((col("year") == 2023) & (col("month") <= 3))
)


output = assembler.transform(df_test)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=False)

scaled_data = scaler.fit(output).transform(output)

scaled_data.select('scaled_features', 'countN')

final_data = scaled_data.select('scaled_features', 'countN')

train, test = final_data.randomSplit([0.9, 0.1], seed=42)


gbtr = GBTRegressor(featuresCol='scaled_features', labelCol='countN', seed=1)


print("start時間:", datetime.datetime.now())
start_time = time.time()

gbtr_model = gbtr.fit(train)

end_time = time.time()
execution_time = end_time - start_time
print("執行時間:", execution_time, "秒")


time.sleep(30)

spark.stop()
