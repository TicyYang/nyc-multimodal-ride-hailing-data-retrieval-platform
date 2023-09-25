print("202201~202303存成csv")
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
                            .config("spark.memory.offHeap.size", "3g") \
                            .config("spark.master", "yarn") \
                            .config("spark.driver.cores", "1") \
                            .config("spark.driver.memory", "512m") \
                            .config("spark.executor.instances", "4") \
                            .config("spark.executor.cores", "5") \
                            .config("spark.executor.memory", "5g") \
                            .config("spark.memory.fraction", "0.8") \
                            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=20 -XX:+ParallelRefProcEnabled") \
                            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=20 -XX:+ParallelRefProcEnabled") \
                            .config("spark.checkpoint.compress", "true") \
                            .getOrCreate()


df = spark.read.parquet('/dataset/ts/TS6-201902_202306.parquet')
df = df.drop('is_holiday', 'TempTime', '__index_level_0__')

mapping = {'yellow': 1, 
           'lyft': 2, 
           'uber': 3}

for key, value in mapping.items():
    df = df.withColumn("Name", when(df["Name"] == key, value).otherwise(df["Name"]))


for c in ['Name', 'year', 'month', 'day', 'hour', 'PULocationID']:
    df = df.withColumn(c, col(c).cast('integer'))

df = df.repartition(40)

feature_variables = df.drop('count', 'countN')

inputcols = feature_variables.columns

assembler = VectorAssembler(inputCols=inputcols, outputCol="features")

df_val = spark.read.parquet('/dataset/ts/TS10.parquet')

df_test = df.filter(
    (col("year") == 2022) |
    ((col("year") == 2023) & (col("month") <= 3))
)

output = assembler.transform(df_test)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=False)

scaled_data = scaler.fit(output).transform(output)

final_data = scaled_data

train, test = final_data.randomSplit([0.9, 0.1], seed=42)

df_val = df_val.withColumn("count", lit(0))

print("start時間:", datetime.datetime.now())
start_time = time.time()

gbtr = GBTRegressor(featuresCol='scaled_features', labelCol='countN', maxDepth=10, maxIter=100, stepSize=0.1, subsamplingRate=0.8, cacheNodeIds=True, seed=1, maxMemoryInMB=10240, maxBins=64)


gbtr_model = gbtr.fit(train)

y_pred = gbtr_model.transform(test)

y_pred.select('countN', 'prediction')

end_time = time.time()
execution_time = end_time - start_time
print("執行時間:", execution_time, "秒")
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='countN')

r2 = evaluator.evaluate(y_pred, {evaluator.metricName: 'r2'})
mae = evaluator.evaluate(y_pred, {evaluator.metricName: 'mae'})
rmse = evaluator.evaluate(y_pred, {evaluator.metricName: 'rmse'})

print(f'R2: {r2}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')


output_df_val = assembler.transform(df_val)


scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=False)

scaled_data = scaler.fit(output_df_val).transform(output_df_val)

valid_data = scaled_data



y_pred_2023 = gbtr_model.transform(valid_data)
#evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='count')

#r2 = evaluator.evaluate(y_pred_2023, {evaluator.metricName: 'r2'})
#mae = evaluator.evaluate(y_pred_2023, {evaluator.metricName: 'mae'})
#rmse = evaluator.evaluate(y_pred_2023, {evaluator.metricName: 'rmse'})

#print(f'R2: {r2}')
#print(f'MAE: {mae}')
#print(f'RMSE: {rmse}')
#print()


#y_pred_2023.select("prediction").show(10)
y_pred_2023 = y_pred_2023.withColumn('prediction_integer', col('prediction').cast('integer'))
y_pred_2023 = y_pred_2023.withColumn('prediction_integer', when(col('prediction_integer') < 0, 0).otherwise(col('prediction_integer')))
y_pred_2023.select('Name', 'year', 'month', 'day', 'hour', 'PULocationID', 'weekday', 'lat', 'lon', 'prediction_integer').write.csv("predictions.csv", header=True, mode="overwrite")

spark.stop()
