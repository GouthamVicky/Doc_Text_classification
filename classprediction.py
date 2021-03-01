from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp
import numpy as np

def classpredict(text,spark,pipeline):

    prediction_data = spark.createDataFrame([[text]]).toDF("text")
    prediction_model = pipeline.fit(prediction_data)
    preds=prediction_model.transform(prediction_data)
    preds.select(preds.select("class.result").show(truncate=False))

    df = preds.select('class.result').toPandas()
    df.columns = ["class"]
    
    text = df.values.tolist()[0][0][0]

    print(text)
    return text
