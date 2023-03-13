from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp
import numpy as np
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
def classpredict(text,spark,pipeline):

    prediction_data = spark.createDataFrame([[text]]).toDF("text")
    prediction_model = pipeline.fit(prediction_data)
    preds=prediction_model.transform(prediction_data)
    preds.select(preds.select("class.result").show(truncate=False))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    confidence_threshold=evaluator.evaluate(preds)
    print(confidence_threshold)

    df = preds.select('class.result').toPandas()
    df.columns = ["class"]
    
    text_class = df.values.tolist()[0][0][0]
    if confidence_threshold>0.7:
        print(text_class)
        return text_class
    else:
        if " gas " in text.lower() or " oil " in text.lower() or " refill " in text.lower() or "gas" in text.lower() or "oil" in text.lower() or "refill" in text.lower():
                text='Gas Bills'
        else:
            text="Cannot be Classified"
            
        
        if text=="aadhar":
            text="Aadhar Card"

        elif text=="drivingLicense":
            text="Driving License"

        elif text=="voterID":
            text="Voter ID"
        else:
            pass
        return text_class


        
    
