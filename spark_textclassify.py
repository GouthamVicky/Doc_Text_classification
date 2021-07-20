from typing import Optional
from fastapi import FastAPI , Depends
from pydantic import BaseModel
import uvicorn
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp
import numpy as np 
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi import Response, status
from doc2text import pdf_to_text,image_to_text 
import shutil
from fastapi import File, UploadFile
from PIL import Image
from classprediction import classpredict
app = FastAPI()




bearer_token = '1SrFepgvVxPzjHQl6bBIqdim4JG2TRNfs'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
file_types_allowed = ['png', 'jpeg', 'jpg', 'JPEG', 'JPG', 'PNG']
all_file_types_allowed = ['png', 'jpeg', 'jpg', 'JPEG', 'JPG', 'PNG','pdf','PDF']

@app.get("/",status_code=200)
async def check(response: Response):
    return "Working"


@app.on_event("startup")
async def startup_event():
    global spark
    spark = sparknlp.start()

    document = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

    use = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

    classsifierdl = ClassifierDLModel.load("./tmp_classifierDL_model") \
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("class")

    global pipeline
    pipeline = Pipeline(
        stages = [
            document,
            use,
            classsifierdl
        ])


@app.post("/text/classification/")
def text(response: Response, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    if token != bearer_token:
        response.status_code=status.HTTP_401_UNAUTHORIZED
        return "Unauthorized access"
    
    if (file.filename.split('.')[-1]) not in all_file_types_allowed:
        response.status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        return "Invalid Document"
    else:
        pass
    
    try:
        if (file.filename.split('.')[-1]) in file_types_allowed:
            print("================ > Image Flow < ================")
            print(file.filename)
            documentName='/tmp/'+file.filename
            im=Image.open(file.file)
            im.save(documentName)

            text=image_to_text(documentName)
        
        else:
            print("==============>PDF FLOW<===================")
            print(file.filename)
            documentName='/tmp/'+file.filename

            with open(documentName, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            text=pdf_to_text(documentName)

        text_class=classpredict(text,spark,pipeline)
        print(text_class)
        json={"class":text_class}
        
        return json
    
    except Exception as e :
        error=str(e)
        response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        return{"error": error, "status": "unable to extract data"}
        

if __name__ == "__main__":
    uvicorn.run("classifyApi:app", host="0.0.0.0",port=19014, log_level="info", workers=4)
