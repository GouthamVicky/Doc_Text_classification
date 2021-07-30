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
    document_1 = DocumentAssembler()\
      .setInputCol("category")\
      .setOutputCol("document")

    # we can also use sentece detector here if we want to train on and get predictions for each sentence
    use_1= UniversalSentenceEncoder.load("sparkmodel/usemodel-20210727T105431Z-001/usemodel/use")\
        .setInputCols("sentence","token")\
        .setOutputCol("embedding")

    # the classes/labels/categories are in category column
    classsifierdl_1 = ClassifierDLApproach.load("sparkmodel/usemodel-20210727T105431Z-001/usemodel/classiferdl")\
        .setInputCols("sentence","token","embedding")\
        .setOutputCol("class")
    global use_clf_pipeline
    use_clf_pipeline = Pipeline(
        stages = [
            document_1,
            use_1,
            classsifierdl_1
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
    
@app.post("/text/classification/utility/")
def text(response: Response, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    if token != bearer_token:
        response.status_code=status.HTTP_401_UNAUTHORIZED
        return "Unauthorized access"
    
    if (file.filename.split('.')[-1]) not in all_file_types_allowed:
        response.status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        return "Invalid Document"
    else:
        pass
    
    
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

    text_class=classpredict(text,spark,use_clf_pipeline)
    print(text_class)
    total_score=[]
    score=[]
    print("The total score is printing========>",total_score)
    score =classpredict.predict_proba()[:, 1]
    score=score.tolist()
    print("The Final probab score======>",score)
    confidence_threshold=int(str(score[0]).split(".")[0])
    print(confidence_threshold)
    
    print(confidence_threshold)
    
    if confidence_threshold >0.45:
        text_class=text_class.tolist()
    else:

        print("LESSER CONFIDENCE FLOW")
        if "Jio DIGITAL LIFE" in text.lower() or "JioPostPaid Plus" in text.lower() or "Jio Number" in text.lower() or "Vodafone India Ltd Company" in text.lower() or "NNECT BROADBAND " in text.lower() or "MAHANAGAR TELEPHONE NIGAM LIMITED" in text.lower() or "TATA DOCOMO" in text.lower() or "BHARAT SANCHAR" in text.lower() or "net+ BROADBAND" in text.lower():
            print("This is PhoneBill")
            text_class= ["phonebill"]
            
        elif text_class=="bankPassbook" and " bank " in text.lower().split(" ") or "pass book" in text.lower().split(" ") or "Cheque" in text.lower().split(" "):
            text_class=["bankPassbook"]
        
        elif text_class=="bankstatement  "and "bank" in text.lower().split(" "):
            text_class=["bankstatement"]
        
        else:
            text_class=["Cannot be Classified"]
            response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        

if __name__ == "__main__":
    uvicorn.run("classifyApi:app", host="0.0.0.0",port=19014, log_level="info", workers=4)
