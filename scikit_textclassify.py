from typing import Optional
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import uvicorn
import os
import numpy as np
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi import Response, status
from doc2text import pdf_to_text, image_to_text
import shutil
from fastapi import File, UploadFile
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import load
app = FastAPI()


bearer_token = '1SrFepgvVxPzjHQl6bBIqdim4JG2TRNfs'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
file_types_allowed = ['png', 'jpeg', 'jpg', 'JPEG', 'JPG', 'PNG']
all_file_types_allowed = ['png', 'jpeg',
                          'jpg', 'JPEG', 'JPG', 'PNG', 'pdf', 'PDF']


@app.get("/", status_code=200)
async def check(response: Response):
    return "Working"




@app.post("/text/classification/")
def text(response: Response, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    if token != bearer_token:
        response.status_code = status.HTTP_401_UNAUTHORIZED
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
            documentName = '/tmp/'+file.filename
            im = Image.open(file.file)
            im.save(documentName)

            text = image_to_text(documentName)

        else:
            print("==============>PDF FLOW<===================")
            print(file.filename)
            documentName = '/tmp/'+file.filename

            with open(documentName, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            text = pdf_to_text(documentName)

        model = load("logreg_text_classification.pkl")
        print(text)
        value=[text]
        print(value)
        text_class=model.predict(value)
    
        print(text_class)
        total_score=model.predict_proba(value)
        print(total_score)
        score =model.predict_proba(value)[:, 1]
        score=score.tolist()
        print(score)
        confidence_threshold=int(str(score[0]).split(".")[0])
        if confidence_threshold >0:
            text_class=text_class.tolist()
        else:
            if " gas " in text.lower() or " oil " in text.lower() or " refill " in text.lower() or "gas" in text.lower() or "oil" in text.lower() or "refill" in text.lower():
                text_class=['Gas Bills']
            else:
                text_class=["Cannot be Classified"]
                response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        json = {"class": text_class[0].capitalize()}
        print(json)
        return json

    except Exception as e:
        error = str(e)
        response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        return{"error": error, "status": "unable to extract data"}


if __name__ == "__main__":
    uvicorn.run("scikit_textclassify:app", host="0.0.0.0",port=19014, log_level="info", workers=1)
