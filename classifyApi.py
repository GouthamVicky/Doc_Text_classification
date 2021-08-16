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
#import nltk
#from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
#nltk.download('stopwords')
app = FastAPI()


bearer_token = '1SrFepgvVxPzjHQl6bBIqdim4JG2TRNfs'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
file_types_allowed = ['png', 'jpeg', 'jpg', 'JPEG', 'JPG', 'PNG']
all_file_types_allowed = ['png', 'jpeg',
                          'jpg', 'JPEG', 'JPG', 'PNG', 'pdf', 'PDF']


@app.get("/", status_code=200)
async def check(response: Response):
    return "Working"


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS=['the', 'what', 'ours', 'her', 'be', 'during', 'theirs', 'being', 'wasn', 'on', 'aren', 'hers', 'needn', 'you', 'by', 'after', 'very', 'a', "haven't", 'other', 'been', 're', 'not', "that'll", 'yourself', 'have', 'few', 'this', 'why', 'their', 'can', 'of', 'had', 'between', 'where', 'did', 'just', 'because', 'out', 'having', 'mightn', 'under', 'through', 'ain', 'about', 'nor', "you've", 'each', 'most', "hasn't", 'haven', "mightn't", 'if', 'against', 'no', "she's", 'his', 'm', "you'd", 'there', 'who', 'o', 'how', 'any', 'or', 'same', 'we', 'that', 'is', 'shouldn', 'weren', "weren't", 'wouldn', "isn't", "shan't", "hadn't", 'he', 'itself', 'so', 'to', "it's", 'and', "should've", 'has', 'she', 'some', 'isn', 'again', 'only', 'own', 'doesn', "won't", 'am', 'whom', 'for', 'your', 'd', 'y', 'shan', 'does', 've', 'them', 'down', 'ma', "mustn't", 'further', 'from', 'over', 'off', 'in', 'should', 'didn', 'they', 'ourselves', 'it', "didn't", "doesn't", "aren't", 'which', 'yourselves', 'doing', "you're", 'those', "needn't", 'i', 'all', 'below', 'into', 'as', 'more', 's', 'was', 'until', 'before', "you'll", 'him', 'themselves', 'do', 'mustn', 'but', 'will', 'above', 'myself', 'now', 'couldn', 'my', "shouldn't", 'at', 'than', 'both', 'our', 'with', "don't", 'yours', "wasn't", 'don', "couldn't", 'here', 'won', 'when', 'll', 'hadn', 'me', 'then', 
't', 'too', 'are', "wouldn't", 'were', 'hasn', 'while', 'herself', 'once', 'its', 'these', 'such', 'himself', 'an', 'up']


def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text



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

        text=clean_text(text)
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
        
        if text_class[0]=="aadhar":
            text_class[0]="Aadhar Card"

        elif text_class[0]=="drivingLicense":
            text_class[0]="Driving License"

        elif text_class[0]=="voterID":
            text_class[0]="Voter ID"
        else:
            pass
        json = {"class": text_class[0]}
        print(json)
        return json

    except Exception as e:
        error = str(e)
        response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        return{"error": error, "status": "unable to extract data"}



@app.post("/text/classification/utility/")
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

        text=clean_text(text)
        print("The text is printing=======>",text)
        value=[text]
        print(value)
        model=load("Pvt_text_classification.pkl")
        text_class=model.predict(value)
    
        print(text_class)
        total_score=model.predict_proba(value)
        print("The total score is printing========>",total_score)
        score =model.predict_proba(value)[:, 1]
        score=score.tolist()
        print("The Final probab score======>",score)
        confidence_threshold=int(str(score[0]).split(".")[0])
        print(confidence_threshold)
        
        print(confidence_threshold)
        
        if confidence_threshold>0:
            
            text_class=text_class.tolist()
        else:

            print("LESSER CONFIDENCE FLOW")
            if "account statement" in text.lower() or "statement account" in text.lower() or "transaction" in text.lower() or "statement period" in text.lower():
                text_class=["bankstatement"]
            elif "Jio DIGITAL LIFE" in text.lower() or "JioPostPaid Plus" in text.lower() or "Jio Number" in text.lower() or "Vodafone India Ltd Company" in text.lower() or "NNECT BROADBAND " in text.lower() or "MAHANAGAR TELEPHONE NIGAM LIMITED" in text.lower() or "TATA DOCOMO" in text.lower() or "BHARAT SANCHAR" in text.lower() or "net+ BROADBAND" in text.lower() or "broadband" in text.lower() or "telephone" in text.lower() :
                print("This is PhoneBill")
                text_class= ["phonebill"]
                
            elif text_class=="bankPassbook" and " bank " in text.lower().split(" ") or "pass book" in text.lower().split(" ") or "Cheque" in text.lower().split(" "):
                text_class=["bankPassbook"]
            elif "kw" in text.lower() or "kwh" in text.lower() or "kvah" in text.lower() or "tangedco" in text.lower():
                text_class=["electricitybill"]
            elif text_class=="bankstatement  "and "bank"  in text.lower().split(" "):
                text_class=["bankstatement"]
            



            
            else:
                text_class=["Cannot be Classified"]
                response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        
        
        json = {"class": text_class[0]}
        print(json)
        return json

    except Exception as e:
        error = str(e)
        response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        return{"error": error, "status": "unable to extract data"}



if __name__ == "__main__":
    uvicorn.run("classifyApi:app", host="0.0.0.0",port=19011, log_level="info", workers=1)
