import boto3
import time
import re
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
from boto3 import client



def image_to_text(documentName):
    with open(documentName, 'rb') as document:
        imageBytes = bytearray(document.read())
    textract = boto3.client('textract')# calling AWS TEXTRACT services using boto3 
    response = textract.detect_document_text(Document={'Bytes': imageBytes})

    text = ""
    # Print detected text
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text +=  item["Text"] + ' '  #appending text line by line 
    
    print(text)
    return text


def pdf_to_text(filename):
    pages=convert_from_path(filename)
    documentName=filename.split('.')[0]
    print(documentName)
    
    for page in pages:
        page.save(documentName,'JPEG')
        break

    im=Image.open(documentName)
    documentName=documentName+'.jpg'
    im.save(documentName)

    with open(documentName, 'rb') as document:
        imageBytes = bytearray(document.read())
    # calling AWS TEXTRACT services using boto3
    textract = boto3.client('textract')
    response = textract.detect_document_text(Document={'Bytes': imageBytes})

    text = ""
    # Print detected text
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text += item["Text"] + ' '
        
    return text
