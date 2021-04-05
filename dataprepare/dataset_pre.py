import os
import csv
import boto3

def textextraction(documentName):
    print(documentName)
# Read document content
    with open(documentName, 'rb') as document:
        imageBytes = bytearray(document.read())

    # Amazon Textract client
    textract = boto3.client('textract')

    # Call Amazon Textract
    response = textract.detect_document_text(Document={'Bytes': imageBytes})

    text = " "
    # Print detected text
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text += item["Text"] + ' '

    text = text.lower()

    return text


output={}
directory = "training/"
folder=os.listdir(directory)
for sub in folder:
    output[sub]=[]
    files=os.listdir(directory+sub)

for img in files:
    text = textextraction(directory+sub+'/'+img)
    output[sub].append(text)


a_file = open("sample.csv", "w")


writer = csv.writer(a_file)

for key, value in output.items():
    for i in value:
        print(key+":"+i)
        writer.writerow([key, i])


a_file.close()

    
