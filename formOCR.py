# import all needed libraries
import cv2
import numpy as np
import re
from roboflow import Roboflow
import csv
import streamlit as st
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' 
from pytesseract import Output


# Fun of Preprocess the input image
def preprocessImg(inImg):
    # Load image and convert to grayscale
    #inImg = cv2.imread(imgAddr)

    # Get current size
    height, width = inImg.shape[:2]

    # Define new size
    new_width = 1400
    new_height = int(new_width * height / width) 

    # Resize
    inImg = cv2.resize(inImg, (new_width, new_height))
    gray = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)

    # Binarize
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 13)
    thresh = 255 - thresh

    return thresh
    
    
# Fun of filtering the words to get ones with high confidence and store them all as one sentence
def filterPytesseractResult(text, conf):
    words= []
    for i in range(len(text)):
        # remove some noisy character -can be solved with better model but this now is just for sake of representation-
        if text[i] == 'No' or re.findall(r'([ivxlc]+)\)', text[i]) or re.findall(r'\S*\]|(?<!\S)[xX] ?', text[i]):
            continue
        if conf[i]>=10:
            words.append(text[i])
    fStr = ''
    for i in range(len(words)):
        fStr+=words[i]+' '
    return fStr
    
    
# Extract Name, Travelling Date, and Flight Number using Regex
def extNameTDateFNo(fs):
    name = re.search(r'(?:(?<=Name: )|(?<=Name of the ))\b[A-Z]{3,}\s[A-Z]{2,}\b', fs).group()
    date = re.search(r'\d{2}/\d{2}/\d{4}', fs).group() 
    flightNo = re.search(r'(?:(?<=Flight No\. )|(?<=Flight Number ))\b[A-Z]{2,}\d{2,}\b', fs).group()

    return name, date, flightNo


# Extract Coordinates of Checked or Answered As Yes boxes Using YOLOv8 Model
# I have trained a model to detect the checked boxes as well as the signed as Yes 
def extBoxesCoord(img):
    # Load the trained model to detect the checkboxes
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("um-84bee").project("checkbox-detector-tv7p5")
    dataset = project.version(4)

    model = project.version(dataset.version).model

    # Do prediction on the image to get the Checked or Answered As Yes boxes coordinates
    pred = model.predict(img, confidence=50, overlap=30).json()
    
    return pred


# Fun of extracting text of the predicted Checked or Answered As Yes boxes
def extBoxesData(img, predBoxes):
    # Extract boxes coordinates to get the ROI
    roi = []
    for box in predBoxes['predictions']:
        x1 = int(box['x'] - (box['width']/2))
        y1 = int(box['y'] - (box['height']/2))
        x2 = int(box['x'] + (box['width']/2))
        y2 = int(box['y'] + (box['height']/2))
        roi.append(img[y1:y2, x1:x2])
               
    # Get ROI Text
    checkBoxesData = []
    for r in roi:
        checkBoxData = pytesseract.image_to_data(r, config='--psm 4', output_type=Output.DICT)
        checkBoxesData.append(filterPytesseractResult(checkBoxData['text'], checkBoxData['conf']))
    return checkBoxesData
    
    
def saveCSV(name, date, flightNo, checkBoxesData):
    with open('data.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Flight No', "Checked Boxes Or Answered As Yes"])  
        writer.writerow([name, date, flightNo, checkBoxesData])


# Get the whole form text
def extFormData(inpImg): 
    img = preprocessImg(inpImg)
    data = pytesseract.image_to_data(img, config='--psm 4', output_type=Output.DICT)
    finalStr = filterPytesseractResult(data['text'], data['conf'])    
    name, date, flightNo = extNameTDateFNo(finalStr)
    
    predBoxesCoord = extBoxesCoord(img)
    checkBoxesData = extBoxesData(img, predBoxesCoord)
    
    saveCSV(name, date, flightNo, checkBoxesData)
    
    return name, date, flightNo, checkBoxesData
    

#extFormData("custom_declaration_1.png")


# Layout
st.title("Form Data Extractor")


@st.cache_data
def load_image(uploaded_file):
   img = Image.open(uploaded_file) 
   img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
   return img

uploaded_file = st.file_uploader("Upload Form Image", type=["png","jpg","jpeg"]) 


if uploaded_file is not None:
    img = load_image(uploaded_file)
    
    # Extraction
    name, date, flight, boxes = extFormData(img) 
    
    # Display 
    st.image(img, use_column_width=True)
    st.write("Extracted Information:")
    
    # Text inputs
    name_input = st.text_input("Name", name)  
    date_input = st.text_input("Date", date)
    flight_input = st.text_input("Flight", flight)
    boxes_input = st.text_input("Checked Boxes", boxes)
    
    file = open('data.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(['name', 'date', 'flight', 'Checked Boxes'])
    print("all")
    if st.button("Update"):
        name = name_input 
        date = date_input
        flight = flight_input
        boxes = boxes_input
        writer.writerow([name, date, flight, boxes_input])
        print("Update")
    # Save button just for confirmation  
    if st.button("Save To CSV"):
        # Write CSV row with updated values         

        file.close()
        st.success("Saved to CSV") 
        print("Save")
        

    # Display  
    st.write("Name:", name)
    st.write("Date:", date)
    st.write("Flight:", flight)
    st.write("Checked Boxes:", boxes_input)
else:
    st.write("Upload an image")


