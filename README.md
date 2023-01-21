# Revolutionizing Japanese Business Card Management with YoloV5 and OCR: Automated Fragmentation and Label Recognition

![image](https://user-images.githubusercontent.com/21255010/213827783-048d6516-8a4d-441a-a9e6-51f249b957e9.png)

## Description
I took on a freelance challenge to develop an algorithm for recognizing and labeling the elements on a business card for Sansan, a digital business card organization service. The algorithm utilizes machine learning techniques, specifically Pytorch, YoloV5 and opencv to enhance the accuracy and efficiency of the service.

During the scanning process, business cards are fragmented into multiple parts for safe and effective data import. However, even with optical character recognition technology, achieving a low error rate in the conversion process remains a challenge.

I accomplished the following steps:

    1. Collected and pre-processed the bussiness card image dataset
    2. Extracted and engineered the labels of each elements in Business Card in format used by YoloV5 neural network architecture
    3. Trained and evaluated the model using Pytorch and YoloV5
    4. Integrated the algorithm using opencv into the existing system to automatically recognize and label the elements on the business card.
This project was a great opportunity to showcase my skills in machine learning, computer vision, and integration while solving a real-world problem for Sansan.

## Data Source
Business Card Image data and its labels are taken from this [Link](https://signate.jp/competitions/26)

## YoloV5 Model Evaluation Results
A platform called [Weights and Biases](https://wandb.ai) is used to track the training process of the YoloV5 model, as well as monitor its metrics. 

**Experiments Logs Link** : https://wandb.ai/aimagic/sansan-business-card-recognition/runs/2a8sztxh/overview?workspace=user-aimagic 

Evaluation Results for each label from Business Card Images

![](https://shubh2016shiv-japanese-business-card-recognition-ocr-app-o2j5tk.streamlit.app/~/+/media/a5efe58af02f23ff0ebe75615b0dd413ff8cc288be2804724dc38182.gif)

## Quick Demo

Here is the quick demo of the project
![](https://github.com/shubh2016shiv/japanese_business_card_recognition_ocr/blob/main/resources/Japanese%20Business%20Card%20Fragmentation%20and%20Label%20Detection-2.gif)

## Web-Application 

**Click on Below icon for opening web application for this project**

[![streamlit](https://th.bing.com/th/id/OIP.hiunGrftVRVZAE3IJXUMowHaEb?pid=ImgDet&rs=1)](https://shubh2016shiv-japanese-business-card-recognition-ocr-app-o2j5tk.streamlit.app/)
