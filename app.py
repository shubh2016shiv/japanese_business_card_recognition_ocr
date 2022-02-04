import streamlit as st
from configparser import ConfigParser
import gdown
import os
import subprocess
import random
from PIL import Image
import shutil
import easyocr
from stqdm import stqdm
from skimage import io
import gc
from japaneseOCR import JapaneseOCR
import json

st.set_page_config(layout="wide")
st.title("Project - Business Card Recognition Challenge")


nav_option = st.sidebar.selectbox(label="Navigation", options=['Home', "Detect and Fragment Labels", "Perform Japanese OCR"])
config = ConfigParser()
config.read('config.ini')

##################################################################
# Functions
def prepareImageForTesting(file, imagePath=None):
    if imagePath is None:
        st.image(file)
        image = Image.open(file)
    else:
        st.image(imagePath + file)
        image = Image.open(imagePath + file)

    if os.path.isdir(config['resources']['detection_folder'] + "businessCard/"):
        shutil.rmtree(config['resources']['detection_folder'] + "businessCard/")
        os.mkdir(config['resources']['detection_folder'] + "businessCard/")
    else:
        os.mkdir(config['resources']['detection_folder'] + "businessCard/")
    image.save(config['resources']['detection_folder'] + "businessCard/businessCard.png")
    
@st.experimental_singleton
def get_ocr_model(ocr_path):
    reader = easyocr.Reader(['en', 'ja'], model_storage_directory=ocr_path,gpu=False,quantize=False,download_enabled=False)
    return reader

def recognize(ocr_model_,extracted_img_):
    try:
        #img = Image.open(extracted_img_)
        img = io.imread(extracted_img_)
        #st.write(img)
        if img is not None:
            gc.collect()
            result = ocr_model_.readtext_batched(img, decoder='beamsearch', beamWidth=1,detail=False,paragraph=False)
            return " ".join(result[0])
    except Exception as e:
        st.error("Failed to detect any text from image due to: ", e)
        return None

def getBoundingBoxPos(imgWidth, imgHeight):
    labelIndexDict = {0: 'companyName', 1: 'fullName', 2: 'positionName', 3: 'address', 4: 'phoneNumber', 5: 'fax',
                      6: 'mobile', 7: 'email', 8: 'url'}
    labelsDict = {}
    with open("./runs/detect/results/labels/businessCard.txt") as f:
        labelFile = f.readlines()

    for detectedLabel in labelFile:
        items = detectedLabel[:-2].split(" ")
        xcen = float(items[1])
        ycen = float(items[2])
        w = float(items[3])
        h = float(items[4])

        left = (xcen - w / 2.) * imgWidth;
        right = (xcen + w / 2.) * imgWidth;
        top = (ycen - h / 2.) * imgHeight;
        bottom = (ycen + h / 2.) * imgHeight;

        labelsDict[labelIndexDict[int(items[0])]] = {'left': left, 'right': right, 'top': top, 'bottom': bottom,
                                                     'probability': float(items[5])}

    return labelsDict

def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)    
    
##################################################################

if nav_option == "Home":
    st.subheader("Challenge presented by Sansan Global PTE. LTD. to Recognize labellings on Japanese business card.")
    st.image(config['resources']['project_title_img'], use_column_width=False)
    st.write("----")
    st.subheader("Business Problem")
    with open(config['resources']['project_objective'], 'r', encoding='utf-8') as obj_file:
        obj = obj_file.read()

    st.write(obj)
    st.write("Link to the official challenge is [🔗](https://signate.jp/competitions/26)")
    
    st.subheader("Tracking Project Logs and Monitoring Training Metrics")
    '''
    WandB (https://wandb.ai) is a great tool for tracking Data Science project and monitoring training metrics. 
    YOLOv5 now comes with a native Weights & Biases integration that tracks model pipelines – including model
    performance, hyperparameters, GPU usage, predictions, and datasets. The YOLO-V5 model training was tracked using
    WandB and the link is given below: 
    '''
    st.markdown(
        '[This Project Experiment Logs 🔗](https://wandb.ai/aimagic/sansan-business-card-recognition/runs/2a8sztxh/overview?workspace=user-aimagic)')
    '''
    The detection was done for 9 labels on Japanese Business Card Images- 'company_name','full_name','position_name','address','phone_number','fax','mobile','email','url'
    using Yolo-V5 model  
    '''

    st.image(config['result']['metrics_gif_path'])

    # Project Artifacts Downloads
    st.subheader("Download Project Artifacts")

    if not os.path.isdir(config['downloads']['downloads_folder']):
        os.makedirs(config['downloads']['downloads_folder'])

    download_option = st.selectbox(label="Download Options",
                                   options=["Jupyter Notebook: Read, Process and Train YOLO-V5 model",
                                            "Set of Business Cards (.zip)",
                                            "Best Yolo-V5 Model",
                                            "Japanese OCR model"])

    if download_option == "Jupyter Notebook: Read, Process and Train YOLO-V5 model":
        download_file_name = config['downloads']['jupyter_notebook']
        with open(download_file_name, "rb") as file:
            st.download_button(label="Download", data=file, file_name="read_process_and_train.ipynb")
    elif download_option == "Set of Business Cards (.zip)":
        ID = None
        with open(config['downloads']['sample_business_cards_share_zip']) as f:
            ID = f.readline()

        if ID is not None:
            try:
                sample_business_cards_path = config['downloads']['downloads_folder'] + "/" + "testBusinessCards.zip"
                if not os.path.exists(sample_business_cards_path):
                    gdown.download_file_from_google_drive(ID, sample_business_cards_path)
                with open(sample_business_cards_path, "rb") as file:
                    st.download_button(label="Download", data=file, file_name="sample_business_cards.zip")
            except Exception as e:
                st.error(e.message)

    elif download_option == "Best Yolo-V5 Model":
        with open(config['downloads']['yolo_v5_weights'], "rb") as file:
            st.download_button(label="Download", data=file, file_name="best.pt")

    elif download_option == "Japanese OCR model":
        ocr_models = config['downloads']['ocr_models']
        model_1,model_2 = ocr_models.split(";")
        st.info(f"There are two models associated with OCR with model names: '{model_1}' and '{model_2}'")
        if not os.path.isdir(config['downloads']['ocr_models_dir']):
            os.makedirs(config['downloads']['ocr_models_dir'])

        try:
            with st.spinner("Downloading OCR models"):
                if not os.path.exists(config['downloads']['ocr_models_dir'] + "/" + 'craft_mlt_25k.pth'):
                    gdown.download_file_from_google_drive('1tWxsXULB1bBGjRdroDU3BpRhc4PqtpW-',
                                                          config['downloads']['ocr_models_dir'] + "/" + 'craft_mlt_25k.pth')

                if not os.path.exists(config['downloads']['ocr_models_dir'] + "/" + 'japanese_g2.pth'):
                    gdown.download_file_from_google_drive('10hpsSpyDDnBWh8jOh_l86tlPXWUzoYFw',
                                                          config['downloads']['ocr_models_dir'] + "/" + 'japanese_g2.pth')

            with open(config['downloads']['ocr_models_dir'] + "/" + 'craft_mlt_25k.pth', "rb") as file:
                st.download_button(label=f"Download {model_1}", data=file, file_name='craft_mlt_25k.pth')

            with open(config['downloads']['ocr_models_dir'] + "/" + 'craft_mlt_25k.pth', "rb") as file:
                st.download_button(label=f"Download {model_2}", data=file, file_name='japanese_g2.pth')
        except Exception as e:
            st.error(e.message)
elif nav_option == "Detect and Fragment Labels":
    st.write("-"*10)
    st.subheader("Identification and Fragmentation of Labels on Digital Business Card using Pre-trained Yolo-V5 Model Pipeline")
    detect_option = st.selectbox("Select below options to identify labels on business card",
                                 ["Show already available Digital Japanese Business Cards", "Upload"])

    if detect_option == "Show already available Digital Japanese Business Cards":
        IDs = []
        with open(config['resources']['available_cards']) as f:
            links = f.readline()
            links = links.split(",")
            for link in links:
                IDs.append(link.split("/")[-2])

        randomIDs = random.sample(IDs, 9)
        if not os.path.isdir(config['resources']['detection_folder'] + "randomBusinessCards"):
            os.mkdir(config['resources']['detection_folder'] + "randomBusinessCards")

            for i, randomID in enumerate(randomIDs):
                gdown.download_file_from_google_drive(randomID,
                                                      config['resources'][
                                                          'detection_folder'] + "randomBusinessCards/Business Card " + str(
                                                          i) + ".png")

        businessCard = st.selectbox("Select One Business Card", ["Business Card " + str(i) + ".png" for i in range(9)])
        prepareImageForTesting(businessCard, config['resources']['detection_folder'] + "randomBusinessCards/")
    elif detect_option == "Upload":
        st.info("You may upload one image from the downloaded zip file")
        upload_file = st.file_uploader("Upload the Business Card Image", type='png')
        if upload_file is not None:
            prepareImageForTesting(upload_file, imagePath=None)

    if st.button("Detect Labels"):
        with st.spinner("Detection in progress..."):
            if os.path.isdir('./runs'):
                shutil.rmtree("./runs")

            detectCommand = "/home/appuser/venv/bin/python ./detect.py --weights ./resources/yolo_model_weights/best.pt --source ./resources/test_image/businessCard/ --img 512 --conf 0.6 --save-crop --save-conf --line-thickness 2 --iou-thres 0.5 --save-txt --name results"
            p = subprocess.Popen(detectCommand, stdout=subprocess.PIPE, shell=True)
            p.wait()
            p.terminate()
        st.subheader("Detected Labels and their Scores")
        st.image(config['result']['labelled_img'])

elif nav_option == "Perform Japanese OCR":
    st.write("-"*10)
    st.subheader("Recognition of Japanese and English text on Extracted Fragment from Business Cards using EasyOCR ")
    gc.collect()
    if not os.path.isdir(config['downloads']['ocr_models_dir']):
        os.makedirs(config['downloads']['ocr_models_dir'])
    
        if not os.path.exists(config['downloads']['ocr_models_dir'] + "/" + 'craft_mlt_25k.pth'):
            with st.spinner("Downloading OCR model: 'craft_mlt_25k.pth'"):
                gdown.download_file_from_google_drive('1tWxsXULB1bBGjRdroDU3BpRhc4PqtpW-',
                                                      config['downloads'][
                                                          'ocr_models_dir'] + "/" + 'craft_mlt_25k.pth')
    
        if not os.path.exists(config['downloads']['ocr_models_dir'] + "/" + 'japanese_g2.pth'):
            with st.spinner("Downloading OCR model: 'japanese_g2.pth'"):
                gdown.download_file_from_google_drive('10hpsSpyDDnBWh8jOh_l86tlPXWUzoYFw',
                                                      config['downloads'][
                                                          'ocr_models_dir'] + "/" + 'japanese_g2.pth')
    ocr_model = get_ocr_model(config['downloads']['ocr_models_dir'])
    if os.path.isdir('./runs'):
        extracted_label_imgs = [config['result']['crop_address'],
                                config['result']['crop_company_name'],
                                config['result']['crop_email'],
                                config['result']['crop_fax'],
                                config['result']['crop_full_name'],
                                config['result']['crop_mobile'],
                                config['result']['crop_phone_no'],
                                config['result']['crop_position_name'],
                                config['result']['crop_url']]

        extracted_label = st.selectbox(label="Extracted Fragments",options=[
            "Address","Company Name","Email","Fax","Full Name","Mobile","Phone Number","Position","URL"
        ])

        if extracted_label == "Address":
            fragment = extracted_label_imgs[0]
        elif extracted_label == "Company Name":
            fragment = extracted_label_imgs[1]
        elif extracted_label == "Email":
            fragment = extracted_label_imgs[2]
        elif extracted_label == "Fax":
            fragment = extracted_label_imgs[3]
        elif extracted_label == "Full Name":
            fragment = extracted_label_imgs[4]
        elif extracted_label == "Mobile":
            fragment = extracted_label_imgs[5]
        elif extracted_label == "Phone Number":
            fragment = extracted_label_imgs[6]
        elif extracted_label == "Position":
            fragment = extracted_label_imgs[7]
        elif extracted_label == "URL":
            fragment = extracted_label_imgs[8]
        
        st.image(fragment)
        st.sidebar.info("Japanese text formatting options")
        ja_font = st.sidebar.radio(label="Japanese Fonts",options=['游明朝','Yu Gothic','Yu Mincho','游ゴシック'])
        font_size = st.sidebar.slider(label="Font Size (px)", min_value=30,max_value=50,value=40,step=1)
        if st.button("Perform OCR on selected fragment"):
            text = recognize(ocr_model,fragment)
            text = f'<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{text}</p>'
            st.markdown(text, unsafe_allow_html=True)

        st.write("-"*10)
        if st.sidebar.button("Show Complete OCR Result"):
            jOCR = JapaneseOCR(ocr_model=ocr_model,extracted_images=extracted_label_imgs)
            jOCR.recognize()

            recognizedLabels = jOCR.detections

            labelExpander = st.expander("Expand to reveal detected labels and their bounding box on Business Card")
            col1, col2, col3 = st.columns(3)
            with col1:
                fullNameExpander = st.expander("Expand to reveal detected Full Name as text and cropped image")
            with col2:
                companyNameExpander = st.expander("Expand to reveal detected Company Name as text and cropped image")
            with col3:
                positionExpander = st.expander("Expand to reveal detected Position as text and cropped image")
            col1, col2, col3 = st.columns(3)

            with col1:
                addressExpander = st.expander("Expand to reveal detected Address as text and cropped image")
            with col2:
                mobileExpander = st.expander("Expand to reveal detected Mobile as text and cropped image")
            with col3:
                phoneNumberExpander = st.expander("Expand to reveal detected Phone Number as text and cropped image")

            col1, col2, col3 = st.columns(3)
            with col1:
                emailExpander = st.expander("Expand to reveal detected Email as text and cropped image")
            with col2:
                faxExpander = st.expander("Expand to reveal detected Fax as text and cropped image")
            with col3:
                urlExpander = st.expander("Expand to reveal detected URL as text and cropped image")

            with labelExpander:
                st.image(config['result']['labelled_img'])

            with fullNameExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['fullName']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_full_name'])
            with companyNameExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['companyName']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_company_name'])
            with positionExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['positionName']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_position_name'])
            with addressExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['address']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_address'])
            with mobileExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['mobile']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_mobile'])
            with phoneNumberExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['phoneNumber']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_phone_no'])
            with emailExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['email']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_email'])
            with faxExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['fax']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_fax'])
            with urlExpander:
                text = f'''<p style="font-family:{ja_font}; color:Green; font-size: {font_size}px;">{recognizedLabels['url']}</p>'''
                st.markdown(text, unsafe_allow_html=True)
                st.image(config['result']['crop_url'])
                
            jsonExpander = st.expander("Show and Download the results in JSON",expanded=True)

            with jsonExpander:
                col1, col2 = st.columns(2)
                with col1:
                    with open(config['result']['dummy_json_path']) as f:
                        jsonFile = f.read()

                    labelledBusinessCard = io.imread(config['result']['labelled_img'])
                    im_height = labelledBusinessCard.shape[0]
                    im_width = labelledBusinessCard.shape[1]
                    labelsDict = getBoundingBoxPos(im_width, im_height)

                    jsonResult = json.loads(jsonFile)

                    jsonResult['Business Card'][0]["Full Name"] = jOCR.detections['fullName']
                    jsonResult['Business Card'][0]["Company Name"] = jOCR.detections['companyName']
                    jsonResult['Business Card'][0]["Position Name"] = jOCR.detections['positionName']
                    jsonResult['Business Card'][0]["Address"] = jOCR.detections['address']
                    jsonResult['Business Card'][0]["Mobile"] = jOCR.detections['mobile']
                    jsonResult['Business Card'][0]["Phone Number"] = jOCR.detections['phoneNumber']
                    jsonResult['Business Card'][0]["Email"] = jOCR.detections['email']
                    jsonResult['Business Card'][0]["Fax"] = jOCR.detections['fax']
                    jsonResult['Business Card'][0]["URL"] = jOCR.detections['url']
                    jsonResult['Business Card'][0]["Bounding Box"]["Full Name Location"]["left"] = \
                        labelsDict['fullName'][
                            'left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Full Name Location"]["top"] = \
                        labelsDict['fullName']['top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Full Name Location"]["right"] = \
                        labelsDict['fullName'][
                            'right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Full Name Location"]["bottom"] = \
                        labelsDict['fullName'][
                            'bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Full Name Location"]["probability"] = \
                        labelsDict['fullName']['probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Company Name Location"]["left"] = \
                        labelsDict['companyName']['left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Company Name Location"]["top"] = \
                        labelsDict['companyName']['top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Company Name Location"]["right"] = \
                        labelsDict['companyName']['right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Company Name Location"]["bottom"] = \
                        labelsDict['companyName']['bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Company Name Location"]["probability"] = \
                        labelsDict['companyName']['probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Position Name Location"]["left"] = \
                        labelsDict['positionName']['left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Position Name Location"]["top"] = \
                        labelsDict['positionName']['top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Position Name Location"]["right"] = \
                        labelsDict['positionName']['right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Position Name Location"]["bottom"] = \
                        labelsDict['positionName']['bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Position Name Location"]["probability"] = \
                        labelsDict['positionName']['probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Address Location"]["left"] = labelsDict['address'][
                        'left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Address Location"]["top"] = labelsDict['address'][
                        'top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Address Location"]["right"] = labelsDict['address'][
                        'right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Address Location"]["bottom"] = \
                        labelsDict['address']['bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Address Location"]["probability"] = \
                        labelsDict['address']['probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Mobile Location"]["left"] = labelsDict['mobile'][
                        'left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Mobile Location"]["top"] = labelsDict['mobile'][
                        'top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Mobile Location"]["right"] = labelsDict['mobile'][
                        'right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Mobile Location"]["bottom"] = labelsDict['mobile'][
                        'bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Mobile Location"]["probability"] = \
                        labelsDict['mobile']['probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Phone Number Location"]["left"] = \
                        labelsDict['phoneNumber'][
                            'left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Phone Number Location"]["top"] = \
                        labelsDict['phoneNumber'][
                            'top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Phone Number Location"]["right"] = \
                        labelsDict['phoneNumber']['right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Phone Number Location"]["bottom"] = \
                        labelsDict['phoneNumber']['bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Phone Number Location"]["probability"] = \
                        labelsDict['phoneNumber']['probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Email Location"]["left"] = labelsDict['email'][
                        'left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Email Location"]["top"] = labelsDict['email']['top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Email Location"]["right"] = labelsDict['email'][
                        'right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Email Location"]["bottom"] = labelsDict['email'][
                        'bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Email Location"]["probability"] = \
                        labelsDict['email'][
                            'probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["Fax Location"]["left"] = labelsDict['fax']['left']
                    jsonResult['Business Card'][0]["Bounding Box"]["Fax Location"]["top"] = labelsDict['fax']['top']
                    jsonResult['Business Card'][0]["Bounding Box"]["Fax Location"]["right"] = labelsDict['fax']['right']
                    jsonResult['Business Card'][0]["Bounding Box"]["Fax Location"]["bottom"] = labelsDict['fax'][
                        'bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["Fax Location"]["probability"] = labelsDict['fax'][
                        'probability']

                    jsonResult['Business Card'][0]["Bounding Box"]["URL Location"]["left"] = labelsDict['url']['left']
                    jsonResult['Business Card'][0]["Bounding Box"]["URL Location"]["top"] = labelsDict['url']['top']
                    jsonResult['Business Card'][0]["Bounding Box"]["URL Location"]["right"] = labelsDict['url']['right']
                    jsonResult['Business Card'][0]["Bounding Box"]["URL Location"]["bottom"] = labelsDict['url'][
                        'bottom']
                    jsonResult['Business Card'][0]["Bounding Box"]["URL Location"]["probability"] = labelsDict['url'][
                        'probability']

                    if os.path.exists(config['result']['json_result_path']):
                        shutil.rmtree(config['result']['json_result_path'] + "/")
                        write_json(config['result']['json_result_path'], 'businessCardLabels.json', jsonResult)
                    else:
                        write_json(config['result']['json_result_path'], 'businessCardLabels.json', jsonResult)

                    with open(config['result']['json_result_path'] + "/businessCardLabels.json") as f:
                        jsonFile = f.read()

                    jsonResult = json.loads(jsonFile)
                    st.json(jsonResult)
                    with open(config['result']['json_result_path'] + "/businessCardLabels.json", "rb") as file:
                        btn = st.sidebar.download_button(
                            label="Download Result in JSON",
                            data=file,
                            file_name="businessCardLabels.json")
                with col2:
                    st.subheader("Reference for (left,top) and (right,bottom) for a label")
                    st.image("./resources/positionReference.png")
    else:
        st.info("First Detect the fragments of Japanese Business Card using Yolo-V5 model. Therefore, first Navigate to 'Detect and Fragment Labels' from Sidebar and then come back here.")
