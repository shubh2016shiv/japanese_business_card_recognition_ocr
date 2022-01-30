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

st.set_page_config(layout="wide")
st.title("Project - Business Card Recognition Challenge")


nav_option = st.sidebar.selectbox(label="Navigation", options=['Home', "Detect Labels", "Perform Japanese OCR"])
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
def get_ocr_model():
    reader = easyocr.Reader(['en', 'ja'], model_storage_directory=config['downloads']['ocr_models_dir'],gpu=False,quantize=False,download_enabled=False)
    return reader

def recognize(ocr_model_,extracted_img_):
    try:
        img = Image.open(extracted_img_)
        if img is not None:
            result = ocr_model_.readtext(img,decoder='beamsearch',beamWidth=1)
            return result
    except Exception as e:
        st.error("Failed to detect any text from image due to: ", e)
        return None

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

    # Project Artifacts Downloads
    st.subheader("Download Project Artifacts")

    if not os.path.isdir(config['downloads']['downloads_folder']):
        os.makedirs(config['downloads']['downloads_folder'])

    download_option = st.selectbox(label="Download Options",
                                   options=["Show Training Logs",
                                            "Jupyter Notebook: Read, Process and Train YOLO-V5 model",
                                            "Set of Business Cards (.zip)",
                                            "Best Yolo-V5 Model",
                                            "Japanese OCR model"])

    if download_option == "Show Training Logs":
        st.markdown('[Complete Training Logs (click here 🔗).](https://wandb.ai/aimagic/sansan-business-card-recognition/runs/2a8sztxh/overview?workspace=user-aimagic)')

    elif download_option == "Jupyter Notebook: Read, Process and Train YOLO-V5 model":
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
elif nav_option == "Detect Labels":
    st.subheader("Labels Identification on Digital Business Card")
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
        st.subheader("Detected Labels and their Scores")
        st.image(config['result']['labelled_img'])

elif nav_option == "Perform Japanese OCR":
    ocr_model = get_ocr_model()
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

        for i in stqdm(range(len(extracted_label_imgs))):
            st.write(recognize(ocr_model, extracted_label_imgs[i]))
        #recognize(ocr_model, config['result']['crop_address'])
