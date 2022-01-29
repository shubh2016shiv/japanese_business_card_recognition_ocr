import streamlit as st
from configparser import ConfigParser
import gdown
import os

st.set_page_config(layout="wide")
st.title("Project - Business Card Recognition Challenge")
st.subheader("Challenge presented by Sansan Global PTE. LTD. to Recognize labellings on Japanese business card.")

nav_option = st.sidebar.selectbox(label="Navigation", options=['Home', "Detect Labels", "Perform Japanese OCR"])
config = ConfigParser()
config.read('config.ini')

if nav_option == "Home":

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



