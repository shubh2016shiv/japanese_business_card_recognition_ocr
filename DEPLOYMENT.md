# Streamlit Cloud Deployment Guide

This guide explains how to deploy the Japanese Business Card Recognition app on Streamlit Cloud.

## Prerequisites

1. **GitHub Repository**: Your code must be in a public GitHub repository
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Required Files**: Ensure all deployment files are in the repository root

## Required Files for Deployment

The following files must be present in your repository root:

### 1. Main Application
- `app.py` - Main Streamlit application file

### 2. Dependencies
- `requirements.txt` - Python package dependencies with pinned versions
- `packages.txt` - System-level dependencies for OpenCV and EasyOCR

### 3. Configuration
- `.streamlit/config.toml` - Streamlit configuration for cloud deployment

### 4. Project Files
- `detect.py` - YOLOv5 detection script
- `japaneseOCR.py` - OCR processing module
- `config.ini` - Application configuration
- `models/` - YOLOv5 model files
- `utils/` - Utility modules
- `resources/` - Data and model files

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository structure looks like this:
```
japanese_business_card_recognition_ocr/
├── app.py
├── detect.py
├── japaneseOCR.py
├── config.ini
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── models/
├── utils/
└── resources/
```

### 2. Deploy on Streamlit Cloud

1. **Sign in** to [share.streamlit.io](https://share.streamlit.io)
2. **Click "New app"**
3. **Fill in the details**:
   - **Repository**: Select your GitHub repository
   - **Branch**: Choose `main` or `master`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (optional)
4. **Click "Deploy"**

### 3. Monitor Deployment

- The deployment process will take 2-5 minutes
- Check the logs for any errors
- The app will be available at `https://your-app-name.streamlit.app`

## Configuration Details

### requirements.txt
Contains pinned versions of all Python dependencies for stability:
```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
easyocr==1.7.0
opencv-python-headless==4.8.1.78
# ... and more
```

### packages.txt
System dependencies required by OpenCV and EasyOCR:
```
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
```

### .streamlit/config.toml
Streamlit configuration optimized for cloud deployment:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "light"
```

## Troubleshooting

### Common Issues

1. **App fails to start**
   - Check that `app.py` is in the repository root
   - Verify all dependencies are listed in `requirements.txt`
   - Check the deployment logs for specific errors

2. **OCR models not loading**
   - Ensure the models directory exists
   - Check that model files are included in the repository
   - Verify the model download URLs are accessible

3. **YOLO detection not working**
   - Ensure `detect.py` is in the repository root
   - Check that model weights are in `resources/yolo_model_weights/`
   - Verify the detection command uses `python` (not absolute paths)

4. **Memory issues**
   - The app uses large ML models (YOLOv5 + EasyOCR)
   - Streamlit Cloud provides limited memory
   - Consider optimizing model loading or using smaller models

### Performance Optimization

1. **Model Caching**: The app uses `@st.cache_resource` to cache the OCR model
2. **Image Processing**: Images are processed efficiently using OpenCV
3. **Memory Management**: Garbage collection is used after OCR processing

## Environment Variables

Currently, no environment variables are required. All configuration is handled through `config.ini`.

## PyTorch Version Compatibility

This app uses YOLOv5 models that were trained with older PyTorch versions. The code includes `weights_only=False` in `torch.load()` to maintain compatibility with PyTorch 2.6+.

**Security Note**: The `weights_only=False` parameter is safe here because we trust our own model files. Only use models from trusted sources.

## Known Limitations

1. **File Upload Size**: Streamlit Cloud has limits on file upload sizes
2. **Memory Usage**: Large models may cause memory issues
3. **Processing Time**: OCR and detection may take time for large images
4. **Concurrent Users**: Performance may degrade with many simultaneous users

## Support

If you encounter issues:

1. Check the Streamlit Cloud logs
2. Test the app locally first
3. Verify all dependencies are correctly specified
4. Ensure all required files are in the repository

## Updates

To update your deployed app:

1. Push changes to your GitHub repository
2. Streamlit Cloud will automatically redeploy
3. Check the deployment status in your Streamlit dashboard

The app will be updated with your latest changes within a few minutes.
