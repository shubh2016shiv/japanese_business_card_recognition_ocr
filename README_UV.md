# Japanese Business Card Recognition - Modern Setup with UV

This project has been modernized to use UV package manager for better dependency management and Python environment handling.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (managed by UV)
- UV package manager installed

### Installation & Running

1. **Install UV** (if not already installed):
   ```bash
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Run the application**:
   ```bash
   # Option 1: Use the launcher script
   python run_app.py
   
   # Option 2: Use UV directly
   uv run streamlit run app.py
   ```

3. **Access the app**: Open your browser to `http://localhost:8501`

## 📦 Project Structure

```
japanese_business_card_recognition_ocr/
├── app.py                 # Main Streamlit application
├── detect.py             # YOLOv5 detection script
├── japaneseOCR.py        # OCR processing module
├── config.ini            # Configuration file
├── pyproject.toml        # Modern Python project configuration
├── requirements.txt      # Legacy requirements (for compatibility)
├── run_app.py           # Modern launcher script
├── .uvrc                # UV configuration
├── models/              # YOLOv5 model files
├── utils/               # Utility modules
└── resources/           # Data and model files
```

## 🔧 Development

### Adding Dependencies
```bash
# Add a new dependency
uv add package_name

# Add a development dependency
uv add --dev package_name

# Sync dependencies
uv sync
```

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

## 🆕 What's New

### Modernized Features:
- ✅ **UV Package Manager**: Fast, reliable dependency management
- ✅ **Python 3.11**: Modern Python version with better performance
- ✅ **Updated Dependencies**: All packages updated to latest compatible versions
- ✅ **Cross-Platform Paths**: Fixed hard-coded paths for Windows compatibility
- ✅ **Modern Streamlit**: Updated deprecated decorators (`@st.cache_data`)
- ✅ **Better Error Handling**: Improved error messages and handling
- ✅ **Clean Project Structure**: Organized with proper configuration files

### Key Improvements:
1. **Dependency Management**: UV provides faster, more reliable package installation
2. **Environment Isolation**: Automatic virtual environment management
3. **Cross-Platform**: Works seamlessly on Windows, macOS, and Linux
4. **Modern Python**: Uses Python 3.11 with all latest features
5. **Updated Libraries**: All dependencies updated to latest stable versions
6. **Better Performance**: Faster startup and execution times

## 🐛 Troubleshooting

### Common Issues:

1. **UV not found**:
   ```bash
   # Reinstall UV
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Permission errors on Windows**:
   ```bash
   # Run PowerShell as Administrator
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Port already in use**:
   ```bash
   # Use a different port
   uv run streamlit run app.py --server.port=8502
   ```

## 📚 Original Project

This is a modernization of the original Japanese Business Card Recognition project that uses:
- YOLOv5 for object detection and label extraction
- EasyOCR for Japanese and English text recognition
- Streamlit for the web interface

The core functionality remains the same, but now with modern tooling and better maintainability.
