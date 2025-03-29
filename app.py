import streamlit as st
import os
from PIL import Image
import io
import time
import numpy as np
import google.generativeai as genai
import requests
import json
import pandas as pd
import datetime

# --- Safe OpenCV Import ---
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available - blur detection disabled", icon="⚠️")

# --- Constants ---
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    # ... (keep your existing class list)
]

# --- Page Configuration ---
st.set_page_config(
    page_title="🌿 AI Plant Health Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title Section ---
st.title("🌿 AI Plant Health Assistant")
st.caption("Upload plant images to identify diseases, pests, or weeds with weather context")

# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API configuration error: {str(e)}")
    st.stop()

# --- Model Initialization ---
@st.cache_resource
def load_models():
    try:
        return (
            genai.GenerativeModel('gemini-1.5-flash-latest'),
            genai.GenerativeModel('gemini-1.5-flash-latest')
        )
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.stop()

vision_model, text_model = load_models()

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/150", width=150)  # Replace with your logo
    user_city = st.text_input("📍 Enter City for Weather", help="For weather-based recommendations")
    enable_blur_check = st.toggle("Enable Blur Detection", CV2_AVAILABLE, disabled=not CV2_AVAILABLE)

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose plant images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Max 5MB per image"
)

# --- Helper Functions ---
def check_blurriness(image_bytes, threshold=100.0):
    if not CV2_AVAILABLE or not enable_blur_check:
        return 0.0, False
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return 0.0, False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance, variance < threshold
    except Exception:
        return 0.0, False

def get_weather_data(city, api_key):
    if not city:
        return {"data": None, "error": None}
    
    try:
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric",
            timeout=10
        )
        data = response.json()
        if data.get("cod") != 200:
            return {"data": None, "error": data.get("message", "Weather data unavailable")}
            
        return {
            "data": {
                "city": data.get("name"),
                "temp": data["main"].get("temp"),
                "humidity": data["main"].get("humidity"),
                "description": data["weather"][0].get("description"),
                "icon": data["weather"][0].get("icon")
            },
            "error": None
        }
    except Exception:
        return {"data": None, "error": "Weather service unavailable"}

# ... (keep your existing get_plant_identification and get_issue_details functions)

# --- Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# --- Main Analysis Flow ---
if st.button("🔍 Analyze Images", type="primary", disabled=not uploaded_files):
    st.session_state.analysis_results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    
    # Weather data
    weather_info = {"data": None, "error": None}
    if user_city:
        weather_info = get_weather_data(user_city, OPENWEATHER_API_KEY)
    
    # Process images
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress, text=f"Processing image {i + 1} of {len(uploaded_files)}...")
        
        result = {
            "weather": weather_info,
            "blur_info": check_blurriness(uploaded_file.getvalue()),
            "identification": get_plant_identification(uploaded_file.getvalue(), uploaded_file.name),
            "details": None
        }
        
        if result["identification"]["identified_issue"] not in ["Unknown/Not Plant", "Error", "Healthy Plant"]:
            result["details"] = get_issue_details(result["identification"]["identified_issue"])
        
        st.session_state.analysis_results[uploaded_file.name] = result
    
    progress_bar.empty()
    st.success("Analysis complete!", icon="✅")

# --- Results Display ---
if st.session_state.analysis_results:
    for filename, result in st.session_state.analysis_results.items():
        with st.expander(f"🌱 {filename}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(Image.open(io.BytesIO(uploaded_files[0].getvalue())), width=300)
                if result["blur_info"][1]:
                    st.warning(f"⚠️ Possible blur (Variance: {result['blur_info'][0]:.1f})")
                if result["weather"]["data"]:
                    st.subheader("🌤️ Weather")
                    st.write(f"{result['weather']['data']['city']}: {result['weather']['data']['description'].capitalize()}")
                    st.write(f"Temp: {result['weather']['data']['temp']}°C")
                    st.write(f"Humidity: {result['weather']['data']['humidity']}%")
            
            with col2:
                ident = result["identification"]
                if ident.get("error"):
                    st.error("Identification failed")
                elif ident["identified_issue"] == "Healthy Plant":
                    st.success("✅ Healthy Plant")
                else:
                    st.warning(f"⚠️ {ident['identified_issue']}")
                
                if result["details"]:
                    st.markdown("---")
                    st.markdown(result["details"]["details"])

# --- Empty State ---
elif not uploaded_files:
    st.info("👋 Upload plant images to get started")
