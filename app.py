import streamlit as st
import os
from PIL import Image
import io
import time
import cv2
import numpy as np
import google.generativeai as genai
import requests
import json
import pandas as pd
import datetime
from streamlit_extras.colored_header import colored_header
from streamlit_extras.badges import badge
from streamlit_extras.metric_cards import style_metric_cards

# --- Constants ---
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", 
    "Tomato___healthy"
]

# --- Page Configuration ---
st.set_page_config(
    page_title="🌿 AI Plant Health Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AI-powered plant health diagnosis tool"
    }
)

# --- Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Create a style.css file with your custom styles

# --- Title Section with Gradient ---
colored_header(
    label="🌿 AI Plant Health Assistant",
    description="Upload plant images to identify diseases, pests, or weeds with weather context",
    color_name="blue-70",
)

# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    if not GOOGLE_API_KEY or not OPENWEATHER_API_KEY:
        st.error("API keys not properly configured. Please check your secrets.toml file.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API configuration error: {str(e)}")
    st.stop()

# --- Model Initialization ---
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

@st.cache_resource
def load_models():
    try:
        vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        text_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return vision_model, text_model
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        st.stop()

vision_model, text_model = load_models()

# --- Sidebar ---
with st.sidebar:
    st.image("plant_logo.png", width=150)  # Add your logo image
    st.title("Settings")
    
    # Location Input with nice icon
    with st.container(border=True):
        user_city = st.text_input(
            "📍 Enter City for Weather", 
            help="Provide your city for weather-based recommendations"
        )
    
    # Analysis Options
    with st.expander("⚙️ Analysis Options", expanded=True):
        max_images = st.slider("Max images to process", 1, 10, 5)
        enable_blur_check = st.toggle("Enable Blur Detection", True)
        enable_weather = st.toggle("Enable Weather Context", True)
    
    # Info Section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Upload** plant images (max 5MB each)
        2. **Enter** your location (optional)
        3. **Click** Analyze Images
        4. **Review** results in tabs
        5. **Provide** feedback to improve
        """)
    
    # Feedback Summary
    if 'feedback' in st.session_state and st.session_state.feedback:
        st.divider()
        st.subheader("📊 Session Feedback")
        feedback_df = pd.DataFrame.from_dict(st.session_state.feedback, orient='index')
        st.dataframe(feedback_df, use_container_width=True)
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit and Google Gemini")
    badge(type="github", name="your-repo/plant-health")

# --- File Uploader with Enhanced UI ---
upload_container = st.container(border=True)
with upload_container:
    st.subheader("📤 Upload Plant Images")
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more images of plants (max 5MB each)"
    )

# --- Image Preview Gallery ---
if uploaded_files:
    with st.expander("🖼️ Image Preview", expanded=True):
        cols = st.columns(min(4, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files[:4]):
            with cols[i % 4]:
                st.image(uploaded_file, caption=uploaded_file.name[:20] + ("..." if len(uploaded_file.name) > 20 else ""), use_column_width=True)

# --- Analysis Button with Status ---
if uploaded_files:
    analyze_col, status_col = st.columns([1, 3])
    with analyze_col:
        analyze_button = st.button("🔍 Analyze Images", type="primary", use_container_width=True)
    with status_col:
        if analyze_button:
            status_msg = st.empty()
            status_msg.info("Starting analysis...")

# --- Helper Functions ---
def validate_image(image_bytes, max_size_mb=5):
    """Validate image size and type."""
    if len(image_bytes) > max_size_mb * 1024 * 1024:
        return False, f"Image exceeds maximum size of {max_size_mb}MB"
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True, ""
    except Exception:
        return False, "Invalid image file"

def check_blurriness(image_bytes, threshold=100.0):
    """Calculate Laplacian variance for blur detection."""
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
    """Fetch current weather data with better error handling."""
    if not city:
        return {"data": None, "error": None}
    
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city}&units=metric"
    
    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("cod") != 200:
            return {"data": None, "error": data.get("message", "Weather data unavailable")}
            
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        
        return {
            "data": {
                "city": data.get("name"),
                "temp": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "description": weather.get("description"),
                "icon": weather.get("icon"),
                "wind_speed": wind.get("speed"),
            },
            "error": None
        }
    except requests.exceptions.RequestException:
        return {"data": None, "error": "Weather service unavailable"}
    except Exception:
        return {"data": None, "error": "Error processing weather data"}

def display_weather_card(weather_data):
    """Display weather information in a styled card."""
    if not weather_data or "error" in weather_data:
        return
    
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            if weather_data.get('icon'):
                st.image(f"http://openweathermap.org/img/wn/{weather_data['icon']}@2x.png", width=80)
        
        with cols[1]:
            st.markdown(f"""
            **{weather_data.get('city', 'Location')}**  
            🌡️ **Temp:** {weather_data.get('temp', 'N/A')}°C  
            💧 **Humidity:** {weather_data.get('humidity', 'N/A')}%  
            🌬️ **Wind:** {weather_data.get('wind_speed', 'N/A')} m/s  
            *{weather_data.get('description', '').capitalize()}*
            """)

def get_plant_identification(image_bytes, filename):
    """Call Gemini Vision to identify plant issues."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = f"""Analyze this plant image and identify the most likely issue:
        1. Disease? Use PlantVillage format (e.g., 'Tomato___Late_blight')
        2. Pest? Name it (e.g., 'Aphids')
        3. Weed? Name it (e.g., 'Dandelion')
        4. If healthy, state 'Healthy Plant'
        5. If unknown, state 'Unknown/Not Plant'
        
        Return ONLY the identification name. No explanations.
        """
        response = vision_model.generate_content([prompt, img], safety_settings=safety_settings)
        response.resolve()
        return {
            "identified_issue": response.text.strip(),
            "error": None
        }
    except Exception as e:
        return {
            "identified_issue": "Error",
            "error": str(e)
        }

def get_issue_details(issue_name):
    """Get detailed information about the identified issue."""
    if not issue_name or issue_name in ["Unknown/Not Plant", "Error", "Healthy Plant"]:
        return {"details": "No details available for this classification.", "error": None}

    try:
        prompt = f"""Provide detailed information about: "{issue_name}".
        Include:
        1. **Type** (Disease/Pest/Weed)
        2. **Symptoms**
        3. **Causes**
        4. **Affected Plants**
        5. **Treatment Options**
        6. **Prevention Tips**
        
        Use Markdown formatting with bullet points.
        """
        response = text_model.generate_content(prompt, safety_settings=safety_settings)
        response.resolve()
        return {
            "details": response.text,
            "error": None
        }
    except Exception as e:
        return {
            "details": "Error retrieving details.",
            "error": str(e)
        }

def save_feedback(filename, feedback, details=""):
    """Save feedback to a CSV file."""
    try:
        feedback_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "filename": filename,
            "feedback": feedback,
            "details": details
        }
        
        # Create or append to feedback file
        if not os.path.exists("feedback.csv"):
            pd.DataFrame([feedback_data]).to_csv("feedback.csv", index=False)
        else:
            pd.DataFrame([feedback_data]).to_csv("feedback.csv", mode='a', header=False, index=False)
    except Exception:
        pass  # Silently fail if feedback can't be saved

# --- Analysis Execution ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

if analyze_button and uploaded_files:
    st.session_state.analysis_results = {}
    start_time_total = time.time()
    
    # Limit number of images to process
    files_to_process = uploaded_files[:max_images]
    num_files = len(files_to_process)
    
    # Initialize progress bar
    progress_bar = st.progress(0, text=f"Processing 1 of {num_files} images...")
    
    # Fetch weather data once if enabled
    weather_info = {"data": None, "error": None}
    if enable_weather and user_city:
        weather_info = get_weather_data(user_city, OPENWEATHER_API_KEY)
        if weather_info["error"]:
            st.warning(f"Weather data unavailable: {weather_info['error']}")
    
    # Process each image
    for i, uploaded_file in enumerate(files_to_process):
        filename = uploaded_file.name
        image_bytes = uploaded_file.getvalue()
        current_result = {"weather": weather_info}
        
        # Update progress
        progress = (i + 1) / num_files
        progress_bar.progress(progress, text=f"Processing {i + 1} of {num_files} images...")
        
        # Validate image
        is_valid, validation_msg = validate_image(image_bytes)
        if not is_valid:
            current_result["error"] = validation_msg
            st.session_state.analysis_results[filename] = current_result
            continue
        
        # Blur check if enabled
        if enable_blur_check:
            blur_variance, is_blurry = check_blurriness(image_bytes)
            current_result["blur_info"] = {"variance": blur_variance, "is_blurry": is_blurry}
        
        # Identification
        identification_result = get_plant_identification(image_bytes, filename)
        current_result["identification"] = identification_result
        identified_issue = identification_result.get("identified_issue", "Error")
        
        # Get details
        if identified_issue not in ["Unknown/Not Plant", "Error", "Healthy Plant"]:
            details_result = get_issue_details(identified_issue)
        else:
            details_result = {"details": "No details available.", "error": None}
        current_result["details"] = details_result
        
        # Store result
        st.session_state.analysis_results[filename] = current_result
    
    progress_bar.empty()
    total_time = time.time() - start_time_total
    st.toast(f"Analysis completed in {total_time:.1f} seconds!", icon="✅")

# --- Display Results ---
if st.session_state.analysis_results:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Create tabs for each image
    tab_titles = [f"{i+1}. {fname[:15]}{'...' if len(fname)>15 else ''}" 
                 for i, fname in enumerate(st.session_state.analysis_results.keys())]
    tabs = st.tabs(tab_titles)
    
    for i, tab in enumerate(tabs):
        with tab:
            filename = list(st.session_state.analysis_results.keys())[i]
            result = st.session_state.analysis_results[filename]
            
            # Layout columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Image display
                st.image(
                    Image.open(io.BytesIO(uploaded_files[i].getvalue())),
                    caption=filename,
                    use_column_width=True
                )
                
                # Blur warning
                if result.get("blur_info", {}).get("is_blurry"):
                    st.warning(
                        f"⚠️ Image may be blurry (Variance: {result['blur_info']['variance']:.1f})",
                        icon="📸"
                    )
                
                # Weather card
                if result.get("weather", {}).get("data"):
                    display_weather_card(result["weather"]["data"])
            
            with col2:
                # Identification result
                ident = result.get("identification", {})
                issue = ident.get("identified_issue", "Error")
                
                if ident.get("error"):
                    st.error(f"❌ Identification Error: {ident['error']}")
                elif issue == "Unknown/Not Plant":
                    st.info("🌱 Unknown / Not a Plant", icon="❓")
                elif issue == "Healthy Plant":
                    st.success("🌿 Healthy Plant", icon="✅")
                elif issue == "Error":
                    st.error("❗ Identification Failed", icon="❌")
                else:
                    st.warning(f"⚠️ Identified: **{issue}**", icon="🔍")
                
                # Details section
                st.markdown("---")
                st.subheader("📝 Details & Recommendations")
                
                details = result.get("details", {})
                if details.get("error"):
                    st.error(f"Details error: {details['error']}")
                else:
                    st.markdown(details.get("details", "No details available."))
                
                # Feedback section
                st.markdown("---")
                st.subheader("💬 Feedback")
                
                feedback_key = f"feedback_{filename}"
                current_feedback = st.session_state.feedback.get(filename, "Not Rated")
                
                feedback = st.radio(
                    "How accurate was this analysis?",
                    options=["Not Rated", "Very Accurate", "Somewhat Accurate", "Inaccurate"],
                    index=["Not Rated", "Very Accurate", "Somewhat Accurate", "Inaccurate"].index(current_feedback) 
                    if current_feedback in ["Very Accurate", "Somewhat Accurate", "Inaccurate"] else 0,
                    key=feedback_key,
                    horizontal=True
                )
                
                if feedback != current_feedback:
                    st.session_state.feedback[filename] = feedback
                    save_feedback(filename, feedback)
                    
                    if feedback != "Not Rated":
                        st.toast("Thanks for your feedback!", icon="👍")

# --- Empty State ---
elif not uploaded_files:
    st.markdown("---")
    with st.container(border=True):
        st.subheader("👋 Get Started")
        st.markdown("""
        1. Upload one or more plant images using the uploader above
        2. Optionally enter your location for weather context
        3. Click the "Analyze Images" button
        """)
        st.image("plant_placeholder.png", width=300)  # Add a placeholder image

# --- What to Do Next ---
"""
## Next Steps with This Code:

1. **Create the supporting files**:
   - Add a `style.css` file for custom styling
   - Add a logo image (`plant_logo.png`)
   - Add a placeholder image (`plant_placeholder.png`)

2. **Deployment options**:
   - Deploy on Streamlit Community Cloud
   - Set up a Docker container for local deployment
   - Consider AWS/Azure/GCP for production deployment

3. **Enhancements to consider**:
   - Add user authentication
   - Implement a database for persistent feedback storage
   - Add multilingual support
   - Create PDF report generation
   - Add historical analysis tracking

4. **Testing**:
   - Write unit tests for helper functions
   - Perform user testing with real plant images
   - Monitor API usage and costs

5. **Maintenance**:
   - Set up error tracking (Sentry, etc.)
   - Monitor performance metrics
   - Regularly update the PlantVillage classes list

The improved version includes:
- Better visual design with cards and colored headers
- Enhanced user experience with progress tracking
- More robust error handling
- Improved feedback system
- Better weather display
- Responsive layout
- Performance optimizations
"""
