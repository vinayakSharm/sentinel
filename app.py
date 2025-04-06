import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import ViltProcessor, ViltForQuestionAnswering
import time
from io import BytesIO
import threading
import queue
import os
import tempfile
from datetime import datetime

# Set page config to wide mode
st.set_page_config(
    layout="wide", 
    page_title="Securade.ai Sentinel",
    page_icon="🔍",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium, polished UI
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 1.5rem;
        max-width: 1200px;
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
        border-radius: 12px;
        margin-top: 1rem;
    }
    
    h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 300;
        margin-bottom: 1.5rem;
    }
    
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: #f1f5f9;
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0), rgba(56, 189, 248, 0.6), rgba(56, 189, 248, 0));
    }
    
    .video-card {
        padding: 0.7rem;
        background: rgba(15, 23, 42, 0.95);
    }
    
    .video-card-content {
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 8px;
        overflow: hidden;
        background: #0f172a;
    }
    
    .card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #0ea5e9, #3b82f6);
        color: white;
        font-weight: 500;
        border: none;
        height: 2.6rem;
        width: 100%;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
        transform: translateY(-1px);
    }
    
    .stop-button button {
        background: linear-gradient(90deg, #ef4444, #f97316);
    }
    
    input[type="text"], .stSelectbox {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(71, 85, 105, 0.5);
        color: #f8fafc;
        border-radius: 8px;
    }
    
    .caption-container {
        height: 220px;
        overflow-y: auto;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(71, 85, 105, 0.3);
        padding: 1rem;
        margin-top: 0.5rem;
        scrollbar-width: thin;
        scrollbar-color: #475569 #1e293b;
    }
    
    .caption-item {
        margin-bottom: 12px;
        line-height: 1.4;
        border-left: 2px solid #3b82f6;
        padding-left: 10px;
        background: rgba(30, 41, 59, 0.7);
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
    }
    
    .caption-item:last-child {
        margin-bottom: 0;
    }
    
    .timestamp {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .status-badge {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    .status-badge.live::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        background-color: #10b981;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    .device-badge {
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.7);
        padding: 6px 12px;
        border-radius: 8px;
        border: 1px solid rgba(71, 85, 105, 0.3);
    }
    
    .source-select {
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    .stSpinner > div {
        border-width: 2px;
        border-color: #38bdf8 #1e293b #1e293b #1e293b !important;
    }
    
    .css-18e3th9 {
        padding-top: 0;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Question answer styling */
    .answer-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 10px 12px;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* App logo */
    .app-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .logo-icon {
        font-size: 2.6rem;
        margin-right: 0.7rem;
        background: linear-gradient(45deg, #0ea5e9, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .app-header-container {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_state():
    if 'initialized' not in st.session_state:
        st.session_state.frame = None
        st.session_state.captions = []
        st.session_state.stop_event = threading.Event()
        st.session_state.frame_queue = queue.Queue(maxsize=1)
        st.session_state.caption_queue = queue.Queue(maxsize=10)
        st.session_state.processor = None
        st.session_state.thread = None
        st.session_state.is_streaming = False
        st.session_state.initialized = True

@st.cache_resource
def load_processor():
    class VideoProcessor:
        def __init__(self):
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            self.vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            
            # Check for available devices
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            
            self.caption_model.to(self.device)
            self.vqa_model.to(self.device)

        def generate_caption(self, image):
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            output = self.caption_model.generate(**inputs, max_new_tokens=50)
            return self.caption_processor.decode(output[0], skip_special_tokens=True)

        def answer_question(self, image, question):
            inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
            outputs = self.vqa_model(**inputs)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            return self.vqa_model.config.id2label[idx]

    return VideoProcessor()

def get_video_source(source_type, source_path=None):
    if source_type == "Webcam":
        return cv2.VideoCapture(0)
    elif source_type == "Video File" and source_path:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'temp_video.mp4')
        with open(temp_path, 'wb') as f:
            f.write(source_path.getvalue())
        return cv2.VideoCapture(temp_path)
    elif source_type == "RTSP Stream" and source_path:
        return cv2.VideoCapture(source_path)
    return None

def process_video(stop_event, frame_queue, caption_queue, processor, source_type, source_path=None):
    cap = get_video_source(source_type, source_path)
    last_caption_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        current_time = time.time()

        # Generate caption every 3 seconds
        if current_time - last_caption_time >= 3.0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = processor.generate_caption(img)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            try:
                if caption_queue.full():
                    caption_queue.get_nowait()
                caption_queue.put_nowait({'timestamp': timestamp, 'caption': caption})
                last_caption_time = current_time
            except queue.Full:
                pass

        try:
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.03)

    cap.release()

def main():
    initialize_state()
    apply_custom_css()
    
    # Elegant app header
    st.markdown("""
    <div class="app-header-container">
        <div class="app-logo">
            <div class="logo-icon">🔍</div>
            <h1>Securade.ai Sentinel</h1>
        </div>
        <p class="app-subtitle">Advanced AI-Powered Video Surveillance & Scene Understanding</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a clean 2-column layout
    col1, col2 = st.columns([1, 1.2])
    
    # Column 1: Video feed
    with col1:
        # Video feed card
        st.markdown('<div class="premium-card video-card">', unsafe_allow_html=True)
        
        # Header with status
        header_content = '<div class="card-header">'
        header_content += '📹 Live Surveillance'
        if st.session_state.is_streaming:
            header_content += '<span class="status-badge live">LIVE</span>'
        header_content += '</div>'
        st.markdown(header_content, unsafe_allow_html=True)
        
        # Video display
        st.markdown('<div class="video-card-content">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Controls card
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🎮 Controls</div>', unsafe_allow_html=True)
        
        # Source selection
        st.markdown('<div class="source-select">', unsafe_allow_html=True)
        source_type = st.selectbox(
            "Source Type",
            ["Webcam", "Video File", "RTSP Stream"],
            label_visibility="collapsed"
        )
        
        source_path = None
        uploaded_file = None
        
        if source_type == "Video File":
            uploaded_file = st.file_uploader("Video File", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
            if uploaded_file:
                source_path = BytesIO(uploaded_file.getvalue())
                st.markdown(f'<small style="color: #94a3b8;">Loaded: {uploaded_file.name}</small>', unsafe_allow_html=True)
        elif source_type == "RTSP Stream":
            source_path = st.text_input("Stream URL", placeholder="rtsp://camera-url", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Start/Stop button
        button_class = ' stop-button' if st.session_state.is_streaming else ''
        button_text = "⏹ Stop Surveillance" if st.session_state.is_streaming else "▶ Start Surveillance"
        
        st.markdown(f'<div class="button-container{button_class}">', unsafe_allow_html=True)
        start_stop = st.button(button_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System info
        if st.session_state.processor is not None:
            device = st.session_state.processor.device
            device_icon = "🔥" if device == "cuda" else "🍎" if device == "mps" else "💻"
            st.markdown(f'<div class="device-badge">{device_icon} Running on {device.upper()}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2: Analysis & Q&A
    with col2:
        # Scene analysis
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🧠 AI Scene Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="caption-container">', unsafe_allow_html=True)
        caption_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual Q&A
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">❓ Visual Question Answering</div>', unsafe_allow_html=True)
        
        question = st.text_input("Question", placeholder="Ask something about the scene...", label_visibility="collapsed")
        ask_button = st.button("Ask AI")
        answer_placeholder = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Button logic
    if start_stop:
        if not st.session_state.is_streaming:
            # Start surveillance
            if st.session_state.processor is None:
                with st.spinner("Loading AI models..."):
                    st.session_state.processor = load_processor()
            
            st.session_state.stop_event.clear()
            st.session_state.frame_queue = queue.Queue(maxsize=1)
            st.session_state.caption_queue = queue.Queue(maxsize=10)
            st.session_state.thread = threading.Thread(
                target=process_video,
                args=(
                    st.session_state.stop_event,
                    st.session_state.frame_queue,
                    st.session_state.caption_queue,
                    st.session_state.processor,
                    source_type,
                    source_path
                ),
                daemon=True
            )
            st.session_state.thread.start()
            st.session_state.is_streaming = True
            st.rerun()
        else:
            # Stop surveillance
            st.session_state.stop_event.set()
            if st.session_state.thread:
                st.session_state.thread.join(timeout=1.0)
            st.session_state.frame = None
            st.session_state.is_streaming = False
            video_placeholder.empty()
            st.rerun()
    
    # Ask button logic
    if ask_button and question and st.session_state.frame is not None:
        with st.spinner("Processing question..."):
            img = Image.fromarray(cv2.cvtColor(st.session_state.frame, cv2.COLOR_BGR2RGB))
            answer = st.session_state.processor.answer_question(img, question)
            answer_placeholder.markdown(f'<div class="answer-box">📝 {answer}</div>', unsafe_allow_html=True)
    elif ask_button and not st.session_state.frame:
        answer_placeholder.markdown('<div class="answer-box" style="border-left-color: #f87171;">⚠️ Please start surveillance first</div>', unsafe_allow_html=True)

    # Update loop
    if st.session_state.is_streaming:
        placeholder = st.empty()
        while True:
            try:
                # Update video frame
                frame = st.session_state.frame_queue.get_nowait()
                st.session_state.frame = frame
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                # Update captions
                while not st.session_state.caption_queue.empty():
                    new_caption = st.session_state.caption_queue.get_nowait()
                    st.session_state.captions.append(new_caption)
                    st.session_state.captions = st.session_state.captions[-4:]  # Keep last 4 captions

                if st.session_state.captions:
                    caption_text = "<div class='captions-container'>" + "".join([
                        f"<div class='caption-item'><span class='timestamp'>[{cap['timestamp']}]</span> {cap['caption']}</div>"
                        for cap in reversed(st.session_state.captions)
                    ]) + "</div>"
                    caption_placeholder.markdown(caption_text, unsafe_allow_html=True)

            except queue.Empty:
                time.sleep(0.01)
                continue

if __name__ == "__main__":
    main()