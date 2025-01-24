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
from datetime import datetime

# Set page config to wide mode
st.set_page_config(layout="wide", page_title="Securade.ai Sentinel")

def initialize_state():
    if 'initialized' not in st.session_state:
        st.session_state.frame = None
        st.session_state.captions = []
        st.session_state.stop_event = threading.Event()
        st.session_state.frame_queue = queue.Queue(maxsize=1)
        st.session_state.caption_queue = queue.Queue(maxsize=10)
        st.session_state.processor = None
        st.session_state.thread = None
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
    elif source_type == "Video File":
        return cv2.VideoCapture(source_path)
    elif source_type == "RTSP Stream":
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
    
    # Main title
    st.title("Securade.ai Sentinel")

    # Create three columns for layout
    video_col, caption_col, qa_col = st.columns([0.4, 0.3, 0.3])

    # Video column
    with video_col:
        st.subheader("Video Feed")
        
        # Video source selection
        source_type = st.selectbox(
            "Select Video Source",
            ["Webcam", "Video File", "RTSP Stream"]
        )
        
        source_path = None
        if source_type == "Video File":
            source_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
            if source_file:
                # Save the uploaded file temporarily
                temp_file = BytesIO(source_file.read())
                source_path = temp_file
        elif source_type == "RTSP Stream":
            source_path = st.text_input("Enter RTSP URL", placeholder="rtsp://your-camera-url")

        start_stop = st.button("Start/Stop Surveillance")
        video_placeholder = st.empty()
        
        if start_stop:
            if st.session_state.stop_event.is_set():
                # Start surveillance
                if st.session_state.processor is None:
                    st.session_state.processor = load_processor()
                st.session_state.stop_event.clear()
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
            else:
                # Stop surveillance
                st.session_state.stop_event.set()
                if st.session_state.thread:
                    st.session_state.thread.join(timeout=1.0)
                st.session_state.frame = None
                video_placeholder.empty()

    # Caption column
    with caption_col:
        st.subheader("Scene Analysis")
        caption_placeholder = st.empty()

    # Q&A column
    with qa_col:
        st.subheader("Visual Q&A")
        question = st.text_input("Ask a question about the scene:")
        ask_button = st.button("Ask")
        answer_placeholder = st.empty()

        if ask_button and question and st.session_state.frame is not None:
            img = Image.fromarray(cv2.cvtColor(st.session_state.frame, cv2.COLOR_BGR2RGB))
            answer = st.session_state.processor.answer_question(img, question)
            answer_placeholder.markdown(f"**Answer:** {answer}")

    # Update loop
    if not st.session_state.stop_event.is_set():
        placeholder = st.empty()
        while True:
            try:
                # Update video frame
                frame = st.session_state.frame_queue.get_nowait()
                st.session_state.frame = frame
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Update captions
                while not st.session_state.caption_queue.empty():
                    new_caption = st.session_state.caption_queue.get_nowait()
                    st.session_state.captions.append(new_caption)
                    st.session_state.captions = st.session_state.captions[-5:]  # Keep last 5 captions

                if st.session_state.captions:
                    caption_text = "\n\n".join([
                        f"**[{cap['timestamp']}]** {cap['caption']}"
                        for cap in reversed(st.session_state.captions)
                    ])
                    caption_placeholder.markdown(caption_text)

            except queue.Empty:
                time.sleep(0.01)
                continue

if __name__ == "__main__":
    main()