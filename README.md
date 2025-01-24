# Securade.ai Sentinel

An advanced AI-powered video surveillance and analysis system that provides automated monitoring, visual Q&A, and real-time video captioning capabilities for CCTV cameras and video feeds.

## Features

- Support for multiple video input sources:
  - Local video files
  - Webcam feeds
  - RTSP streams from IP cameras
- Real-time video captioning using Salesforce BLIP model
- Natural language visual Q&A using VILT model
- Interactive Streamlit interface
- Continuous monitoring and analysis
- Screenshot capture and analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/securade-sentinel.git
cd securade-sentinel
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- streamlit
- torch
- transformers
- opencv-python
- pillow
- numpy

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Select your video source:
   - Upload a local video file
   - Use your webcam
   - Enter an RTSP stream URL

3. The application will begin processing the video feed and displaying:
   - Live video stream
   - Real-time captions
   - Q&A interface for querying the video content

## Models

The application uses two main AI models:

1. **Video Captioning**: Salesforce/blip-image-captioning-large
   - Generates natural language descriptions of video scenes

2. **Visual Q&A**: dandelin/vilt-b32-finetuned-vqa
   - Answers questions about the video content in natural language

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

