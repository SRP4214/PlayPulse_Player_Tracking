import streamlit as st
import cv2
import tempfile
import os
from gun_predict import CNNClassifier
from mined import process_video, match_timeline, highlight_video
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
# from moviepy.editor import VideoFileClip, concatenate_videoclips


# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configure LLM model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

st.set_page_config(page_title="Gaming Video Upload", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #0c0f1c;
        color: #00ffcc;
        font-family: 'Courier New', monospace;
    }
    .stApp {
        background-color: #0c0f1c;
    }
    .stButton>button {
        background-color: #ff007f;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stFileUploader {
        background-color: #11152b;
        border: 2px solid #ff007f;
        padding: 10px;
        border-radius: 10px;
    }
    .stMetric {
        font-size: 22px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

llm = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

# Title
st.title("🎮 Valorant Game Analyser")

uploaded_file = st.sidebar.file_uploader("Upload your gaming video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    st.write(isinstance(tfile.name, str))

    st.sidebar.success("✅ Video uploaded successfully!")

    # Use cached function to avoid reprocessing
    kill_history, gun_used, kills, deaths, headshots, clutches = process_video(tfile.name)

    kill_text, df = match_timeline(kill_history, gun_used)

    # **Display Stats First**
    st.subheader("🏆 Player Stats ⚔️")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🔫 Kills", value=kills)
    with col2:
        st.metric(label="🎯 Headshots", value=len(headshots))
    with col3:
        st.metric(label="💀 Deaths", value=deaths)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🔫 K/D Ratio", value=f"{kills}/{deaths}")
    with col2:
        st.metric(label="🎯 Headshots Ratio", value=f"{len(headshots)/deaths * 100:.2f}%")
    with col3:
        st.metric(label="💀 Clutch Moments", value=len(clutches))

    # **Generate AI Commentary**
    chat_session = llm.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"Print this in a fun way: {str(kill_text)}. Give each timestep_i name as Tick-Tock-i. Don't give any of your messages like 'Okay here is the following...'.",
                ],
            }
        ]
    )
    response = chat_session.send_message("Print this in a fun way.")

    # **Display AI's Response**
    st.subheader("🤖 AI Game Commentary")
    st.write(response.text)

    # **Bar Chart: Number of Kills vs Time**
    fig = px.bar(
        df,
        x="Timestamp (sec)",
        y="Kill Count",
        text="Kill Count",
        hover_data={"Kill Details": True},
        title="Number of Kills vs Timestep Range",
        labels={"Timestamp (sec)": "Time (seconds)", "Kill Count": "Number of Kills"},
    )

    fig.update_traces(marker_color='skyblue', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🎯 SWOT Analysis of Player")
    response = chat_session.send_message("Give SWOT Analysis of \"Me\" player based on the list.")
    st.write(response.text)

    st.subheader('🎥 Highlights 📸')
    
    path = tfile.name
    path.replace("\\", "/")
    output_path = highlight_video(path)
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()

# Create a download button
    st.download_button(
        label="Download Video",
        data=video_bytes,
        file_name="downloaded_video.mp4",
        mime="video/mp4"
    )
