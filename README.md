
# 🎯 PlayPulse: Analyze like a pro, strategize like a champion!🏆

## 🚀 Overview
**PlayPulse** is an advanced tool that extracts key insights from a recorded gameplay video(ex. Valorant).It leverages YOLOv8, Computer Vision, OCR, and Deep Learning to provide in-depth player performance analysis. The tool presents intuitive visuals, round-wise summaries, and even generates highlight videos! 🎥🔥

## 📥 Input
- Recorded gameplay video 🎮

## 🛠️ Methodology
1. **UI Creation:** Built using Streamlit for an interactive and aesthetic interface.
2. **Frame Extraction:** Extract frames at 2-second intervals from the video.
3. **Object Detection & OCR:** Apply YOLOv8, EasyOCR, and Image Processing to extract event details.
4. **Mapping Events:** Generate a map of [Killer -> Victim -> Timestep] from extracted data.
5. **Player Performance Metrics:** Calculate:
   - ✅ Total Kills
   - ❌ Total Deaths
   - 🎯 Headshots & Headshot %
   - ⚖ Kill-Death Ratio
   - 🎯 Clutch
6. **Summary:** Generate a timeline of key events 📜
7. **Data Visualization:** Plot Kills vs. Timestep graph 📊
8. **Highlight Video:** Create a short highlight video with key moments 🔥🎬
9. **Weapon Identification:** Use a custom CNN model to classify player’s weapon 🔫
10. **SWOT Analysis:** Generate strengths, weaknesses, opportunities, and threats of the player 📌
11. **AI-Powered Insights:** Use Gemini API to generate engaging content ✨

![Summary Chart](image/MINed.jfif)
## 🎨 Features
- 📊 Comprehensive Player Stats
- 📹 Auto-Generated Highlight Video
- 🗺 Chronological Summary
- 🔫 Weapon Detection with CNN
- 🏆 Headshot Accuracy & Performance Trends
- 🤖 AI-Powered Insights & Content Generation

## 🏗️ Tech Stack
- Python 🐍
- Streamlit 🖥️
- YOLOv8 🧠
- EasyOCR 🔍
- OpenCV 👀
- Deep Learning (CNN) 🏗️
- Gemini API
- ffmpeg 📸
- PyTorch 👀
- Makesense AI annotator 📸

## 📸 Screenshots
![Home Page](image/1.jfif)
![Player stats](image/2.jfif)
![Graph](image/3.jfif)
![SWOT](image/4.jfif)
![Highlight](image/5.jfif)
## 🔧 Installation & Usage
1️⃣ Clone the repo:
```bash
 git clone https://github.com/siya25097/PlayPulse.git
 cd PlayPulse
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app:
```bash
streamlit run app.py
```
Future Scopes 🚀

1️⃣ Weapon Performance Metrics
Description: Accurate detection for weapons along with its performance and accuracy.

2️⃣ Multi-Game Support
Description: Extend analysis to other games.

3️⃣ Roundwise game analysis
Description: Highlight round wins






