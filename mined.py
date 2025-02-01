import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import easyocr
import re
import shutil
from collections import defaultdict, Counter
import torch
# import gun_predict
from gun_predict import CNNClassifier, check_gun
import pandas as pd
import math
from gun_predict import check_gun
import ffmpeg

video_start = []
video_end = []

weapon_model = torch.load(r"C:\Users\Dell\Downloads\weapon_detect.pth", map_location=torch.device('cpu'))
weapon_model.eval()
def extract_frames(video_path, output_folder, interval=2):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(fps * interval) == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()

    print("Frames extraction complete.")


def get_history(folder_path):
    
    reader = easyocr.Reader(['en'])
    model = YOLO(r"C:\Users\Dell\Downloads\best.pt")
    

    image_files = [f for f in os.listdir(folder_path) if f.startswith("frame_") and f.endswith(".jpg")]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    image_files = [os.path.join(folder_path, f) for f in image_files]
    frames = [cv2.imread(img) for img in image_files]
    print(f"Loaded {len(frames)} frames.")

    history = defaultdict(list)
    gun_used = defaultdict(list)

    for i in range(len(frames)):
        results = model(frames[i])
        frame_path = image_files[i]
        frame_number = int(re.search(r'frame_(\d+)', frame_path).group(1))
        for result in results:
            image_with_boxes = result.plot() 
            
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)


            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                class_id = int(box.cls[0])
                
                crop_color = frames[i][y1:y2, x1:x2]
                gray_image = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
                crop = cv2.bitwise_not(gray_image)
                # plt.figure(figsize=(8, 8))
                # plt.imshow(crop,cmap='gray')
                # plt.axis("off")
                # plt.show()
                text = reader.readtext(crop, detail=1)
                text = sorted(text, key=lambda x : x[0][0][0])
                ordered_text = [result[1] for result in text]
                print(ordered_text)
                key = tuple(ordered_text)
                weapon_name = "Unknown"
                try:
                    weapon_name = check_gun(crop_color, weapon_model)
                except Exception as e:
                    print(f"Error occurred at frame {frame_number}")
                if key in history:
                    last_timestamp = history[key][-1]
                    if frame_number - last_timestamp > 5*60:
                        history[key].append(frame_number)
                        gun_used[key].append({frame_number: weapon_name})
                else:
                    history[key].append(frame_number)
                    gun_used[key].append({frame_number: weapon_name})
                
                print(f"Class: {class_id}, Confidence: {confidence:.2f}, BBox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    cleaned_history = {k: v for k, v in history.items() if k != () and k != ('', '') and '' not in k}
    cleaned_gun_used = {k: v for k, v in gun_used.items() if k != () and k != ('', '') and '' not in k}
    return cleaned_history, cleaned_gun_used
            

def get_headshots(kill_history, frames_folder):

    reader = easyocr.Reader(['en'])
    headshots = []
    
    for (killer, victim), frame_numbers in kill_history.items():

        for frame_number in frame_numbers:
            frame_filename = os.path.join(frames_folder, f'frame_{frame_number}.jpg')
            frame = cv2.imread(frame_filename)

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inverted_image = cv2.bitwise_not(gray_image)
  
            h,w=inverted_image.shape
            inverted_image=inverted_image[int(0.867*h):int(0.9*h),int(0.47*w):int(0.53*w)]

            results = reader.readtext(inverted_image, detail=0)
            if len(results)>0:
                results[0]=results[0].replace(" ","").lower()
                print(results)
                str="headshot"
                ct=0
                for i in range(8):
                    if(str[i]==results[0][i]):
                        ct+=1
                if ct>=4 and frame_number not in headshots:
                    headshots.append(frame_number)
        
    return headshots

def get_clutches(kill_history, frames_folder):

    reader = easyocr.Reader(['en'])
    clutches = []

    for (killer, victim), frame_numbers in kill_history.items():
        if killer != "Me" and victim!="Me":  
            continue

        for frame_number in frame_numbers:
            frame_filename = os.path.join(frames_folder, f'frame_{frame_number}.jpg')
            frame = cv2.imread(frame_filename)

            if frame is None:
                print(f"❌ Frame not found: {frame_filename}")
                continue

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inverted_image = cv2.bitwise_not(gray_image)

            h, w = inverted_image.shape
            clutch_roi = inverted_image[int(0.131*h):int(0.261*h), int(0.37*w):int(0.625*w)]
            plt.imshow(clutch_roi, cmap="gray")
            plt.show()
            results = reader.readtext(clutch_roi, detail=0)
            if len(results) > 0:
                detected_text = results[0].replace(" ", "").lower()
                print(f"Frame {frame_number} OCR Result: {detected_text}")

                target_word = "clutch"
                match_count = sum(1 for i in range(min(len(detected_text), len(target_word))) if detected_text[i] == target_word[i])

                if match_count >= 3 and frame_number not in clutches:  # Allow some OCR errors
                    clutches.append(frame_number)

    return clutches

def highlight_video(video_path):
    try:
        # Open the input video
        output_path="output-video.mp4"
        cap = cv2.VideoCapture(video_path)
        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize VideoWriter to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (for .mp4)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        clips = []
        
        # Iterate over the start and end times and process clips
        for start, end in zip(video_start, video_end):
            # Set the position to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start * fps)
            
            while(cap.isOpened()):
                ret, frame = cap.read()
                current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                
                if not ret or current_time >= end:
                    break
                
                # Write frames to the output video
                out.write(frame)

        # Release video objects
        cap.release()
        out.release()

        print("✅ Video successfully created:", output_path)
        return output_path

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None
def process_video(video_path):
    
    output_folder = './Frames_02/Frames_02'
    # extract_frames(video_path, output_folder, interval=2)
    kill_history, gun_used = get_history(output_folder)
    print(kill_history)
    kills = sum(len(timestamps) for (killer, victim), timestamps in kill_history.items() if killer == "Me")

    deaths = sum(len(timestamps) for (killer, victim), timestamps in kill_history.items() if victim == "Me")
    headshots = get_headshots(kill_history, output_folder)
    clutches = get_clutches(kill_history, output_folder)
    for (killer, victim), timestamps in kill_history.items():
        if killer=="Me":
            for i in timestamps:
                print(i)
                video_start.append(int(i/59.5-2))
                video_end.append(math.ceil(i/59.5+3))
        if victim == "Me":
             for i in timestamps:
                print(i)
                video_start.append(int(i/59.5-2))
                video_end.append(math.ceil(i/59.5+3))

    return kill_history, gun_used, kills, deaths, headshots, clutches

def match_timeline(kill_history, gun_used):
    kill_list = []

    for (killer, victim), time_steps in kill_history.items():
        for time in time_steps:
            kill_list.append((time, killer, victim))

    kill_list.sort()
    kill_text = []
    for time, killer, victim in kill_list:
        for entry in gun_used[(killer, victim)]:
            if entry.get(time, 'Unknown') != 'Unknown':
                kill_text.append(f"timestep_{math.ceil(time / 60)}: {killer} killed {victim} using {entry[time]} weapon")
    bin_size = 200

    time_steps = []
    kill_details = []  # Store detailed information for tooltips

    for (killer, victim), times in kill_history.items():
        for t in times:
            time_steps.append(t)
            kill_details.append((t, killer, victim))  # Store details
    
    min_time = min(time_steps)
    max_time = max(time_steps)

    # Compute binned kills
    binned_kills = Counter((((t - min_time) // bin_size) * bin_size + min_time) / 60 + 1 for t in time_steps)

    bins = sorted(binned_kills.keys())
    kill_counts = [binned_kills[b] for b in bins]

    # Prepare data for hover tooltips
    bin_to_details = {b: [] for b in bins}

    for t, killer, victim in kill_details:
        bin_key = (((t - min_time) // bin_size) * bin_size + min_time) / 60 + 1
        bin_to_details[bin_key].append(f"{killer} → {victim}")
    df = pd.DataFrame({
        "Timestamp (sec)": bins,
        "Kill Count": kill_counts,
        "Kill Details": [", ".join(bin_to_details[b]) for b in bins]  # Tooltip data
    })
    return kill_text, df