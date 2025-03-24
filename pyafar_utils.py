"""
Utility functions for processing and analyzing facial action units using PyAFAR.
"""

from PyAFAR_GUI import infant_afar, adult_afar
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

AUs_infant = ["au_1", "au_2", "au_3", "au_4", "au_6", "au_9", "au_12", "au_20", "au_28"]
AUs_adult = ["au_1", "au_2", "au_3", "au_4", "au_6", "au_7", "au_9", "au_10", "au_12",
             "au_14", "au_15", "au_17", "au_20", "au_23", "au_24", "au_28"]
AUs_adult_int = ["au_6", "au_10", "au_12", "au_14", "au_17"]

AU_descriptions = {
    "au_1": "Inner brow raiser - Raises the inner eyebrows (surprise, sadness)",
    "au_2": "Outer brow raiser - Raises the outer eyebrows (surprise)",
    "au_3": "Brow furrower - Furrows the brows (anger, concentration)",
    "au_4": "Brow lowerer - Lowers the brows (anger, frustration)",
    "au_6": "Cheek raiser - Raises the cheeks (smiling, happiness)",
    "au_7": "Lid tightener - Tightens the eyelids (squinting, confusion)",
    "au_9": "Nose wrinkler - Wrinkles the nose (disgust, concentration)",
    "au_10": "Upper lip raiser - Raises the upper lip (disgust, contempt)",
    "au_12": "Lip corner puller - Pulls the lip corners up (smiling, happiness)",
    "au_14": "Dimpler - Raises the corners of the mouth (smiling, happiness)",
    "au_15": "Lip corner depressor - Lowers the corners of the mouth (sadness, displeasure)",
    "au_17": "Chin raiser - Raises the chin (contempt, pride)",
    "au_20": "Lip stretcher - Stretches the lips horizontally (fear, tension)",
    "au_23": "Lip tightener - Tightens the lips (anger, concentration)",
    "au_24": "Lip pressor - Presses the lips together (thoughtfulness, anger)",
    "au_28": "Lip suck - Sucks in the lips (uncertainty, thinking)"
}


def run_infant(input_path, AUs=AUs_infant):
    infant_result = infant_afar.infant_afar(filename=input_path, AUs=AUs, GPU=True, max_frames=float('inf'))

    # For some reason, sometimes there's more Frames than other columns
    min_length = min(len(values) for values in infant_result.values())
    for key in infant_result:
        infant_result[key] = infant_result[key][:min_length]

    df = pd.DataFrame.from_dict(infant_result)
    return df


def run_adult(input_path, AUs=AUs_adult, AU_Int=AUs_adult_int):
    adult_result = adult_afar.adult_afar(filename=input_path, AUs=AUs, GPU=True, max_frames=float('inf'),
                                          AU_Int=AU_Int, batch_size=128, PID=False)

    # For some reason, sometimes there's more Frames than other columns
    min_length = min(len(values) for values in adult_result.values())
    for key in adult_result:
        adult_result[key] = adult_result[key][:min_length]

    df = pd.DataFrame.from_dict(adult_result)
    return df


def save_video_with_au(df, input_path, output_path, landmarks=True, max_frames=float('inf')):
    df = fix_dataframe(df)
    cap = cv2.VideoCapture(input_path)

    # Check if the video is opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open the video file.")

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Print video properties
    print(f"Frame width: {frame_width}")
    print(f"Frame height: {frame_height}")
    print(f"FPS: {fps}")

    # Set the output video properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

    # Loop over the frames of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(min(frame_count, max_frames))):
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break

        # Get the action unit values for the current frame from your DataFrame
        AUs = [col for col in df.columns if col.startswith("Occ_") or col.startswith("Int_")]
        current_au_values = df.iloc[frame_idx][AUs].values

        # Create a blank image for the AU overlay (the right side of the video)
        overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Color-code the AU values (you can adjust the threshold and colors)
        text_height = int((frame_height - 100) / len(AUs))
        for i, (au, value) in enumerate(zip(AUs, current_au_values)):
            # Set color based on AU intensity
            color = (0, int(255 * value), 255 - int(255 * value))  # Green for high, Red for low

            # Display text
            text = f"{au.upper()}: {AU_descriptions.get(au[4:], 'Unknown')} ({value:.2f})"
            cv2.putText(overlay, text, (50, 100 + i * text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2, cv2.LINE_AA)

            if landmarks:
                # Extract the relevant columns
                landmark_columns_x = [col for col in df.columns if col.startswith("x_")]
                landmark_columns_y = [col for col in df.columns if col.startswith("y_")]

                # Get the x and y coordinates for the landmarks as numpy arrays
                x_coords = df.iloc[frame_idx][landmark_columns_x].values * frame_width
                y_coords = df.iloc[frame_idx][landmark_columns_y].values * frame_height

                for (x, y) in zip(x_coords, y_coords):
                    cv2.drawMarker(frame, (int(x), int(y)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                                   markerSize=3, thickness=1)

        # Combine the original frame (left) and the overlay (right) and write it to the output video
        combined_frame = np.hstack((frame, overlay))
        out.write(combined_frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def crop_video(input_path, output_path, start_time, end_time=-1):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open the video file.")

    # Calculate the corresponding frame numbers
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    if end_time == -1:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        end_frame = int(end_time * fps)

    # Set the video writer to save the clipped video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Skip to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read and write frames from the start to the end frame
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        out.write(frame)

    # Release the video objects
    cap.release()
    out.release()


def merge_videos_vertically(input_paths, output_path):
    # Open video captures
    caps = [cv2.VideoCapture(video) for video in input_paths]

    # Get video properties from the first video
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)

    # Ensure all videos are of the same size and length
    for cap in caps:
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != frame_width or
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != frame_height or
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < frame_count):
            raise ValueError("All videos must have the same resolution")
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count:
            print("A video doesn't have the same number of frames - cropping")
            [cap.read() for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - frame_count)]

    # Define output video properties
    out_height = frame_height * len(input_paths)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, out_height))

    for _ in range(frame_count):
        frames = [cap.read()[1] for cap in caps]
        if any(frame is None for frame in frames):
            break  # Stop if any video ends unexpectedly
        stacked_frame = np.vstack(frames)
        out.write(stacked_frame)

    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()


def fix_dataframe(df):
    return pd.DataFrame({"Frame": range(0, df["Frame"].max() + 1)}).merge(df, on="Frame", how="left").fillna(-1)


def export_au_to_csv(df, output_path, type='all'):
    if type == 'all':
        df.to_csv(output_path, index=False)
    elif type == 'au':
        selected_columns = ["Frame"] + [col for col in df.columns if col.startswith("Occ_") or col.startswith("Int_")]
        df[selected_columns].to_csv(output_path, index=False)
    elif type == 'infant':
        selected_columns = ["Frame"] + [col for col in df.columns if col.startswith("Occ_") and col[4:] in AUs_infant]
        df[selected_columns].to_csv(output_path, index=False)
