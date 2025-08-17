import streamlit as st
import os
import cv2
import numpy as np
from water import extract_frames, embed, extract, save_frames, create_video_from_images
from attack import apply_noise, apply_compression, apply_resize, apply_rotation, apply_brightness_contrast, apply_frame_averaging, apply_frame_dropping, apply_frame_insertion, apply_frame_shuffling
import matplotlib.pyplot as pltsd
# Set the custom path
custom_path = "assets/sample_video.mp4"

# Initialize session state for parameters
if 'param_initial' not in st.session_state:
    st.session_state['param_initial'] = {
        'a': 10.7,
        'b': 11.2,
        'Z1_history': [0.83 + 0.35j],
        'Z2_history': [0.67 + 0.54j],
        'c1': 0.53 + 0.26j,
        'c2': 0.83 + 0.35j,
        'T': 0.25,
        'W_shape': (20, 20),
        'seed': 30
    }

# 1. Load the default video
st.title("Watermarking & Attacks Test Framework")
st.write("Step 1: Load Video")
video_path = custom_path
if st.button('Load Video'):
    st.session_state['fps'] = extract_frames(video_path, 'extracted_frames')
    st.write(f"Video loaded and frames extracted. FPS: {st.session_state['fps']}")

# 2. Configure initial parameters and watermark shape
st.write("Step 2: Configure Watermark Parameters")
param_initial = st.session_state['param_initial']
param_initial['a'] = st.slider("a", 0.0, 20.0, param_initial['a'])
param_initial['b'] = st.slider("b", 0.0, 20.0, param_initial['b'])
param_initial['W_shape'] = (st.slider("Watermark Width", 10, 50, param_initial['W_shape'][0]),
                            st.slider("Watermark Height", 10, 50, param_initial['W_shape'][1]))
st.session_state['param_initial'] = param_initial

# 3. Display the watermark and 'EMBED' button
if st.button('EMBED'):
    st.write("Embedding Watermark into Frames...")
    embedded_frames, watermark, param_embed = embed('extracted_frames', param_initial)
    save_frames(embedded_frames, 'embedded_frames')
    st.session_state['watermark'] = watermark
    st.write("Watermark Embedded!")
    st.image(watermark)
    st.write(watermark)
    st.session_state['embedded_frames'], watermark, param_embed = embed(
    frames_folder='extracted_frames', 
    param=param_initial)


# 4. Convert embedded frames into a video
if st.button('Convert to Video'):
    create_video_from_images('embedded_frames', 'embedded_video.mp4', st.session_state['fps'])
    st.video('embedded_video.mp4')

# 5. Extract and compare original vs embedded watermarks
if st.button('Extract Watermark'):
    W_extracted,param_copy = extract('embedded_frames', param_initial)
    
    original_watermark = np.array(st.session_state['watermark'])
    extracted_watermark = np.array(W_extracted)

    st.write(f"Original watermark shape: {original_watermark.shape}")
    st.write(f"Extracted watermark shape: {extracted_watermark.shape}")

    # Ensure both watermarks are compatible
    if original_watermark.shape == extracted_watermark.shape:
        mean_difference = np.mean(np.abs(original_watermark - extracted_watermark))
        st.write(f"Mean difference between original and extracted watermark: {mean_difference}")
    else:
        st.write("Error: The original and extracted watermarks have different shapes and cannot be directly compared.")

    # Display the watermarks for visual comparison
    st.image(original_watermark, caption="Original Watermark")
    st.image(extracted_watermark, caption="Extracted Watermark")

# 6. Perform all attacks
if st.button('Perform All Attacks'):
    attack_names = ["Noise", "Compression", "Resize", "Rotation", "Brightness/Contrast", "Frame Averaging", "Frame Dropping", "Frame Insertion", "Frame Shuffling"]
    attack_results = []
    status = st.empty()
    embedded_frames = st.session_state['embedded_frames']
    for attack in attack_names:
        status.text(f"Performing {attack} Attack...")
        
        if attack == "Noise":
            attacked_frames = [apply_noise(frame, 'Gaussian', mean=0, var=0.01) for frame in embedded_frames]
        elif attack == "Compression":
            attacked_frames = [apply_compression(frame, 'JPEG', quality=50) for frame in embedded_frames]
        elif attack == "Resize":
            attacked_frames = [apply_resize(frame, 0.5) for frame in embedded_frames]
        elif attack == "Rotation":
            attacked_frames = [apply_rotation(frame, 30) for frame in embedded_frames]
        elif attack == "Brightness/Contrast":
            attacked_frames = [apply_brightness_contrast(frame, brightness=50, contrast=1.5) for frame in embedded_frames]
        elif attack == "Frame Averaging":
            attacked_frames = apply_frame_averaging(embedded_frames)
        elif attack == "Frame Dropping":
            attacked_frames = apply_frame_dropping(embedded_frames, drop_rate=0.2)
        elif attack == "Frame Insertion":
            attacked_frames = apply_frame_insertion(embedded_frames, insert_rate=0.2)
        elif attack == "Frame Shuffling":
            attacked_frames = apply_frame_shuffling(embedded_frames, shuffle_rate=0.2)

        # Save attacked frames for each attack
        save_frames(attacked_frames, 'attacked_frames')
        create_video_from_images('attacked_frames', f'attacked_video_{attack}.mp4', st.session_state['fps'])
        
        # Extract watermark from attacked frames
        W_attacked,pc = extract('attacked_frames', param_initial)
        
        # Calculate NC and BER
        NC_value = np.corrcoef(st.session_state['watermark'].flatten(), W_attacked.flatten())[0, 1]
        BER_value = np.mean(st.session_state['watermark'] != W_attacked)
        
        attack_results.append({
            "attack": attack,
            "NC": NC_value,
            "BER": BER_value,
            "extracted_watermark": W_attacked,
            "video_path": f'attacked_video_{attack}.mp4'
        })

    status.text("All Attacks Completed")

    # 7. Final Report
    st.write("### Final Report")
    for result in attack_results:
        st.write(f"**{result['attack']} Attack:**")
        st.write(f"- NC: {result['NC']:.4f}")
        st.write(f"- BER: {result['BER']:.4f}")
        st.video(result['video_path'])
        st.image(result['extracted_watermark'], caption=f"Extracted Watermark after {result['attack']} Attack")

