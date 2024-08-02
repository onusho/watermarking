import numpy as np
import cv2
import pywt
from numpy.linalg import svd

def chaotic_map(Z, C, alpha, beta, n):
    for _ in range(n):
        Z = Z**2 + C + alpha * np.real(Z) + 1j * beta * np.imag(Z)
    return Z

def embed_watermark(frame, watermark, Z1, Z2, C1, C2, alpha, beta):
    # Apply DWT to each channel
    channels = cv2.split(frame)
    dwt_frame = [pywt.dwt2(ch, 'haar') for ch in channels]
    
    # Generate chaotic sequences
    Z1 = chaotic_map(Z1, C1, alpha, beta, 1000)
    Z2 = chaotic_map(Z2, C2, alpha, beta, 1000)
    
    # Select 4x4 blocks from LL subband and apply SVD
    for c in range(3):  # Process each channel separately
        LL, (LH, HL, HH) = dwt_frame[c]
        for i in range(0, LL.shape[0], 4):
            for j in range(0, LL.shape[1], 4):
                block = LL[i:i+4, j:j+4]
                U, S, V = svd(block)
                
                # Embed watermark bit
                watermark_bit = watermark[i//4 % watermark.shape[0], j//4 % watermark.shape[1]]
                encrypted_bit = watermark_bit ^ (np.imag(Z2) > 0)
                U[0, 0] += 0.1 * encrypted_bit  # Adjust strength as needed
                
                # Reconstruct block
                LL[i:i+4, j:j+4] = np.dot(U, np.dot(np.diag(S), V))
        
        dwt_frame[c] = (LL, (LH, HL, HH))
    
    # Inverse DWT
    watermarked_frame = [pywt.idwt2(dwt_ch, 'haar') for dwt_ch in dwt_frame]
    watermarked_frame = np.clip(np.stack(watermarked_frame, axis=-1), 0, 255).astype(np.uint8)
    
    return watermarked_frame, Z1, Z2

def extract_watermark(watermarked_frame, watermark_shape, Z1, Z2, C1, C2, alpha, beta):
    # Apply DWT to each channel
    channels = cv2.split(watermarked_frame)
    dwt_frame = [pywt.dwt2(ch, 'haar') for ch in channels]
    
    # Generate chaotic sequences
    Z1 = chaotic_map(Z1, C1, alpha, beta, 1000)
    Z2 = chaotic_map(Z2, C2, alpha, beta, 1000)
    
    extracted_watermark = np.zeros(watermark_shape, dtype=bool)
    
    # Select 4x4 blocks from LL subband and apply SVD
    for c in range(3):  # Process each channel separately
        LL, _ = dwt_frame[c]
        for i in range(0, LL.shape[0], 4):
            for j in range(0, LL.shape[1], 4):
                block = LL[i:i+4, j:j+4]
                U, _, _ = svd(block)
                
                # Extract watermark bit
                extracted_bit = U[0, 0] > 0
                decrypted_bit = extracted_bit ^ (np.imag(Z2) > 0)
                extracted_watermark[i//4 % watermark_shape[0], j//4 % watermark_shape[1]] = decrypted_bit
    
    return extracted_watermark

# Video processing
def process_video(input_video, output_video, watermark, Z1, Z2, C1, C2, alpha, beta):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        watermarked_frame, Z1, Z2 = embed_watermark(frame, watermark, Z1, Z2, C1, C2, alpha, beta)
        out.write(watermarked_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:  # Update every 10 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.2f}%)")
    
    cap.release()
    out.release()
    print("Video processing completed")

# Example usage
watermark = cv2.imread('watermark.png', 0) > 128  # Binary watermark

Z1 = complex(0.1, 0.1)
Z2 = complex(0.2, 0.2)
C1 = complex(0.3, 0.3)
C2 = complex(0.4, 0.4)
alpha = 0.5
beta = 0.6

input_video = 'D:/Projects/Chaotic Map/big_buck_bunny_720p_2mb.mp4'
output_video = 'watermarked_videoDWT.mp4'

process_video(input_video, output_video, watermark, Z1, Z2, C1, C2, alpha, beta)

# To extract watermark from a single frame (for verification)
cap = cv2.VideoCapture(output_video)
ret, frame = cap.read()
cap.release()

if ret:
    extracted_watermark = extract_watermark(frame, watermark.shape, Z1, Z2, C1, C2, alpha, beta)
    cv2.imwrite('extracted_watermarkDWT.png', extracted_watermark.astype(np.uint8) * 255)