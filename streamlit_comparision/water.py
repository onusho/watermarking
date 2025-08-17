import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import os
import copy



def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        
        print(f"Extracted frame {frame_count}")
        frame_count += 1
        if frame_count > 60:
         break
    
    video_capture.release()
    print("Frame extraction completed.")
    return fps

frame_rate = extract_frames('test_video.mp4', 'extracted_frames')

def create_watermark(shape,seed=42):
    np.random.seed(seed)  
    return np.random.randint(0, 2, shape, dtype=np.uint8)

def dwt_transform(image, wavelet='haar'):
    b, g, r = cv2.split(image.astype(np.float32))

    coeffsb = pywt.dwt2(b, wavelet)
    coeffsg = pywt.dwt2(g, wavelet)
    coeffsr = pywt.dwt2(r, wavelet)
    
    LLb, (LHb, HLb, HHb) = coeffsb
    LLg, (LHg, HLg, HHg) = coeffsg
    LLr, (LHr, HLr, HHr) = coeffsr

    LL = cv2.merge((LLb, LLg, LLr))
    LH = cv2.merge((LHb, LHg, LHr))
    HL = cv2.merge((HLb, HLg, HLr))
    HH = cv2.merge((HHb, HHg, HHr))
    
    return (LL, (LH, HL, HH))

def dwt_transform_inverse(coeffs, wavelet='haar'):
    LL, (LH, HL, HH) = coeffs
    LLb, LLg, LLr = cv2.split(LL)
    LHb, LHg, LHr = cv2.split(LH)
    HLb, HLg, HLr = cv2.split(HL)
    HHb, HHg, HHr = cv2.split(HH)

    coeffsb = (LLb, (LHb, HLb, HHb))
    coeffsg = (LLg, (LHg, HLg, HHg))
    coeffsr = (LLr, (LHr, HLr, HHr))

    Ib = pywt.idwt2(coeffsb, wavelet=wavelet)
    Ig = pywt.idwt2(coeffsg, wavelet=wavelet)
    Ir = pywt.idwt2(coeffsr, wavelet=wavelet)

    return cv2.merge((Ib, Ig, Ir))

def CFold_1(Z):
    return np.mod(Z.real, 1) + (np.mod(Z.imag, 1)) * 1j

def compute_Z1(a, c1, Z1, Z2):
    return CFold_1(a * ((Z1 / Z2)**2) + c1)

def compute_Z2(b, c2, Z1, Z2):    
    return CFold_1(b * ((Z2 / Z1)**2) + c2)

def compute_x_y(Z1, Z2, rcB4, ccB4):
    return (int(np.floor(Z1.real * (10**14)) % rcB4), 
            int(np.floor(Z2.imag * (10**14)) % ccB4))

def compute_channel(Z2, channels=3):
    return int(np.floor(Z2.real * (10**14)) % channels)

def compute_B(Z2):
    return int(np.floor(Z2.imag * (10**14)) % 2)

def condition_for_zero(Wij, U, T, epsilon):
    return (Wij == 0 and 
            np.abs(U[0, 0]) > np.abs(U[1, 0]) + epsilon and 
            np.abs(np.abs(U[1, 0]) - np.abs(U[0, 0])) < T)

def condition_for_one(Wij, U, T, epsilon):
    return (Wij == 1 and  
            np.abs(U[0, 0]) < np.abs(U[1, 0]) - epsilon and 
            np.abs(np.abs(U[1, 0]) - np.abs(U[0, 0])) > T)

def inverse_svd(U, S, Vt):
    S_full = np.zeros((U.shape[0], Vt.shape[0]))
    np.fill_diagonal(S_full, S)
    return np.dot(U, np.dot(S_full, Vt))


def embedding_extraction_function(I=None, W=None, param=None, extract=True):
    if I is None:
        raise Exception("Frame not provided.")
    elif W is None:
        raise Exception("Watermark not provided for Embedding/Extraction.")
    elif param is None:
        raise Exception("Parameters not provided.")

    LL, (LH, HL, HH) = dwt_transform(I)    
    rc, cc, channels = LL.shape
    rcB4, ccB4 = rc // 4, cc // 4

    mask = np.zeros((rcB4, ccB4, channels))     


    epsilon = 0#1e-6

    for i in range(param['W_shape'][0]):
        for j in range(param['W_shape'][1]):
            while True:
                Z1_next = compute_Z1(a=param['a'], 
                                     c1=param['c1'], 
                                     Z1=param['Z1_history'][-1], 
                                     Z2=param['Z2_history'][-1])
                Z2_next = compute_Z2(b=param['b'], 
                                     c2=param['c2'], 
                                     Z1=param['Z1_history'][-1], 
                                     Z2=param['Z2_history'][-1])

                param['Z1_history'].append(Z1_next)
                param['Z2_history'].append(Z2_next)

                x, y = compute_x_y(Z1=Z1_next, 
                                   Z2=Z2_next, 
                                   rcB4=rcB4, 
                                   ccB4=ccB4)

                channel = compute_channel(Z2=Z2_next)

                B = compute_B(Z2=Z2_next)
                
                if mask[x, y, channel] == 0:
                    break

            block = LL[x * 4: x * 4 + 4, 
                       y * 4: y * 4 + 4,
                       channel]
            
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            # EMBEDDING
            if not extract:
                meanV = (np.abs(U[0, 0]) + np.abs(U[1, 0])) / 2

                # XOR with B
                Wij = W[i, j] ^ B

                if Wij == 0:
                    U[0, 0] = np.sign(U[0, 0]) * (meanV + param['T'] / 2)
                    U[1, 0] = np.sign(U[1, 0]) * (meanV - param['T'] / 2)
                else:
                    U[0, 0] = np.sign(U[0, 0]) * (meanV - param['T'] / 2)
                    U[1, 0] = np.sign(U[1, 0]) * (meanV + param['T'] / 2)

                LL[x * 4: x * 4 + 4, 
                   y * 4: y * 4 + 4,
                   channel] = inverse_svd(U, S, Vt)

            # EXTRACTION
            else:
                if np.abs(U[0, 0]) - np.abs(U[1, 0]) > epsilon:
                    extracted_bit = 0
                else:
                    extracted_bit = 1

                B = compute_B(Z2=Z2_next)
                W[i, j] = extracted_bit ^ B

            mask[x, y, channel] = 1
        
    if not extract:
        return dwt_transform_inverse((LL, (LH, HL, HH)))
    elif extract:
        return W

def embed(frames_folder, param):
    frame_paths = [frm for frm in os.listdir(frames_folder) if frm.endswith(".png")]
    frame_paths.sort()
    
    param_copy = copy.deepcopy(param)
    
    watermark = create_watermark(shape=param_copy['W_shape'],seed=param_copy['seed'])

    embedded_frames = []

    for frame_path in frame_paths:
        path = os.path.join(frames_folder, frame_path)
        print(f"Embedding Frame: {frame_path}")

        frame = cv2.imread(path)
        if frame is None:
            continue

        embedded_frame = embedding_extraction_function(
            I=frame, 
            W=watermark.copy(),  # Use a copy to avoid modifications
            param=param_copy,
            extract=False)
        
        embedded_frames.append(embedded_frame)

    print("Completed Embedding Frames")
    return embedded_frames, watermark, param_copy


param_initial = {
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

embedded_frames, watermark, param_embed = embed(
    frames_folder='extracted_frames', 
    param=param_initial)

plt.imshow(watermark, cmap=plt.cm.gray)
plt.title("Original Watermark")
plt.show()

def save_frames(frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_count, frame in enumerate(frames):
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved Frame {frame_count}")

def create_video_from_images(image_folder, output_video_path, frame_rate):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  
    
    if not images:
        print("No images found in the folder.")
        return
    
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error reading image {image_path}")
            continue
        
        video_writer.write(frame)
        print(f"Writing frame {image_path}")
    
    video_writer.release()
    print(f"Video creation completed. Saved to {output_video_path}")

save_frames(frames=embedded_frames, output_folder='embedded_frames')

create_video_from_images(
    image_folder='embedded_frames', 
    output_video_path='embedded_video.mp4', 
    frame_rate=frame_rate)


def extract(frames_folder, param):
    frame_paths = [frm for frm in os.listdir(frames_folder) if frm.endswith(".png")]
    frame_paths.sort()
    
    param_copy = copy.deepcopy(param)
    
    watermark_extracted = np.zeros((param['W_shape'][0], param['W_shape'][1], len(frame_paths)), dtype=np.uint8)

    for idx, frame_path in enumerate(frame_paths):
        path = os.path.join(frames_folder, frame_path)
        frame = cv2.imread(path)
        if frame is None:
            continue
        print(f"Extracting watermark from Frame: {frame_path}")
        W_extracted = np.zeros(param['W_shape'], dtype=np.uint8)
        W_extracted = embedding_extraction_function(
            I=frame, 
            W=W_extracted,
            param=param_copy,
            extract=True)
        
        watermark_extracted[:, :, idx] = W_extracted

    W_final = np.median(watermark_extracted, axis=2)
    W_final = np.round(W_final).astype(np.uint8)

    return W_final, param_copy 

W_extracted, param_extract = extract(
    frames_folder='embedded_frames',
    param=param_initial)

# f,a = plt.subplots(1,2)

# a[0].imshow(watermark, cmap ='gray')
# a[0].set_title("Original")
# a[1].imshow(W_extracted[:,:,-1], cmap=plt.cm.gray)
# a[1].set_title("Extracted Watermark")
# plt.show()


# accuracy = np.mean(watermark == W_extracted[:,:,-1]) * 100
# print(f"Watermark extraction accuracy: {accuracy:.2f}%")
