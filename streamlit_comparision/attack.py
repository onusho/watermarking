import numpy as np
import cv2
import random

def apply_noise(image, noise_type, **params):
    if noise_type == 'Gaussian':
        mean = params.get('mean', 0)
        var = params.get('var', 0.01)
        sigma = var ** 0.5
        
        # Ensure the noise is of the same type as the image
        gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
        
        # Clip values to ensure they stay within the valid range for uint8
        noisy_image = cv2.add(image.astype('float32'), gauss)
        noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
        return noisy_image
    
    elif noise_type == 'Salt and Pepper':
        prob = params.get('prob', 0.01)
        noisy = np.copy(image)
        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        noisy[probs < prob] = black
        noisy[probs > 1 - prob] = white
        return noisy
    
    elif noise_type == 'Speckle':
        gauss = np.random.randn(*image.shape).astype('float32')
        noisy_image = image + image * gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
        return noisy_image

def apply_compression(image, compression_type, **params):
    if compression_type == 'JPEG':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), params.get('quality', 50)]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)
    elif compression_type == 'JPEG2000':
        encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), params.get('quality', 50)]
        _, encimg = cv2.imencode('.jp2', image, encode_param)
        return cv2.imdecode(encimg, 1)

def apply_resize(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

def apply_rotation(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def apply_brightness_contrast(image, brightness, contrast):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Video-specific attacks
def apply_frame_averaging(frames):
    for i in range(1, len(frames) - 1):
        frames[i] = (frames[i-1] // 3 + frames[i] // 3 + frames[i+1] // 3).astype(np.uint8)
    return frames

def apply_frame_dropping(frames, drop_rate=0.1):
    return [frame for i, frame in enumerate(frames) if random.random() > drop_rate]

def apply_frame_insertion(frames, insert_rate=0.1):
    new_frames = []
    for frame in frames:
        new_frames.append(frame)
        if random.random() < insert_rate:
            new_frames.append(frame)
    return new_frames

def apply_frame_shuffling(frames, shuffle_rate=0.1):
    num_to_shuffle = int(len(frames) * shuffle_rate)
    indices_to_shuffle = random.sample(range(len(frames)), num_to_shuffle)
    shuffled_frames = [frames[i] for i in indices_to_shuffle]
    random.shuffle(shuffled_frames)
    for i, index in enumerate(indices_to_shuffle):
        frames[index] = shuffled_frames[i]
    return frames
