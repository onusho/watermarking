from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to calculate PSNR and SSIM
def evaluate_quality(original, watermarked):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    
    psnr_value = psnr(original_gray, watermarked_gray)
    ssim_value = ssim(original_gray, watermarked_gray)
    
    return psnr_value, ssim_value

# Example usage for embedding
video_frame = cv2.imread('extracted_frames/frame_0000.png')
watermarked_LLband = Embedding_Extraction(video_frame, W, a, b, Z1, Z2, 0)

# Calculate PSNR and SSIM between original and watermarked images
psnr_value, ssim_value = evaluate_quality(video_frame, watermarked_LLband)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")

# Example usage for extraction
extracted_watermark = Embedding_Extraction(watermarked_LLband, W, a, b, Z1, Z2, 1)

plt.imshow(extracted_watermark, cmap=plt.cm.gray)
plt.show()
