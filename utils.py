import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash

def dhash(image, hash_size=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return imagehash.ImageHash(diff)

def compare_frames(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    score, diff = ssim(gray1, gray2, full=True)
    
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    percentage_changed = (changed_pixels / total_pixels) * 100
    
    combined_score = (1 - score) * 100 + percentage_changed
    return combined_score

def process_frame(args):
    frame, prev_frame, threshold = args
    difference = compare_frames(frame, prev_frame)
    if difference > threshold:
        return frame, dhash(frame)
    return None, None