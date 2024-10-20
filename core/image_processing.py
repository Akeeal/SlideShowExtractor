# Copyright 2024 Akeeal Mohammed

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
from functools import lru_cache

def dhash(image, hash_size=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return imagehash.ImageHash(diff)

def compare_frames(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    score, _ = ssim(gray1, gray2, full=False)
    # score, diff = ssim(gray1, gray2, full=True) old_score, new_score
    
    # Compute percentage of changed pixels
    diff = cv2.absdiff(gray1, gray2)
    changed_pixels = np.count_nonzero(diff > 25)  # Threshold for change
    total_pixels = diff.size
    percentage_changed = (changed_pixels / total_pixels) * 100

    # diff = (diff * 255).astype("uint8")
    # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # changed_pixels = np.count_nonzero(thresh)
    # total_pixels = thresh.size
    # percentage_changed = (changed_pixels / total_pixels) * 100
    
    # Add a measure of sharpness
    laplacian = cv2.Laplacian(gray1, cv2.CV_64F).var()
    
    combined_score = (1 - score) * 100 + percentage_changed + laplacian
    return combined_score

def extract_text_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use morphological operations to connect nearby text
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return dilated
    # return eroded

@lru_cache(maxsize=128)
def compute_image_hash(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imagehash.phash(img_pil)

def compute_image_similarity(img1, img2, img3, similarity_measures):
    similarities = []
    weights = []

    if similarity_measures.get('ssim', False):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        ssim_score1, _ = ssim(gray1, gray2, full=False)
        ssim_score2, _ = ssim(gray2, gray3, full=False)
        ssim_score = (ssim_score1 + ssim_score2) / 2
        similarities.append(ssim_score)
        weights.append(1.0)
    
    if similarity_measures.get('hist', False):
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist3 = cv2.calcHist([img3], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_sim1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_sim2 = cv2.compareHist(hist2, hist3, cv2.HISTCMP_CORREL)
        hist_sim = (hist_sim1 + hist_sim2) / 2
        similarities.append(hist_sim)
        weights.append(1.0)
    
    if similarity_measures.get('phash', False):
        hash1 = compute_image_hash(img1)
        hash2 = compute_image_hash(img2)
        hash3 = compute_image_hash(img3)
        hash_sim1 = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
        hash_sim2 = 1 - (hash2 - hash3) / len(hash2.hash) ** 2
        hash_sim = (hash_sim1 + hash_sim2) / 2
        similarities.append(hash_sim)
        weights.append(1.0)
    
    if not similarities:
        return 0  # If no similarity measure is selected, consider images as completely different

    return sum(s * w for s, w in zip(similarities, weights)) / sum(weights)

# def compute_image_similarity(img1, img2, similarity_measures):
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
#     similarities = []
#     weights = []

#     if similarity_measures.get('ssim', False):
#         ssim_score, _ = ssim(gray1, gray2, full=True)
#         similarities.append(ssim_score)
#         weights.append(1.0)
    
#     if similarity_measures.get('hist', False):
#         hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
#         hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
#         hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
#         similarities.append(hist_sim)
#         weights.append(1.0)
    
#     if similarity_measures.get('phash', False):
#         img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)) 
#         img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)) 
#         hash1 = imagehash.phash(img1_pil) 
#         hash2 = imagehash.phash(img2_pil) 
#         hash_sim = 1 - (hash1 - hash2) / len(hash1.hash) ** 2 # Normalize the hash similarity
#         similarities.append(hash_sim)
#         weights.append(1.0)
    
#     if not similarities:
#         return 0  # If no similarity measure is selected, consider images as completely different

#     return sum(s * w for s, w in zip(similarities, weights)) / sum(weights)

# - SSIM compares local patterns of pixel intensities across the two images.
# - It considers luminance, contrast, and structure.
# - The result is a value between -1 and 1, where 1 indicates perfect similarity.
# - It's effective at detecting structural changes in slides, like added or removed content.

# - It calculates histograms of pixel intensities for both images.
# - cv2.compareHist compares these histograms using correlation.
# - The result is a value between -1 and 1, where 1 indicates perfect similarity.
# - It's good at detecting overall changes in image brightness or contrast.

# - It reduces images to a small, fixed size and converts to grayscale.
# - It computes the discrete cosine transform (DCT) of the image.
# - A hash is generated based on frequency domain representation.
# - The similarity is calculated by comparing these hashes.
# - It's robust against minor changes and can detect similar images even after resizing or small edits.
