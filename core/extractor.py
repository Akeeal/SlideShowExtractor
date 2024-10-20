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

from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image
import os

from .image_processing import dhash, compare_frames, ssim, imagehash

class ExtractorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, video_path, output_path, fast_mode, format, similarity_threshold, fixed_interval, remove_fading_text, similarity_measures):
        super(ExtractorThread, self).__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.fast_mode = fast_mode
        self.format = format
        self.similarity_threshold = similarity_threshold
        self.similarity_measures = similarity_measures
        self.fixed_interval = fixed_interval
        self.remove_fading_text = remove_fading_text
        self.is_running = True
        self.cap = None
        self.total_frames = 0
        self.frame_count = 0
        self.image_list = []
        self.image_hashes = set()
        self.fps = 0

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        if not self.cap.isOpened():
            self.finished.emit("Error opening video file")
            return

        if self.fixed_interval:
            self.extract_fixed_interval()
        else:
            self.extract_adaptive()

        self.cap.release()

        if self.is_running:
            filtered_image_list = self.filter_similar_slides(self.image_list, self.similarity_threshold)
            
            if self.format == 'pdf':
                self.save_as_pdf(filtered_image_list)
            elif self.format in ['png', 'jpeg']:
                self.save_as_images(filtered_image_list, self.format)

        cv2.destroyAllWindows()
        self.finished.emit(f"Slides extracted to: {self.output_path}" if self.is_running else "Extraction stopped")

    def extract_fixed_interval(self):
        interval_frames = int(self.fixed_interval * self.fps)
        while self.cap.isOpened() and self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.frame_count % interval_frames == 0:
                self.image_list.append(frame)
                self.image_hashes.add(dhash(frame))

            self.frame_count += 1
            self.progress.emit(int(self.frame_count / self.total_frames * 100))

    def extract_adaptive(self):
        frame_skip = max(1, int(self.fps / 10)) if self.fast_mode else 1
        prev_frame = None
        prev_text_mask = None

        while self.cap.isOpened() and self.is_running:
            for _ in range(frame_skip):
                ret = self.cap.grab()
                if not ret:
                    break
                self.frame_count += 1

            ret, frame = self.cap.retrieve()
            if not ret:
                break

            if prev_frame is not None:
                if not self.remove_fading_text or not self.is_fading_text(frame, prev_frame, prev_text_mask):
                    self.image_list.append(frame)
                    self.image_hashes.add(dhash(frame))
                    if self.remove_fading_text:
                        prev_text_mask = self.extract_text_mask(frame)

            prev_frame = frame
            self.progress.emit(int(self.frame_count / self.total_frames * 100))

    def is_fading_text(self, current_frame, prev_frame, prev_text_mask):
        if prev_text_mask is None:
            return False

        current_text_mask = self.extract_text_mask(current_frame)
        
        # Calculate the difference in text areas
        text_diff = cv2.absdiff(prev_text_mask, current_text_mask)
        text_change = np.sum(text_diff) / (prev_text_mask.size * 255)

        # If there's a significant change in text areas, it might be fading text
        return text_change > 0.01  # Adjust this threshold as needed

    def extract_text_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use morphological operations to connect nearby text
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        return eroded

    def filter_similar_slides(self, image_list, similarity_threshold):
        if len(image_list) <= 1:
            return image_list

        filtered_images = [image_list[0]]
        for current_image in image_list[1:]:
            is_similar = False
            for filtered_image in filtered_images[-3:]:  # Compare with last 3 filtered images
                similarity = self.compute_image_similarity(filtered_image, current_image)
                if similarity > similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                filtered_images.append(current_image)
        
        return filtered_images

    def compute_image_similarity(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        similarities = []
        weights = []

        if self.similarity_measures['ssim']:
            ssim_score, _ = ssim(gray1, gray2, full=True)
            similarities.append(ssim_score)
            weights.append(1.0)
        
        if self.similarity_measures['hist']:
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarities.append(hist_sim)
            weights.append(1.0)
        
        if self.similarity_measures['phash']:
            img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            hash1 = imagehash.phash(img1_pil)
            hash2 = imagehash.phash(img2_pil)
            hash_sim = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
            similarities.append(hash_sim)
            weights.append(1.0)
        
        if not similarities:
            return 0  # If no similarity measure is selected, consider images as completely different

        return sum(s * w for s, w in zip(similarities, weights)) / sum(weights)

    def stop(self):
        self.is_running = False
    
    def save_as_pdf(self, image_list):
        # Create a temporary directory
        temp_dir = os.path.join(os.path.dirname(self.output_path), 'temp_slides')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            pdf = FPDF(orientation='L')
            for idx, image in enumerate(image_list):
                temp_path = os.path.join(temp_dir, f'temp{idx}.png')
                cv2.imwrite(temp_path, image)
                pdf.add_page()
                pdf.image(temp_path, 0, 0, 297, 210)
            pdf.output(self.output_path, "F")
        except Exception as e:
            print(f"Error saving PDF: {str(e)}")
            self.finished.emit(f"Error saving PDF: {str(e)}")
        finally:
            # Clean up temporary files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    def save_as_images(self, image_list, format):
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        try:
            for idx, image in enumerate(image_list):
                filename = f'slide_{idx:03d}.{format}'
                output_file = os.path.join(self.output_path, filename)
                
                # For PNG format
                if format.lower() == 'png':
                    cv2.imwrite(output_file, image)
                
                # For JPEG format
                elif format.lower() == 'jpeg' or format.lower() == 'jpg':
                    # Convert from BGR to RGB color space
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Use PIL for JPEG saving to ensure better quality control
                    pil_image = Image.fromarray(image_rgb)
                    pil_image.save(output_file, format='JPEG', quality=95)
                
                else:
                    raise ValueError(f"Unsupported image format: {format}")

            # Emit progress updates
            progress = int((idx + 1) / len(image_list) * 100)
            self.progress.emit(progress)

        except Exception as e:
            error_message = f"Error saving images: {str(e)}"
            print(error_message)
            self.finished.emit(error_message)
        else:
            success_message = f"Slides extracted to: {self.output_path}"
            self.finished.emit(success_message)