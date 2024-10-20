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

import tempfile
import time
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from fpdf import FPDF
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import imagehash


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
        self.cap = None
        self.total_frames = 0
        self.frame_count = 0
        self.image_list = []
        self.image_hashes = set()
        self.fps = 0
        self.is_running = True

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        if not self.cap.isOpened():
            self.finished.emit("Error opening video file")
            return

        try:
            if self.fixed_interval:
                self.extract_fixed_interval()
            else:
                self.extract_adaptive()

            self.cap.release()

            if self.is_running:
                self.finished.emit("Extraction completed successfully")
                filtered_image_list = self.filter_similar_slides(self.image_list, self.similarity_threshold)
            else:
                self.finished.emit("Extraction stopped by user")
        except Exception as e:
            self.finished.emit(f"Error during extraction: {str(e)}")
        finally:
            self.cleanup()

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
        
    def cleanup(self):
        # Release any resources, close files, etc.
        pass  # Implement any necessary cleanup logic
    
    def save_as_pdf(self, image_list):
        pdf = None
        temp_dir = None
        retry_count = 0
        max_retries = 3
        
        try:
            temp_dir = tempfile.mkdtemp()
            pdf = FPDF(orientation='L')
            
            for idx, image in enumerate(image_list):
                temp_image_path = os.path.join(temp_dir, f'temp_image_{idx}.png')
                cv2.imwrite(temp_image_path, image)
                pdf.add_page()
                pdf.image(temp_image_path, 0, 0, 297, 210)

            while retry_count < max_retries:
                try:
                    pdf.output(self.output_path, "F")
                    print(f"PDF saved to: {self.output_path}")  # Debug print
                    break
                except PermissionError:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    time.sleep(1)  # Wait for 1 second before retrying
        
        except Exception as e:
            raise Exception(f"Error saving PDF: {str(e)}")
        
        finally:
            if pdf:
                del pdf  # Ensure the PDF object is deleted
            
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

    def save_as_images(self, image_list, format):
        try:
            os.makedirs(self.output_path, exist_ok=True)
            for idx, image in enumerate(image_list):
                filename = f'slide_{idx:03d}.{format}'
                output_file = os.path.join(self.output_path, filename)
                
                if format.lower() == 'png':
                    cv2.imwrite(output_file, image)
                elif format.lower() in ['jpeg', 'jpg']:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    Image.fromarray(image_rgb).save(output_file, format='JPEG', quality=95)
                else:
                    raise ValueError(f"Unsupported image format: {format}")
            print(f"Images saved to: {self.output_path}")  # Debug print
        except Exception as e:
            raise Exception(f"Error saving images: {str(e)}")

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Error opening video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frame_interval = 1
            if self.fast_mode:
                frame_interval = max(1, fps // 2)  # Process 2 frames per second in fast mode
            elif self.fixed_interval:
                frame_interval = int(self.fixed_interval * fps)

            prev_frame = None
            prev_hash = None
            frame_count = 0

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                if prev_frame is None:
                    self.image_list.append(frame)
                    prev_frame = frame
                    prev_hash = self.compute_image_hash(frame)
                else:
                    if self.is_new_slide(frame, prev_frame, prev_hash):
                        if not self.remove_fading_text or not self.is_fading_text(frame, prev_frame):
                            self.image_list.append(frame)
                            prev_frame = frame
                            prev_hash = self.compute_image_hash(frame)

                progress = int((frame_count / total_frames) * 100)
                self.progress.emit(progress)

            cap.release()

            if self.is_running:
                if self.format == 'pdf':
                    self.save_as_pdf(self.image_list)
                else:
                    self.save_as_images(self.image_list, self.format)
                self.finished.emit("Extraction completed successfully")
            else:
                self.finished.emit("Extraction stopped by user")
        except Exception as e:
            self.finished.emit(f"Error during extraction: {str(e)}")
        finally:
            self.cleanup()

    def is_new_slide(self, current_frame, prev_frame, prev_hash):
        current_hash = self.compute_image_hash(current_frame)
        hash_diff = current_hash - prev_hash

        if hash_diff >= self.similarity_threshold:
            return True

        if self.similarity_measures.get('ssim', False):
            ssim_score = self.compute_ssim(current_frame, prev_frame)
            if ssim_score < self.similarity_threshold:
                return True

        if self.similarity_measures.get('hist', False):
            hist_diff = self.compute_histogram_difference(current_frame, prev_frame)
            if hist_diff > 1 - self.similarity_threshold:
                return True

        return False

    def compute_image_hash(self, image):
        return imagehash.phash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    def compute_ssim(self, image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2)

    def compute_histogram_difference(self, image1, image2):
        hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def is_fading_text(self, current_frame, prev_frame):
        diff = cv2.absdiff(current_frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        return np.sum(thresh) / thresh.size < 0.01  # Adjust threshold as needed

    def cleanup(self):
        # Implement any necessary cleanup logic
        self.image_list.clear()
        # Release any other resources if needed