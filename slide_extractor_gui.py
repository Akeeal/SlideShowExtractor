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


import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, 
                             QSlider, QHBoxLayout, QProgressBar, QCheckBox, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt5.QtGui import QColor, QPalette, QDragEnterEvent, QDropEvent
import cv2
import numpy as np
from fpdf import FPDF
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import multiprocessing
import subprocess

# Function to compute the difference hash of an image
def dhash(image, hash_size=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return imagehash.ImageHash(diff)

# Function to process a single frame
def process_frame(args):
    frame, prev_frame, threshold = args
    difference = compare_frames(frame, prev_frame)
    if difference > threshold:
        return frame, dhash(frame)
    return None, None

# Function to compare two frames and return a similarity score
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

# Thread class for extracting slides from a video
class ExtractorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, video_path, output_path, threshold, fast_mode, format, similarity_threshold):
        QThread.__init__(self)
        self.video_path = video_path
        self.output_path = output_path
        self.threshold = threshold
        self.fast_mode = fast_mode
        self.format = format
        self.similarity_threshold = similarity_threshold
        self.is_running = True


    def run(self):
        # Main extraction logic
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        image_list = []
        image_hashes = set()

        if not cap.isOpened():
            self.finished.emit("Error opening video file")
            return

        ret, frame = cap.read()
        if not ret:
            self.finished.emit("Error reading the first frame")
            return

        prev_frame = frame
        image_list.append(frame)
        image_hashes.add(dhash(frame))

        frame_count = 0
        frame_skip = 1 if not self.fast_mode else 10
        batch_size = 100  # Process frames in batches

        with multiprocessing.Pool() as pool:
            while cap.isOpened() and self.is_running:
                frames = []
                for _ in range(batch_size):
                    for _ in range(frame_skip):
                        ret = cap.grab()
                        if not ret:
                            break
                    ret, frame = cap.retrieve()
                    if ret:
                        frames.append((frame, prev_frame, self.threshold))
                        prev_frame = frame
                        frame_count += frame_skip
                    else:
                        break

                if not frames:
                    break

                results = pool.map(process_frame, frames)
                for frame, frame_hash in results:
                    if frame is not None and frame_hash not in image_hashes:
                        image_list.append(frame)
                        image_hashes.add(frame_hash)

                self.progress.emit(int(frame_count / total_frames * 100))

        if self.is_running:
            # Apply additional filtering to reduce similar slides
            filtered_image_list = self.filter_similar_slides(image_list, self.similarity_threshold)
            
            if self.format == 'pdf':
                self.save_as_pdf(filtered_image_list)
            elif self.format in ['png', 'jpeg']:
                self.save_as_images(filtered_image_list, self.format)

        cap.release()
        cv2.destroyAllWindows()
        self.finished.emit(f"Slides extracted to: {self.output_path}" if self.is_running else "Extraction stopped")
    
    def filter_similar_slides(self, image_list, similarity_threshold=0.95):
    # Filter out similar slides based on similarity threshold

        filtered_images = []
        for i, current_image in enumerate(image_list):
            if i == 0:
                filtered_images.append(current_image)
                continue
            
            prev_image = filtered_images[-1]
            similarity = self.compute_image_similarity(prev_image, current_image)
            
            if similarity < similarity_threshold:
                filtered_images.append(current_image)
        
        return filtered_images

    def compute_image_similarity(self, img1, img2):
    # Compute similarity between two images
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between two images
        score, _ = ssim(gray1, gray2, full=True)
        return score

    def stop(self):
        self.is_running = False

    def save_as_pdf(self, image_list):
    # Save extracted slides as PDF
        pdf = FPDF(orientation='L')
        for idx, image in enumerate(image_list):
            temp_path = f'temp/temp{idx}.png'
            cv2.imwrite(temp_path, image)
            pdf.add_page()
            pdf.image(temp_path, 0, 0, 297, 210)
            os.remove(temp_path)
        pdf.output(self.output_path, "F")

    def save_as_images(self, image_list, format):
    # Save extracted slides as individual images
        for idx, image in enumerate(image_list):
            filename = f'slide_{idx:03d}.{format}'
            cv2.imwrite(os.path.join(self.output_path, filename), image)

# Main GUI class for the Slide Extractor application
class SlideExtractorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Initialize the user interface
        self.setWindowTitle('Slide Extractor - By Akeeal')
        self.setFixedSize(400, 500)  # Increased height to accommodate new buttons
        self.setAcceptDrops(True)
        
        # Set color palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor('#F0F0F0'))
        palette.setColor(QPalette.WindowText, QColor('#799496'))
        palette.setColor(QPalette.Button, QColor('#4A90E2'))
        palette.setColor(QPalette.ButtonText, QColor('#799496'))
        palette.setColor(QPalette.Highlight, QColor('#00E5E8'))
        self.setPalette(palette)
        
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
                background-color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setFormat("%p%")  # Show percentage in the progress bar
        layout.addWidget(self.progress_bar)
        
        self.file_label = QLabel('Drag and drop video file here or click "Select Video File"')
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet("""
            padding: 10px;
            border: 2px dashed #CCCCCC;
            border-radius: 5px;
            color: #000000;
        """)
        layout.addWidget(self.file_label)
        
        # slider_layout = QHBoxLayout()
        # Threshold slider
        self.slider_label = QLabel('Threshold: 2.0')
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(20)  # 2.0 in float
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)  # 1.0 in float
        self.slider.valueChanged.connect(self.update_slider_label)
        
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)
        layout.addLayout(slider_layout)

        # Similar slides removal slider
        self.similarity_slider_label = QLabel('Similar Slides Removal: 0.90')
        self.similarity_slider_label.setAlignment(Qt.AlignCenter)
        self.similarity_slider = QSlider(Qt.Horizontal)
        self.similarity_slider.setMinimum(0)
        self.similarity_slider.setMaximum(200)  # 0.8 to 1.0 in steps of 0.001
        self.similarity_slider.setValue(100)  # 0.90 in float
        self.similarity_slider.setTickPosition(QSlider.TicksBelow)
        self.similarity_slider.setTickInterval(20)  # 0.02 in float
        self.similarity_slider.valueChanged.connect(self.update_similarity_slider_label)
        
        similarity_slider_layout = QVBoxLayout()
        similarity_slider_layout.addWidget(self.similarity_slider_label)
        similarity_slider_layout.addWidget(self.similarity_slider)
        layout.addLayout(similarity_slider_layout)


        # Fast Mode checkbox
        self.fast_mode_checkbox = QCheckBox('Fast Mode')
        self.fast_mode_checkbox.setChecked(True)  # Set Fast Mode on by default
        layout.addWidget(self.fast_mode_checkbox)

        
        self.select_file_button = QPushButton('Select Video File')
        self.select_file_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_file_button)
        
        self.extract_pdf_button = QPushButton('Extract to PDF')
        self.extract_pdf_button.clicked.connect(self.extract_slides_pdf)
        self.extract_pdf_button.setEnabled(False)
        layout.addWidget(self.extract_pdf_button)
        
        self.extract_png_button = QPushButton('Extract to PNG')
        self.extract_png_button.clicked.connect(self.extract_slides_png)
        self.extract_png_button.setEnabled(False)
        layout.addWidget(self.extract_png_button)
        
        self.extract_jpeg_button = QPushButton('Extract to JPEG')
        self.extract_jpeg_button.clicked.connect(self.extract_slides_jpeg)
        self.extract_jpeg_button.setEnabled(False)
        layout.addWidget(self.extract_jpeg_button)
        
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        self.open_output_button = QPushButton('Open Output')
        self.open_output_button.clicked.connect(self.open_output)
        self.open_output_button.setEnabled(False)
        layout.addWidget(self.open_output_button)
        
        self.quit_button = QPushButton('Quit')
        self.quit_button.clicked.connect(QApplication.instance().quit)
        layout.addWidget(self.quit_button)
        
        self.setLayout(layout)
        
        self.video_path = None
        self.output_path = None
        self.extractor_thread = None

    def update_slider_label(self, value):
        # Update the label for the threshold slider
        float_value = value / 10.0
        self.slider_label.setText(f'Threshold: {float_value:.1f}')

    def update_similarity_slider_label(self, value):
        # Update the label for the similarity threshold slider
        float_value = 0.8 + (value / 1000)
        self.similarity_slider_label.setText(f'Similar Slides Removal: {float_value:.3f}')

    def select_file(self):
        # Open file dialog to select a video file
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if self.video_path:
            self.file_label.setText(f'Selected: {os.path.basename(self.video_path)}')
            self.extract_pdf_button.setEnabled(True)
            self.extract_png_button.setEnabled(True)
            self.extract_jpeg_button.setEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Handle drag enter events for file dropping
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        # Handle drop events for file dropping
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.video_path = f
                self.file_label.setText(f'Selected: {os.path.basename(f)}')
                self.extract_pdf_button.setEnabled(True)
                self.extract_png_button.setEnabled(True)
                self.extract_jpeg_button.setEnabled(True)
                break

    def extract_slides_pdf(self):
        self.extract_slides('pdf')

    def extract_slides_png(self):
        self.extract_slides('png')

    def extract_slides_jpeg(self):
        self.extract_slides('jpeg')

    def extract_slides(self, format):
        # Main method to start the slide extraction process
        if self.video_path:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            if format == 'pdf':
                self.output_path = self.get_unique_filename(os.path.dirname(self.video_path), f'{base_name}.pdf')
            else:
                self.output_path = self.get_unique_filename(os.path.dirname(self.video_path), f'{base_name}_{format}')
                os.makedirs(self.output_path, exist_ok=True)
            
            threshold = self.slider.value() / 10.0
            similarity_threshold = 0.8 + (self.similarity_slider.value() / 1000)
            fast_mode = self.fast_mode_checkbox.isChecked()
            self.extractor_thread = ExtractorThread(self.video_path, self.output_path, threshold, fast_mode, format, similarity_threshold)

            self.extractor_thread.progress.connect(self.update_progress)
            self.extractor_thread.finished.connect(self.extraction_finished)
            
            self.extractor_thread.start()
            self.extract_pdf_button.setEnabled(False)
            self.extract_png_button.setEnabled(False)
            self.extract_jpeg_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.slider.setEnabled(False)
            self.similarity_slider.setEnabled(False)  # Disable similarity slider during extraction
            self.fast_mode_checkbox.setEnabled(False)
            self.select_file_button.setEnabled(False)

    def stop_extraction(self):
        # Stop the ongoing extraction process
        if self.extractor_thread and self.extractor_thread.isRunning():
            self.extractor_thread.stop()
            self.stop_button.setEnabled(False)

    def update_progress(self, value):
        # Update the progress bar
        self.progress_bar.setValue(value)

    def extraction_finished(self, message):
        # Handle the completion of the extraction process
        self.file_label.setText(message)
        self.progress_bar.setValue(100)
        self.stop_button.setEnabled(False)
        self.extract_pdf_button.setEnabled(True)
        self.extract_png_button.setEnabled(True)
        self.extract_jpeg_button.setEnabled(True)
        self.slider.setEnabled(True)
        self.similarity_slider.setEnabled(True)  # Re-enable the similarity slider
        self.fast_mode_checkbox.setEnabled(True)
        self.select_file_button.setEnabled(True)
        self.open_output_button.setEnabled(True)

    def open_output(self):
        # Open the output folder or file
        if self.output_path and os.path.exists(self.output_path):
            if sys.platform == 'darwin':  # macOS
                subprocess.call(('open', self.output_path))
            elif sys.platform == 'win32':  # Windows
                os.startfile(self.output_path)
            else:  # linux variants
                subprocess.call(('xdg-open', self.output_path))

    def get_unique_filename(self, directory, filename):
        # Generate a unique filename to avoid overwriting
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(directory, filename)):
            filename = f"{name}_{counter}{ext}"
            counter += 1
        return os.path.join(directory, filename)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SlideExtractorGUI()
    ex.show()
    sys.exit(app.exec_())

