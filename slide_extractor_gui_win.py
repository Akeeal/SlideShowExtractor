import sys
import os
from pathlib import Path
import tempfile
import multiprocessing
import subprocess
import gc

from PyQt5.QtWidgets import (QApplication, QMessageBox, QWidget, QLabel, QVBoxLayout, QPushButton, 
                             QSlider, QHBoxLayout, QProgressBar, QCheckBox, QFileDialog, QLineEdit,
                             QSpacerItem, QSizePolicy, QFrame, QToolTip)
from PyQt5.QtCore import Qt, QMimeData, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QDragEnterEvent, QDropEvent, QCursor

# Import custom modules
from gui.widgets import ClickableLabel
from core.extractor import ExtractorThread
from utils.single_instance import SingleInstance
from utils.file_operations import get_unique_filename

def setup_global_style():
    app = QApplication.instance()
    app.setStyleSheet("""
        QToolTip {
            background-color: #2980b9;
            color: black;
            border: 2px solid #1c4966;
            padding: 5px;
            border-radius: 3px;
            font-size: 12px;
        }
    """)

class BorderedFrame(QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("""
            BorderedFrame {
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                padding: 1px;
                margin: 1px;
                background-color: white;
            }
            BorderedFrame QLabel, 
            BorderedFrame QCheckBox, 
            BorderedFrame QLineEdit,
            BorderedFrame QSlider::handle:horizontal {
                color: black;
            }
        """)

class HoverButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QPushButton {
                background-color: #6BA5E7;
                font-size: 14px;
                color: white;
                border: 2px;
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #4A90E2;
                font-size: 15px;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.showTooltip)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self.timer.start(2000)

    def leaveEvent(self, event):
        self.timer.stop()
        QToolTip.hideText()

    def showTooltip(self):
        QToolTip.showText(QCursor.pos(), self.toolTip())

class SlideExtractorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.extraction_in_progress = False
        self.video_path = None
        self.output_path = None
        self.extractor_thread = None

    def initUI(self):
        self.setWindowTitle('Slide Extractor - By Akeeal')
        self.setFixedSize(700, 650)
        self.setAcceptDrops(True)
        
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor('#F0F0F0'))
        palette.setColor(QPalette.WindowText, QColor('black'))
        palette.setColor(QPalette.Button, QColor('#4A90E2'))
        palette.setColor(QPalette.ButtonText, QColor('#799496'))
        palette.setColor(QPalette.Highlight, QColor('#00E5E8'))
        self.setPalette(palette)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        left_layout = QVBoxLayout()
        left_layout.addStretch(1)
        self.file_label = ClickableLabel('Click here \n or \n drag and drop video file')
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                border: 5px dashed #CCCCCC;
                border-radius: 5px;
                color: #000000;
                background-color: #F8F8F8;
                font-size: 16px;
            }
            QLabel:hover {
                background-color: #E8E8E8;
            }
        """)
        self.file_label.setFixedSize(300, 600)
        self.file_label.clicked.connect(self.select_file)
        self.file_label.setToolTip("Click here to select a video file or drag and drop a file onto this area")
        left_layout.addWidget(self.file_label)
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
                background-color: #FFFFFF;
                color: black;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setFormat("%p%")
        right_layout.addWidget(self.progress_bar)
        
        similarity_frame = BorderedFrame()
        similarity_layout = QVBoxLayout(similarity_frame)
        self.similarity_slider_label = QLabel('Removes Similar Slides By: 98% Threshold')
        self.similarity_slider_label.setToolTip('Adjust the similarity threshold for slide removal.\n'
                                                'Lower values will REMOVE more slides.\n'
                                                'Higher values will ADD more slides')
        self.similarity_slider_label.setAlignment(Qt.AlignCenter)
        self.similarity_slider = QSlider(Qt.Horizontal)
        self.similarity_slider.setMinimum(80)
        self.similarity_slider.setMaximum(100)
        self.similarity_slider.setValue(98)
        self.similarity_slider.setTickPosition(QSlider.TicksBelow)
        self.similarity_slider.setTickInterval(1)
        self.similarity_slider.valueChanged.connect(self.update_similarity_slider_label)
        similarity_layout.addWidget(self.similarity_slider_label)
        similarity_layout.addWidget(self.similarity_slider)
        right_layout.addWidget(similarity_frame)

        interval_frame = BorderedFrame()
        interval_frame.setObjectName("FixedIntervalFrame")
        interval_frame.setStyleSheet("""
            #FixedIntervalFrame {
                background-color: white;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
            }
            #FixedIntervalFrame QCheckBox {
                background-color: transparent;
                color: black;                     
            }
            #FixedIntervalFrame QLineEdit {
                background-color: white;
                color: black;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
            }
        """)
        interval_layout = QHBoxLayout(interval_frame)
        self.fixed_interval_checkbox = QCheckBox('Extract at Fixed Interval')
        self.fixed_interval_checkbox.setChecked(True)
        self.fixed_interval_input = QLineEdit()
        self.fixed_interval_input.setPlaceholderText('Interval in seconds')
        self.fixed_interval_input.setText('3')
        self.fixed_interval_input.setEnabled(True)
        self.fixed_interval_input.setToolTip('Enter the interval in seconds at which to extract slides.')
        self.fixed_interval_checkbox.stateChanged.connect(self.toggle_fixed_interval)
        interval_layout.addWidget(self.fixed_interval_checkbox)
        interval_layout.addWidget(self.fixed_interval_input)
        right_layout.addWidget(interval_frame)

        fading_text_frame = BorderedFrame()
        fading_text_layout = QVBoxLayout(fading_text_frame)
        self.remove_fading_text_checkbox = QCheckBox('Remove Slides with Fading Text')
        self.remove_fading_text_checkbox.setChecked(True)
        self.remove_fading_text_checkbox.setToolTip('EXPERIMENTAL - Removes slides with fading text.')
        fading_text_layout.addWidget(self.remove_fading_text_checkbox)
        right_layout.addWidget(fading_text_frame)

        similarity_measures_frame = BorderedFrame()
        similarity_measures_layout = QHBoxLayout(similarity_measures_frame)
        self.ssim_checkbox = QCheckBox('SSIM')
        self.ssim_checkbox.setChecked(True)
        self.ssim_checkbox.setToolTip('EXPERIMENTAL - Structural Similarity Index')
        self.hist_checkbox = QCheckBox('Histogram')
        self.hist_checkbox.setChecked(True)
        self.hist_checkbox.setToolTip('EXPERIMENTAL - Histogram comparison')
        self.phash_checkbox = QCheckBox('Perceptual Hash')
        self.phash_checkbox.setChecked(True)
        self.phash_checkbox.setToolTip('EXPERIMENTAL - Perceptual hashing')
        similarity_measures_layout.addWidget(self.ssim_checkbox)
        similarity_measures_layout.addWidget(self.hist_checkbox)
        similarity_measures_layout.addWidget(self.phash_checkbox)
        right_layout.addWidget(similarity_measures_frame)

        fast_mode_frame = BorderedFrame()
        fast_mode_layout = QVBoxLayout(fast_mode_frame)
        self.fast_mode_checkbox = QCheckBox('Fast Mode')
        self.fast_mode_checkbox.setChecked(True)
        self.fast_mode_checkbox.setToolTip("Skips frames for faster processing, but may result in lower quality output.")
        fast_mode_layout.addWidget(self.fast_mode_checkbox)
        right_layout.addWidget(fast_mode_frame)

        self.extract_pdf_button = HoverButton('Extract to PDF')
        self.extract_pdf_button.clicked.connect(self.extract_slides_pdf)
        self.extract_pdf_button.setEnabled(False)
        self.extract_pdf_button.setToolTip("Extract slides from the video and save as a PDF file")
        right_layout.addWidget(self.extract_pdf_button)
        
        self.extract_png_button = HoverButton('Extract to PNG')
        self.extract_png_button.clicked.connect(self.extract_slides_png)
        self.extract_png_button.setEnabled(False)
        self.extract_png_button.setToolTip("Extract slides from the video and save as PNG images")
        right_layout.addWidget(self.extract_png_button)

        self.extract_jpeg_button = HoverButton('Extract to JPEG')
        self.extract_jpeg_button.clicked.connect(self.extract_slides_jpeg)
        self.extract_jpeg_button.setEnabled(False)
        self.extract_jpeg_button.setToolTip("Extract slides from the video and save as JPEG images")
        right_layout.addWidget(self.extract_jpeg_button)

        self.stop_button = HoverButton('Stop')
        self.stop_button.clicked.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop the current extraction process")
        right_layout.addWidget(self.stop_button)

        self.quit_button = HoverButton('Quit')
        self.quit_button.clicked.connect(QApplication.instance().quit)
        self.quit_button.setToolTip("Exit the application")
        right_layout.addWidget(self.quit_button)

        right_layout.insertStretch(1, 1)
        right_layout.insertStretch(3, 1)
        right_layout.insertStretch(5, 1)
        right_layout.insertStretch(7, 1)
        right_layout.insertStretch(9, 1)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        
        self.video_path = None
        self.output_path = None
        self.extractor_thread = None
        print("UI initialization complete")
        
    def update_similarity_slider_label(self, value):
        self.similarity_slider_label.setText(f'Removes Similar Slides By: {value}% Threshold')

    def select_file(self):
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if self.video_path:
            self.file_label.setText(f'Selected: {os.path.basename(self.video_path)}')
            self.extract_pdf_button.setEnabled(True)
            self.extract_png_button.setEnabled(True)
            self.extract_jpeg_button.setEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
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

    def toggle_fixed_interval(self, state):
        self.fixed_interval_input.setEnabled(state == Qt.Checked)
        if state == Qt.Checked and not self.fixed_interval_input.text():
            self.fixed_interval_input.setText('3')

    def stop_extraction(self):
        try:
            if self.extractor_thread and self.extractor_thread.isRunning():
                self.extractor_thread.stop()
                self.extractor_thread.wait(5000)  # Wait for up to 5 seconds
                if self.extractor_thread.isRunning():
                    self.extractor_thread.terminate()  # Force termination if still running
            self.extraction_in_progress = False
            self.update_ui_for_extraction_end()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop extraction: {str(e)}")

    def extract_slides(self, format):
        if self.video_path:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            if format == 'pdf':
                self.output_path = get_unique_filename(os.path.dirname(self.video_path), f'{base_name}.pdf')
            else:
                self.output_path = get_unique_filename(os.path.dirname(self.video_path), f'{base_name}_{format}')
                os.makedirs(self.output_path, exist_ok=True)

            similarity_threshold = self.similarity_slider.value() / 100.0
            fast_mode = self.fast_mode_checkbox.isChecked()
            
            fixed_interval = None
            if self.fixed_interval_checkbox.isChecked():
                try:
                    fixed_interval = float(self.fixed_interval_input.text())
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the fixed interval.")
                    return

            remove_fading_text = self.remove_fading_text_checkbox.isChecked()

            similarity_measures = {
                'ssim': self.ssim_checkbox.isChecked(),
                'hist': self.hist_checkbox.isChecked(),
                'phash': self.phash_checkbox.isChecked()
            }

            try:
                self.extractor_thread = ExtractorThread(
                    self.video_path, 
                    self.output_path, 
                    fast_mode, 
                    format, 
                    similarity_threshold, 
                    fixed_interval,
                    remove_fading_text,
                    similarity_measures
                )

                self.extractor_thread.progress.connect(self.update_progress)
                self.extractor_thread.finished.connect(self.extraction_finished)
                
                self.extraction_in_progress = True
                self.extractor_thread.start()
                
                self.update_ui_for_extraction_start()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start extraction: {str(e)}")
                self.extraction_in_progress = False
                
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def extraction_finished(self, message):
        try:
            self.extraction_in_progress = False
            self.update_ui_for_extraction_end()
            self.file_label.setText(message)
            self.progress_bar.setValue(100)
            
            if "Error" in message:
                QMessageBox.warning(self, "Extraction Issue", message)
            else:
                QMessageBox.information(self, "Extraction Complete", message)
                # Use QTimer to delay opening the output
                QTimer.singleShot(1000, self.open_output)  # Increased delay to 1 second
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred after extraction: {str(e)}")

    def update_ui_for_extraction_start(self):
        self.extract_pdf_button.setEnabled(False)
        self.extract_png_button.setEnabled(False)
        self.extract_jpeg_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.similarity_slider.setEnabled(False)
        self.fast_mode_checkbox.setEnabled(False)
        self.fixed_interval_checkbox.setEnabled(False)
        self.fixed_interval_input.setEnabled(False)
        self.ssim_checkbox.setEnabled(False)
        self.hist_checkbox.setEnabled(False)
        self.phash_checkbox.setEnabled(False)
        self.remove_fading_text_checkbox.setEnabled(False)

    def update_ui_for_extraction_end(self):
        self.stop_button.setEnabled(False)
        self.extract_pdf_button.setEnabled(True)
        self.extract_png_button.setEnabled(True)
        self.extract_jpeg_button.setEnabled(True)
        self.similarity_slider.setEnabled(True)
        self.fast_mode_checkbox.setEnabled(True)
        self.fixed_interval_checkbox.setEnabled(True)
        self.fixed_interval_input.setEnabled(self.fixed_interval_checkbox.isChecked())
        self.ssim_checkbox.setEnabled(True)
        self.hist_checkbox.setEnabled(True)
        self.phash_checkbox.setEnabled(True)
        self.remove_fading_text_checkbox.setEnabled(True)

    def open_output(self):
        try:
            if self.output_path and os.path.exists(self.output_path):
                if os.path.isfile(self.output_path):  # For PDF output
                    if sys.platform == 'win32':  # Windows
                        os.startfile(self.output_path)
                    else:  # Linux variants
                        subprocess.call(('xdg-open', self.output_path))
                elif os.path.isdir(self.output_path):  # For PNG/JPEG output
                    if sys.platform == 'win32':  # Windows
                        os.startfile(self.output_path)
                    else:  # Linux variants
                        subprocess.call(('xdg-open', self.output_path))
                else:
                    QMessageBox.warning(self, "Error", f"Output path exists but is neither a file nor a directory: {self.output_path}")
            else:
                QMessageBox.warning(self, "Error", f"Output path does not exist: {self.output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open output: {str(e)}")

    def closeEvent(self, event):
        if self.extraction_in_progress:
            reply = QMessageBox.question(self, 'Window Close', 'Extraction is in progress. Are you sure you want to close?',
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_extraction()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
        
        # Ensure all threads are stopped and resources are released
        if self.extractor_thread:
            self.extractor_thread.stop()
            self.extractor_thread.wait()
        QApplication.instance().quit()

    
def setup_global_tooltip_style():
    QToolTip.setFont(QApplication.font())
    QToolTip.setStyleSheet("""
        QToolTip {
            background-color: #2980b9;
            color: black;
            border: 2px solid #1c4966;
            padding: 5px;
            border-radius: 3px;
        }
    """)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    setup_global_style()
    single_instance = SingleInstance()

    if not single_instance.try_lock():
        QMessageBox.warning(None, "Application Already Running",
                            "An instance of Slide Extractor is already running.")
        sys.exit(1)

    ex = SlideExtractorGUI()
    ex.show()

    try:
        exit_code = app.exec_()
    finally:
        single_instance.unlock()
        del single_instance  # Explicitly delete the single_instance object

    sys.exit(exit_code)