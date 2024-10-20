# Slide Show Extractor GUI version

<img src="https://github.com/Akeeal/SlideShowExtractor/blob/main/screenShotWin.png" width=50% height=50%>

Slide Extractor the GUI version is a Python application that extracts slides from video files. It uses computer vision techniques to identify unique slides and can output them as PDF, PNG, or JPEG files. I couldn't find one GUI. So I created it. I use it regularly I hope it helps other students out there! There is a working Windows and MacOS versions if there are any issues let me know otherwise have a lovely day.

If you like this please consider buying me a coffee, support me in my future projects

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/akeeal)

### Features
---

* NEW - New Algorithms SSIM, Histogram and Perceptual Hash
  
- Extract slides from various video formats (MP4, AVI, MOV, MKV)
- Output slides as PDF, PNG, or JPEG into a folder ( example 'name_png' )
- Adjustable threshold for slide detection
- Similar slide removal to reduce duplicates
- Fast mode for quicker processing
- User-friendly GUI with drag-and-drop functionality

### Installation
---

Ensure you have installed Python 3.7 or later on your system.
Clone this repository or download the source code.

```bash
  git clone https://github.com/Akeeal/SlideShowExtractor.git
```

### Change directory
```bash
  cd SlideShowExtractor
```

### Install the required dependencies:
```bash
  pip install PyQt5 opencv-python numpy fpdf gh scikit-image imagehash Pillow
```
OR
```bash
  pip install -r requirements.txt
```

### Run the application in MacOS:
```bash
  python slide_extractor_gui.py
```

### Run the application in Windows:
```bash
  python slide_extractor_gui_win.py
```

### Usage
---

Launch the application

Select a video file:

Click the "Select Video File" button, 
or
Drag and drop a video file onto the application window.

### Adjust the settings:
---

~~Threshold: Controls the sensitivity of slide detection. Lower values detect more subtle changes.~~
~~Similar Slides Removal: Adjusts how aggressively similar slides are filtered out.~~
Fast Mode: Skips frames for quicker processing (enabled by default).~~

### Choose the output format:
---

Click "Extract to PDF" for a single PDF file containing all slides.

Click "Extract to PNG" or "Extract to JPEG" for individual image files.

Wait for the extraction process to complete.
Click "Open Output" to view the extracted slides.

Recommendations
---

### Start with the default settings and adjust as needed:
## These settings usually work for most videos.

Threshold: 5.0
Similar Slides Removal: 1.00
Fast Mode: Enabled

### If you're missing slides:
---

~~*Decrease the Threshold value*~~

*Decrease the Remove Similar Slides value*

*Disable Fast Mode*

### If you're getting too many duplicate or similar slides:

---

~~*Increase the Threshold value*~~

*Increase the Remove Similar Slides value value*

For long videos, use Fast Mode to reduce processing time. Disable it if you notice missing slides.
The PDF output is great for an overview, while PNG or JPEG outputs are useful if you need to edit individual slides.
For presentations with animations or gradual build-ups, you may need to lower the Similar Slides Removal value to capture all steps.
Test the extraction on a short segment of your video first to fine-tune the settings before processing the entire file.

### Troubleshooting
---
If the application crashes or freezes, try processing the video in smaller segments.
Ensure you have sufficient free disk space for the output files, especially when extracting long videos as images.
For very large video files, consider using Fast Mode and increasing the Threshold to reduce processing time and memory usage.

## Roadmap

- ~~Create a release version for Windows and Mac~~
- ~~Improve on user interface~~
- ~~Optimise code for speed~~
- Add more integrations? Any ideas

---
Contributing
Contributions to improve Slide Extractor are welcome. Please feel free to submit pull requests or open issues on the GitHub repository.
License
Apache License

If you like this please consider buying me a coffee

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/akeeal)
