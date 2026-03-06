EchoSight is an AI-powered accessibility system that helps visually impaired users understand their surroundings. Using computer vision, it detects objects, faces, text, and obstacles in real time and converts them into audio feedback, enabling safer navigation and greater independence in everyday environments.

## 🧠 Model Pipeline Architecture

The application processes live webcam frames through two lightweight, concurrent pipelines designed for real-time performance on edge CPU devices.

### 1. Face Detection & Tracking Pipeline
This pipeline relies on traditional computer vision cascades combined with custom temporal tracking to track users and estimate their distance.

*   **Preprocessing:** Frames are converted to grayscale and enhanced using Histogram Equalization (`cv::equalizeHist`) to improve contrast and detection accuracy under varying lighting conditions.
*   **Detection:** Uses OpenCV's Haar Cascade Classifier ([haarcascade_frontalface_default.xml](cci:7://file:///c:/Users/Aryan/.gemini/antigravity/scratch/FaceTextRecognition/haarcascade_frontalface_default.xml:0:0-0:0)) to detect frontal faces.
*   **Distance Estimation:** A spatial depth focal approximation is used. It calculates the ratio of the bounding box width to the frame width to estimate the distance of the person in meters.
*   **Temporal Tracking (IOU):** To prevent flickering and track individuals across frames, the system uses **Intersection Over Union (IOU)**. It matches new bounding boxes to existing live tracks (requiring an IOU > 0.4). Tracks are persisted and "garbage collected" if a face is lost for more than 1 second.
*   **Audio Throttling Engine:** To prevent audio spam, speech alerts ("Person detected X meters ahead") are gated. An alert is only triggered if **2 seconds** have passed since the last announcement, OR if the person's distance changes significantly (by more than 1.0 meter).

### 2. Text Presence Scanner
Instead of using heavy OCR (Optical Character Recognition) models, this pipeline uses morphological operations to quickly scan environment frames for the *visual presence* of text blocks.

*   **Preprocessing & Edge Detection:** The frame undergoes Grayscale conversion, Gaussian Blurring (5x5), and Adaptive Thresholding to isolate sharp transitions. `Canny` edge detection (50, 150) is then applied.
*   **Morphological Dilation:** A rectangular kernel (15x3) dilates the edges, connecting closely spaced character edges into solid, detectable "text blocks."
*   **Contour & Confidence Filtering:** The system extracts external contours and filters them based on physical properties:
    *   **Size & Aspect Ratio:** Discards blocks that are too small or fail typical text aspect ratios (width must be > height, bounded between 1.0x and 20.0x).
    *   **Edge Density Score:** Calculates the density of edges within the bounding box. Text blocks typically fall within a specific edge density range. Blocks must achieve a **≥ 60% confidence score**.
*   **Multi-Frame Confirmation:** To eliminate false positives, the system requires text to be validated across **3 consecutive frames** before confirming its presence.
*   **Audio Throttling Engine:** Once triggered, the "Text ahead" audio alert is subjected to a strict **5-second cooldown** reset to prevent repetitive auditory spam.
