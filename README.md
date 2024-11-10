# Person and PPE (Personal Protective Equipment) Detection Using YOLOv8

This project leverages the YOLOv8 object detection model to detect persons and their associated PPE in images. The system is designed to assist in ensuring safety compliance in industrial and construction environments by identifying the presence or absence of critical safety gear.

## Project Overview

### Key Features:
- **Custom Object Detection Models:** Trained YOLOv8 models for detecting persons and various PPE items such as helmets, vests, and gloves.
- **Flask-based API:** Provides a web service for easy image upload and inference.
- **Docker Containerization:** Ensures portability and consistent runtime environments for the application.
- **JSON Output:** Detection results are saved in JSON format for further analysis.

---

## 1. Model Training

### Dataset Preparation:
1. **Dataset:** 
   - Custom dataset consisting of 416 images labeled in PascalVOC format.
2. **Conversion to YOLO Format:**
   - The labels were converted into YOLO format using `PASCALVOC_to_yolo.py`.
   - For person detection, only the `person` class was included.
   - For PPE detection, the following classes were retained:
     - Helmet
     - Vest
     - Gloves
   - Imbalanced classes such as `glasses`, `ear-protector`, and `safety-harness` were omitted to improve training.

3. **Data Split:**
   - Training: 70%
   - Validation: 20%
   - Testing: 10%

### Model Training:
- **Person Detection Model:**
  - Trained on YOLOv8 using 70 epochs.
  - Best weights saved as `person_det_best.pt`.
  
- **PPE Detection Model:**
  - Trained separately for 80 epochs.
  - Best weights saved as `PPE_det_best.pt`.

### Model Evaluation:
- **Person Detection Model:**
  - mAP@0.5: 85.7%
  - mAP@0.5:0.95: 58.7%
  - F1 Score: 0.802
  
- **PPE Detection Model:**
  - mAP@0.5: 49.4%
  - mAP@0.5:0.95: 31.6%
  - F1 Score: 0.50

---

## 2. Inference (AI Backend)

The trained models are used for inference on new images using `person_ppe_inference.py`. Detection results for persons and PPE are saved as JSON files in structured directories (`person_detection` and `ppe_detection`).

---

## 3. Flask API Backend

A Flask application serves as the UI backend for image uploads and inference.

### Key Endpoints:
- **`/upload` (POST):** Accepts an image file and returns JSON results with detection outputs.
  - **Usage with `curl`:**
    ```bash
    curl -X POST -F "file=@<path_to_image>" http://127.0.0.1:5000/upload
    ```
  - **Sample JSON Output:**
    ```json
    {
      "message": "Inference completed",
      "results": {
        "person_json": "outputs/person_detection/<filename>.json",
        "ppe_json": "outputs/ppe_detection/<filename>.json"
      }
    }
    ```

- **`/download/<path>` (GET):** Allows downloading of specific result files.

### Running the Flask App Locally:
```bash
python app.py
