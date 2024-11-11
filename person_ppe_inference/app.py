from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from person_ppe_inference import inference
import os
import uuid


app= Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'

if not os.path.exists(UPLOAD_FOLDER and OUTPUT_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    os.makedirs(OUTPUT_FOLDER)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

person_det = '/app/person_weights/person_det_best.pt'
ppe_det = '/app/ppe_weights/PPE_det_best.pt'


@app.route('/upload', methods = ['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file= request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400

    #Save uploaded file   
    filename = file.filename
    file_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    #Run inference
    try:
        inference(file_path, OUTPUT_FOLDER, person_det, ppe_det)
    except Exception as e:
        return jsonify({'error': f"Error during inference:{str(e)}"}), 500
    
    #Return JSON results
    json_results = {
        "person_json": os.path.join(OUTPUT_FOLDER, 'person_detection', f"{os.path.splitext(filename)[0]}.json"),
        "ppe_json": os.path.join(OUTPUT_FOLDER, 'ppe_detection', f"{os.path.splitext(filename)[0]}.json"),
    }

    return jsonify({"message": "Inference completed", "results": json_results})


@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)