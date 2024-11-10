from ultralytics import YOLO
import json
import cv2
import numpy as np
import os
#from google.colab.patches import cv2_imshow
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import torch
import argparse


def save_predictions(results, model, img, output_path, filename):
    annotator = Annotator(img)
    image_predictions=[]
    img_with_boxes = annotator.result()
    
    for bbox in results.boxes:
        box = bbox.xyxy[0]
        classes = bbox.cls
        confidence= float(bbox.conf)
        image_predictions.append({
        "filename": os.path.join(output_path, filename),
        "bbox": box.tolist(),
        "class": model.names[int(classes)],
        "Confidence": confidence
        })

        # Draw bounding box rectangle
        rect= cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 150, 100), 4)

        # Put text for class label
        label= f"{model.names[int(classes)]}:{confidence:.2f}"
        cv2.putText(rect, label, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    
    #Save JSON files
    json_output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.json")
    with open(f"{json_output_file}.json", "w") as f:
        json.dump(image_predictions, f, indent=4)

    #Save annotated image
    cv2.imwrite(os.path.join(output_path, filename), img_with_boxes)



def inference(input_path, output_path, person_det, ppe_det):

    person_model= YOLO(model=person_det)
    ppe_model= YOLO(model= ppe_det)
    
    #img_path = input("Enter the full path of the image (or type 'exit' to quit): ").strip() + '.jpg'
    img_path= input_path.strip()
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img2 = img.copy()
    person_results = person_model.predict(img)
    ppe_results = ppe_model.predict(img2)  # return a list of Results objects


    person_opt= output_path + 'person_detection/' 

    if not os.path.exists(person_opt):
        os.makedirs(person_opt)

    ppe_opt= output_path + 'ppe_detection/' 

    if not os.path.exists(ppe_opt):
        os.makedirs(ppe_opt)

    #Run Person Detection
    save_predictions(person_results[0], person_model, img, person_opt, filename)

    #Run PPE Detection
    save_predictions(ppe_results[0], ppe_model, img2, ppe_opt, filename)

        
def parse():
    parser= argparse.ArgumentParser(prog= 'Person_ppe_inference',
                                    description= 'Fucntion to predict person and ppe detection in the images and to save in directory' )
    parser.add_argument('input_path', nargs= "+", type= str )
    parser.add_argument('output_path',nargs= "+", type= str )
    parser.add_argument('person_det',nargs= "+", type= str )
    parser.add_argument('ppe_det',nargs= "+", type= str )
    args= parser.parse_args()
    return args
       
def main():
    args= parse()
    inference(args.input_path[0], args.output_path[0], args.person_det[0], args.ppe_det[0])

if __name__== '__main__':
    main()
 
 
 
 
 
 
 
 
 
  #cv2_imshow(img_with_boxes)
#inference('d:/PPE_detection/datasets/test/images/', 'd:/person_ppe_inference/yolov8/', 'd:/person_detection/datasets/weights/person_det_best.pt', 'd:/PPE_detection/datasets/weights/PPE_det_best.pt')

#For CLI
#change line number 53 to img_path= input_path.strip() +'.jpg'
#python inference.py 'E:/AIMonk_Labs_Assessment/person_detection/datasets/test/images/' 'E:/AIMonk_Labs_Assessment/person_ppe_inference/yolov8/' 'E:/AIMonk_Labs_Assessment/person_detection/datasets/weights/person_det_best.pt/' 'E:/AIMonk_Labs_Assessment/PPE_detection/datasets/weights/PPE_det_best.pt' 
# OR 
#python inference.py 'E:/AIMonk_Labs_Assessment/PPE_detection/datasets/test/images/' 'E:/AIMonk_Labs_Assessment/person_ppe_inference/yolov8/' 'E:/AIMonk_Labs_Assessment/person_detection/datasets/weights/person_det_best.pt/' 'E:/AIMonk_Labs_Assessment/PPE_detection/datasets/weights/PPE_det_best.pt' 