import xml.etree.ElementTree as ET
import glob
import os
import argparse

classes = ['hard-hat', 'gloves', 'mask', 'glasses', 'boots',
           'vest', 'ppe-suit', 'ear-protector', 'safety-harness']


def convert_xml_to_yolo(input_path, output_path):

    for filename in os.listdir(input_path):
        if not filename.endswith('.xml'):
            continue
        file = os.path.join(input_path, filename)
        tree = ET.parse(file)
        root = tree.getroot()
        
        bounding_boxes = []
        for size in root.findall('size'):
            size[0]= (size.find('width'))
            size[1]= (size.find('height'))        
            for object in root.findall('object'):
                name = object.find('name').text
                if name == 'person':
                    continue
                bndbox = object.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Convert the bounding box to YOLO format.
                dw = 1/int(size[0].text)
                dh = 1/int(size[1].text)
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                x = x_center*dw
                w = width*dw
                y = y_center*dh
                h = height*dh

                bounding_boxes.append([classes.index(name), x, y, w, h])
                #print(bounding_boxes)
    
        out_dir= output_path + '/yolo_PPE/' 

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        output_file= out_dir + os.path.splitext(filename)[0] + '.txt'
        with open(output_file,'w') as f:
            for bounding_box in bounding_boxes:
                f.write(' '.join(str(x) for x in bounding_box) + '\n')

def parse():
    parser= argparse.ArgumentParser(prog= 'pascalVOC_to_yolo',
                                    description= 'Fucntion to convert image labels with PascalVOC XML file format to Yolo format' )
    parser.add_argument('input_path', nargs= "+", type= str )
    parser.add_argument('output_path',nargs= "+", type= str )
    args= parser.parse_args()
    return args
       
def main():
    args= parse()
    convert_xml_to_yolo(args.input_path[0], args.output_path[0])

if __name__== '__main__':
    main()


#convert_xml_to_yolo(xml_file, output)
#xml_file = r'D:\Person_PPE_detection\datasets\labels'
#output= r'D:\Person_PPE_detection\datasets'
