import xml.etree.ElementTree as ET
import os

# Function to convert bounding box coordinates from Pascal VOC format to YOLO format
def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Function to convert Pascal VOC XML annotations to YOLO format
def voc_to_yolo(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    yolo_annotation = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        b = (xmin, ymin, xmax, ymax)
        yolo_box = convert_coordinates((w, h), b)
        yolo_annotation.append((class_id,) + yolo_box)
    return yolo_annotation

# usage
def main():
    # Define the classes
    classes = ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector','safety-harness']

  # Add your class names here

    # Directory containing Pascal VOC annotations
    voc_dir = '/content/drive/MyDrive/MenDetection/labels'

    # Directory to save YOLO annotations
    yolo_dir = '/content/drive/MyDrive/YOLOV8'

    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    # Convert each Pascal VOC annotation file to YOLO format
    for filename in os.listdir(voc_dir):
        if filename.endswith('.xml'):
            xml_file = os.path.join(voc_dir, filename)
            yolo_annotation = voc_to_yolo(xml_file, classes)
            # Save YOLO annotation
            with open(os.path.join(yolo_dir, filename.replace('.xml', '.txt')), 'w') as f:
                for annotation in yolo_annotation:
                    line = ' '.join([str(x) for x in annotation])
                    f.write(line + '\n')

if __name__ == "__main__":
    main()
