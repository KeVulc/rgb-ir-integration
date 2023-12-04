import ultralytics
ultralytics.checks()

from ultralytics import YOLO

  
def get_rgb_results(imagepath, model):
    model = YOLO(model)
    result = model(imagepath)  # predict on an image
    
    return result

if __name__ == '__main__':
    model = YOLO('yolov8x.pt')
    result = model('video-BzZspxAweF8AnKhWK-frame-000745-SSCRtAHcFjphNPczJ.jpg')  # predict on an image

    names = result[0].names
    boxes = result[0].boxes

    found_classes = boxes.cls
    found_pobabilites = boxes.conf
    found_xyxy = boxes.xyxy

    for index, value in enumerate(found_classes.tolist()):
      print(f'{names[int(value)]}: {found_pobabilites[int(index)]}')