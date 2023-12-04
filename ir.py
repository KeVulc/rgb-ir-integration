import ultralytics
ultralytics.checks()

from ultralytics import YOLO

def get_ir_results(imagepath, model):
    model = YOLO(model)
    result = model(imagepath)  # predict on an image
    return result

if __name__ == '__main__':
    model = YOLO('best_ir.pt')
    result = model('video-4FRnNpmSmwktFJKjg-frame-000745-L6K5SC6fYjHNC8uff.jpg')  # predict on an image

    names = result[0].names
    boxes = result[0].boxes

    found_classes = boxes.cls
    found_pobabilites = boxes.conf
    found_xyxy = boxes.xyxy

    for index, value in enumerate(found_classes.tolist()):
      print(f'{names[int(value)]}: {found_pobabilites[int(index)]}')