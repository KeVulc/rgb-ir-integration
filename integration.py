from ir import get_ir_results
from rgb import get_rgb_results

ir_rgb_pair = ('video-vbrSzr4vFTm5QwuGH-frame-000504-LKnve8s6tqRWKA9Bb.jpg', 'video-PGdt7pJChnKoJDt35-frame-000504-yzeMPPAQRLYPYDMYc.jpg')


ir_result = get_ir_results(ir_rgb_pair[0], 'best_ir.pt')


ir_names = ir_result[0].names
ir_boxes = ir_result[0].boxes

ir_found_classes = ir_boxes.cls
ir_found_pobabilites = ir_boxes.conf
ir_found_xyxy = ir_boxes.xyxy

rgb_result = get_rgb_results(ir_rgb_pair[1], 'yolov8x.pt')

rgb_names = rgb_result[0].names
rgb_boxes = rgb_result[0].boxes

rgb_found_classes = rgb_boxes.cls
rgb_found_pobabilites = rgb_boxes.conf
rgb_found_xyxy = rgb_boxes.xyxy


ir_idx_list = []
rgb_idx_list = []


for index, value in enumerate(ir_found_classes.tolist()):
  print(f'ir | {ir_names[int(value)]}: {ir_found_pobabilites[int(index)]}')
  if ir_names[int(value)] == 'person':
    ir_idx_list.append(int(index))
    

for index, value in enumerate(rgb_found_classes.tolist()):
  print(f'rgb| {rgb_names[int(value)]}: {rgb_found_pobabilites[int(index)]}')
  if ir_names[int(value)] == 'person':
    rgb_idx_list.append(int(index))
    
for i in range(max(len(ir_idx_list), len(rgb_idx_list))):
  print(len(ir_idx_list), len(rgb_idx_list))
  ir_idx = None
  ir_prob = None
  ir_xyxy = None
  rgb_idx = None
  rgb_prob = None
  rgb_xyxy = None
  if i < len(ir_idx_list):
    ir_idx  = ir_idx_list[i]
    ir_prob = ir_found_pobabilites[i]
    ir_xyxy = ir_found_xyxy[i]
  if i < len(rgb_idx_list):
    rgb_idx  = rgb_idx_list[i]
    rgb_prob = rgb_found_pobabilites[i]
    rgb_xyxy = rgb_found_xyxy[i]
    
  print(f'(ir, rgb, max): ({ir_prob}, {rgb_prob}, {max(ir_prob, rgb_prob)})')
