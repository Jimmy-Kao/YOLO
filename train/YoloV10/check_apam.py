from ultralytics import YOLO

model = YOLO('yolov10s_apam_P2.yaml')
for name, module in model.model.named_modules():
    if "APAM" in str(type(module)):
        print(name, module)
