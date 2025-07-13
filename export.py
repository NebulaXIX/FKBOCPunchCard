from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx",
             dynamic=True,)
