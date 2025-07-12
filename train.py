from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")
results = model.train(data="dataset/dataset.yaml",
                      epochs=100,
                      imgsz=128,
                      batch=0.70,
                      device="mps",
                      exist_ok=True,
                      plots=True,
                      save=True)
