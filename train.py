from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained model
    model = YOLO("yolo11n.pt")
    results = model.train(data="dataset/dataset.yaml",
                          epochs=500,
                          imgsz=128,
                          batch=0.9,
                          device=0,
                          exist_ok=True,
                          plots=True,
                          save_period=25)
