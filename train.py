from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",

    device="cpu",
    workers=0,
    batch=4,
    imgsz=416,
    epochs=30,
    optimizer="Adam",
    lr0=0.001,
    augment=True,
    mosaic=0.7,
    mixup=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=20,
    save_period=1,
)
