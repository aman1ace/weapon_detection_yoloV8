from ultralytics import YOLO

model = YOLO(r"runs/detect/train8/weights/best.pt")


def main():
    model.predict(
        source=0,
        show=True,
        conf=0.45,
        stream=False,
        imgsz=480
    )


if __name__ == "__main__":
    main()
