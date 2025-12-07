from ultralytics import YOLO
import sys

model = YOLO(r"runs/detect/train8/weights/best.pt")


def run_detection():
    source = "dataset/test/images"

    model.predict(
        source=source,
        conf=0.40,
        save=True,
        imgsz=640
    )

    print("Detection complete. Check runs/detect/predict folder.")


if __name__ == "__main__":
    run_detection()
