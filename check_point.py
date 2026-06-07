import os
from ultralytics import YOLO

def main():
    print(f"cwd: {os.getcwd()}")
    model = YOLO("yolov8n.pt")
    print("model loaded")

if __name__ == "__main__":
    main()