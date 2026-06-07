import os
from ultralytics import YOLO

def main():
    
    print(f"当前执行路径: {os.getcwd()}")

    # 2. loading the model (if not present, it will auto-download the 6MB lightweight brain)
    # this is the YOLOv8 Nano version, only about 6MB
    model = YOLO("yolov8n.pt")
    
    print("---")
    print("succeed to load the model")
    print("---")

if __name__ == "__main__":
    main()