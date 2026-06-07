import os
from ultralytics import YOLO

def main():
    # 1. 打印当前工作的绝对路径，确保我们在 D 盘
    print(f"当前执行路径: {os.getcwd()}")

    # 2. 尝试加载一个极小的模型 (它会自动下载到当前目录)
    # 这是 YOLOv8 的 Nano 版本，仅 6MB 左右
    model = YOLO("yolov8n.pt")
    
    print("---")
    print("✅ 恭喜！模型加载成功。")
    print("✅ 你的 Python 3.11 已经成功连接到了 Ultralytics 库。")
    print("---")

if __name__ == "__main__":
    main()