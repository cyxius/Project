from ultralytics import YOLO

# 1. 加载模型（如果本地没有，它会自动下载 6MB 的轻量化大脑）
model = YOLO("yolov8n.pt")

# 2. 对雾天图片进行推理
# save=True 会自动在当前目录下生成一个 runs 文件夹，里面有画好框的结果
results = model.predict(source="test_fog.jpg", save=True, conf=0.25)

print("---")
print("实验完成！请去 'runs/detect/predict' 文件夹查看模型在雾里的表现。")