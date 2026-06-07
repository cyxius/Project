from ultralytics import YOLO

model = YOLO("yolov8n.pt")


results = model.predict(source="test_fog.jpg", save=True, conf=0.25)

print("---")
print("Complete. Check file 'runs/detect/predict' ")