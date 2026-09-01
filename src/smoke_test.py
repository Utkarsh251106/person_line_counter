from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
results = model.predict(source="https://ultralytics.com/images/bus.jpg", verbose=False)

result = results[0]

print("Model loaded successfully.")
print("Masks available:", result.masks is not None)

if result.masks is not None:
    print("Number of masks:", len(result.masks.data))
