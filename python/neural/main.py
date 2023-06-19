from keras.models import load_model
import numpy as np
from PIL import Image
import cv2

image = cv2.imread("neural/test/25.png", 0)
model = load_model("neural/model.h5")


height, width = image.shape

segment_width = width // 2
segments = []
for i in range(2):
    start_x = i * segment_width
    end_x = (i + 1) * segment_width
    segment = image[:, start_x:end_x]
    segments.append(segment)
for i, segment in enumerate(segments):
    cv2.imshow(f"Segment {i+1}", segment)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predictions = []
for segment in segments:
    processed_segment = cv2.resize(segment, (28, 28))
    processed_segment = processed_segment / 255.0
    processed_segment = np.expand_dims(processed_segment, axis=-1)
    prediction = model.predict(np.array([processed_segment]))
    predictions.append(prediction)

for i, prediction in enumerate(predictions):
    print(f"Prediction for segment {i+1}: {np.argmax(prediction)}")
