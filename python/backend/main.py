import base64
from io import BytesIO
import cv2
from fastapi import FastAPI
from keras.models import load_model
import numpy as np
from pydantic import BaseModel
from PIL import Image

app = FastAPI()
model = load_model("backend/model.h5")


class MathProblem(BaseModel):
    problem: str
    encodedImage: str


def readb64(uri: str) -> Image:
    data = uri.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(data)))


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict")
def predict_problem_solution(problem: MathProblem) -> dict:
    prbl: str = problem.problem + "="
    image = readb64(problem.encodedImage)
    image.save("sol.png")
    image = cv2.imread("sol.png", 0)
    _, width = image.shape
    segment_width = width // 2
    segments = []
    for i in range(2):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width
        segment = image[:, start_x:end_x]
        segments.append(segment)
        predictions = []
    for segment in segments:
        processed_segment = cv2.resize(segment, (28, 28))
        processed_segment = processed_segment / 255.0
        processed_segment = np.expand_dims(processed_segment, axis=-1)
        prediction = model.predict(np.array([processed_segment]))
        predictions.append(prediction)
    res = "".join([str(np.argmax(pr)) for pr in predictions])
    prbl = "a=" + prbl + res
    print(prbl)
    loc = {}
    exec(prbl, globals(), loc)
    return {"resultCheck" : loc["a"]}
