import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch
from PIL import Image

from ..interfaces.ifactory import IFactory

def create_opencv_image_from_stringio(image_stream):
    # Open the image using PIL and convert it to RGB
    pil_image = Image.open(image_stream).convert("RGB")
    # Convert the PIL image to a NumPy array
    image_np = np.array(pil_image)
    # Convert RGB to BGR format (since OpenCV uses BGR format by default)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv

class FaceEmbeddingFactory(IFactory):

    @staticmethod
    def create(name, image_file):
        app = FaceAnalysis(
            name=name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        image = create_opencv_image_from_stringio(image_file)
        faces = app.get(image)

        return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
