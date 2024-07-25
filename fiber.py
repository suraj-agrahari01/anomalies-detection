from anomalib.deploy.inferencers import TorchInferencer
from anomalib.data.utils import read_image
import torch
import numpy as np
import time
import os
from PIL import Image


model_path = "model/model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

inferencer = TorchInferencer(
    path=model_path,
    device=device,
)

image = r"fiber_folder\bad\fiber415.jpg"

start_time = time.time()
predictions = inferencer.predict(image)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# Extract the segmentations field
segmentation_mask = np.array(predictions.segmentations)

# Convert the segmentation mask to a PIL Image and save it
image = Image.fromarray(segmentation_mask.astype('uint8'))
image.save("segmentation_mask.png")
