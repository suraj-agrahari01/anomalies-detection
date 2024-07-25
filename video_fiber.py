# from anomalib.deploy.inferencers import TorchInferencer
# from anomalib.data.utils import read_image
# import torch
# import numpy as np
# import time
# import os
# from PIL import Image
# import cv2

# # Path to the new .pt model file
# model_path = "model/model.pt"

# # Check if CUDA is available and set the device accordingly
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Initialize the inferencer with the model path and device
# inferencer = TorchInferencer(
#     path=model_path,
#     device=device,
# )

# # Function to process an image


# def process_image(image_path):
#     start_time = time.time()
#     predictions = inferencer.predict(image_path)
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")

#     segmentation_mask = np.array(predictions.segmentations)
#     segmentation_image = Image.fromarray(segmentation_mask.astype('uint8'))
#     segmentation_image.save("output/segmentation_mask.png")

# # Function to process a video


# # Function to process a video
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = None
#     frame_count = 0

#     # Create a directory to save segmented frames
#     output_dir = 'segment_output'
#     os.makedirs(output_dir, exist_ok=True)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_path = f"temp_frame_{frame_count}.png"
#         cv2.imwrite(frame_path, frame)

#         predictions = inferencer.predict(frame_path)

#         segmentation_mask = np.array(predictions.segmentations)
#         segmentation_image = segmentation_mask.astype('uint8')

#         # Save the segmentation result as an image
#         output_image_path = os.path.join(
#             output_dir, f"segmented_frame_{frame_count}.png")
#         segmentation_image_pil = Image.fromarray(segmentation_image)
#         segmentation_image_pil.save(output_image_path)

#         if out is None:
#             height, width, _ = segmentation_image.shape
#             out = cv2.VideoWriter('segmentation_video.mp4', fourcc, cap.get(
#                 cv2.CAP_PROP_FPS), (width, height), False)

#         out.write(segmentation_image)

#         cv2.imshow("nice ", cv2.resize(segmentation_image,
#                    (segmentation_image.shape[1]//2, segmentation_image.shape[0]//2)))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         os.remove(frame_path)

#         frame_count += 1

#         print(f"Processed {frame_count} frames.")

#     cap.release()
#     out.release()


# # Path to the input file (image or video)
# input_path = r"try.mp4"

# # Check if the input path is an image or a video
# if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
#     process_image(input_path)
# elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
#     process_video(input_path)
# else:
#     print("Unsupported file format.")


import cv2
import os
import torch
import numpy as np
from PIL import Image
from anomalib.deploy.inferencers import TorchInferencer

model_path = "model/model.pt"


device = "cuda" if torch.cuda.is_available() else "cpu"


inferencer = TorchInferencer(
    path=model_path,
    device=device,
)


def process_image(image_path):
    predictions = inferencer.predict(image_path)

    segmentation_mask = np.array(predictions.segmentations)
    segmentation_image = Image.fromarray(segmentation_mask.astype('uint8'))
    segmentation_image.save("output/segmentation_mask.png")


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path = 'output/segmentation_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = f"temp_frame_{frame_count}.png"
        cv2.imwrite(frame_path, frame)

        predictions = inferencer.predict(frame_path)

        segmentation_mask = np.array(predictions.segmentations)
        segmentation_image = segmentation_mask.astype('uint8')

        output_image_path = os.path.join(
            'segment_output', f"segmented_frame_{frame_count}.png")
        segmentation_image_pil = Image.fromarray(segmentation_image)
        segmentation_image_pil.save(output_image_path)

        out.write(segmentation_image)

        frame_count += 1

        print(f"Processed {frame_count} frames.")

        os.remove(frame_path)

    cap.release()
    out.release()

    print(f"Segmentation video saved to: {output_video_path}")


# Path to the input file (image or video)
input_path = r"fiber.avi"

# Check if the input path is an image or a video
if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
    process_image(input_path)
elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
    process_video(input_path)
else:
    print("Unsupported file format.")
