import numpy as np
from imutils import paths
import imutils
import os

def create_dataset(data_path):
    image_paths = list(paths.list_images(data_path))
    print("Dataset size:", len(image_paths))

    for i, image_path in enumerate(image_paths):
        print(image_path.split(os.path.sep))