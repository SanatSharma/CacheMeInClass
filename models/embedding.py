import numpy as np
import cv2
import os

class Embedding:
    def __init__(self, detector, embedding):
        self.detector = detector
        self.embedding = embedding
    
    def forward(image_path):
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        #self.detector.setInput(image_blob)
        detections = self.detector.forward(image_blob)

        if len(detections)>0:
            print(detections)



def train_network(train_data, model, indexer):
    pass

