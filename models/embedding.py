import numpy as np
import cv2
import imutils
from tqdm import tqdm
from sklearn.svm import SVC

class Embedding:
    def __init__(self, detector, embedding, confidence=.5):
        self.detector = detector
        self.embedding = embedding
        self.confidence = confidence
    
    def forward(self,image_path):
        image = cv2.imread(image_path)

        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        self.detector.setInput(image_blob)
        detections = self.detector.forward()

        if len(detections)>0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    return np.zeros(128)

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedding.setInput(faceBlob)
                vec = self.embedding.forward()
                return vec.flatten()
            
        return np.zeros(128)


class TrainedModel:
    def __init__(self, embedding_model, recognizer, indexer):
        self.embedding_model = embedding_model
        self.recognizer = recognizer
        self.indexer = indexer
    
    def evaluate(self, test_data, show_bounding=True):
        correct, total = 0,0
        
        for idx, data in tqdm(enumerate(test_data), total=len(test_data)):
            images = data['path']
            labels = data['label']
            print(labels)
            embedding = [self.embedding_model.forward(image_path) for image_path in images]
            
            if np.count_nonzero(embedding[0]) == 0:
                continue

            probs = self.recognizer.predict_proba(embedding)
            print(probs)

            for i in range(len(probs)):
                p = np.argmax(probs)
                if p == labels[i]:
                    correct += 1
                total += 1
                if show_bounding:
                    for i, img in enumerate(images):
                        name = self.indexer.get_object(p.item())
                        calculate_bounding_box(img, self.embedding_model.detector, name)           
        
        print("Correctness", str(correct) + "/" + str(total) + ": " + str(round(correct/total, 5)))

def train_network(train_data, model, indexer):
    embeddings = []
    labels = []
    for idx, data in tqdm(enumerate(train_data), total=len(train_data)):
        images = data['path']
        labels.extend(data['label'])
        embeddings.extend([model.forward(path) for path in images])
    
    # Train net
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(embeddings, labels)

    return TrainedModel(model,recognizer, indexer)

def calculate_bounding_box(image_path, detector, name):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    image_blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(image_blob)
    detections = detector.forward()

    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    text = "{}".format(name)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(image, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
