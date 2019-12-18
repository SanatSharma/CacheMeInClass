import pickle
import argparse
import cv2
import face_recognition
from utils import *
from sklearn.svm import SVC
import numpy as np
import imutils


def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--image_path', type=str, required=True,help='data path to test_image')
    parser.add_argument('--model_path', type=str, default='models/recognizer.pickle', help='Path to trained model')
    parser.add_argument('--indexer_path', type=str, default="models/indexer.pickle", help='Path to student indexer')
    
    args = parser.parse_args()
    return args

def extract_embeddings(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,model='hog')

    print(boxes)
    embeddings = face_recognition.face_encodings(rgb, boxes)
    return embeddings, boxes

def recognize(file_path, recognizer:SVC, indexer:Indexer):
    image = cv2.imread(file_path)
    image = imutils.resize(image, width=600)

    embeddings, boxes = extract_embeddings(image)

    for idx, embedding in enumerate(embeddings):
        vec = embedding.flatten()
        probs = recognizer.predict_proba(vec)[0]
        prediction = np.argmax(probs)
        proba = probs[prediction]
        name = indexer.get_object(prediction)

        # draw the bounding box of the face along with the associated
        # probability
        text = "{}:{:.2f}%".format(name, proba * 100)
        startY, endX, endY, startX= boxes[idx]
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.rectangle(image, (startX, y-20), (endX, startY), (0,0,255), cv2.FILLED)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    
    cv2.imshow("Test",image)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    args = arg_parse()

    recognizer = pickle.loads(open(args.model_path, "rb").read())
    indexer = pickle.loads(open(args.indexer_path, "rb").read())

    recognize(args.image_path, recognizer, indexer)


