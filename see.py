from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import cv2
from data.dataset import *
from models.embedding import *

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--data_path', type=str, default='data/images',
        help='data path to images')
    parser.add_argument('--detector_model_path', default='models/res10_300x300_ssd_iter_140000.caffemodel', type=str, 
        help="path to OpenCV's deep learning face detector")
    parser.add_argument('--detector_proto_path', default='models/deploy.prototxt', type=str, 
        help='path to caffe prototxt path')
    parser.add_argument('--embedding_model', default='models/openface_nn4.small2.v1.t7', type=str,
	    help="path to OpenCV's deep learning face embedding model")
    parser.add_argument('--face_landmark', default='models/shape_predictor_68_face_landmarks.dat',
        type=str, help="path to dlbib Face Landmark")
    parser.add_argument('--save_model', default="models/recognizer.pickle", type=str, help="path to save trained model")
    parser.add_argument('--save_indexer', default="models/indexer.pickle", type=str, help="path to save indexer")
    parser.add_argument("--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Test Batch size')
    parser.add_argument('--dlib_emb', default=False, action='store_true', help='Run prebuilt dlib embeddings or not')
    parser.add_argument('--train_split', default=.8, type=float, help="Train-test split")

    args = parser.parse_args()
    return args

def main_handler(args):
    
    train_data, test_data, student_indexer = create_dataset(args.data_path, args, cutoff=args.train_split)
    detector = cv2.dnn.readNetFromCaffe(args.detector_proto_path, args.detector_model_path)
    embedding_model = cv2.dnn.readNetFromTorch(args.embedding_model)

    # Model with face alignment
    #model = Embedding(detector, embedding_model, args.confidence, args.face_landmark)

    if args.dlib_emb:
        # Model with dlib embeddings
        model = Embedding(detector, embedding_model, args.confidence, prebuilt=True)
    else:
        # Model without face alignment
        model = Embedding(detector,embedding_model, args.confidence)

    print("Training")
    trained_model = train_network(train_data, model, student_indexer)

    print("Saving model")
    # write the actual face recognition model to disk
    f = open(args.save_model, "wb")
    f.write(pickle.dumps(trained_model.recognizer))
    f.close()

    f = open(args.save_indexer, "wb")
    f.write(pickle.dumps(trained_model.indexer))
    f.close()

    print("Evaluating")
    #trained_model.evaluate(test_data, show_bounding=False)
    trained_model.evaluate(test_data)

    


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    main_handler(args)