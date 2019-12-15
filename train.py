from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    args = parser.parse_args()
    return args

def main_handler(args):
    pass

if __name__ == '__main__':
    args = arg_parse()
    print(args)
    main_handler(args)