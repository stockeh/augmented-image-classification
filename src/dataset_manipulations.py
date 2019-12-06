# Data manipulations, for loading, altering, and saving batches of input images
import glob
import pickle
import numpy as np

def load_cifar_10(cifar_file_path, dataset='train'):
    images = []
    labels = []
    file_name = 'data_batch_*' if dataset == 'train' else 'test_batch'
    for file in glob.glob(cifar_file_path + file_name):
        with open(file, 'rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
            images.append(raw[b'data'])
            labels.append(raw[b'labels'])

    return (np.array(images).reshape(-1, 3, 32, 32)/255.0).astype(np.float32), np.array(labels).reshape((-1, 1))
