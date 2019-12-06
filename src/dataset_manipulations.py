# Data manipulations, for loading, altering, and saving batches of input images
import os
import glob
import pickle
import numpy as np

def load_cifar_10(cifar_glob_path):
    """Load the original CIFAR 10 data out of the box.

    `cifar_glob_path` is a realtive path that can include globs for multiple
    files. Will typically be set to something like cifar-10-py/data_batch_*
    """
    images = []
    labels = []
    for file in glob.glob(cifar_glob_path):
        with open(file, 'rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
            images.append(raw[b'data'])
            labels.append(raw[b'labels'])

    return (np.array(images).reshape(-1, 3, 32, 32)/255.0).astype(np.float32), np.array(labels).reshape((-1, 1))

def save_cifar_10(new_cifar_path, image_data):
    """Save off CIFAR image data.

    Expects the shape of our format and converts it to the original format that
    CIFAR 10 is packaged in. Output is meant to be read in by the
    `load_cifar_10` function, but is not batched like the original CIFAR data.
    """
    try:
        os.mkdir(new_cifar_path)
    except FileExistsError:
        print('Directory {} already exists'.format(new_cifar_path))
    except:
        print('Couldn\'t create directory {}'.format(new_cifar_path))
        return
    raw = {b'data': (image_data[0].reshape(-1, 3072) * 255.0).astype(np.uint8), b'labels': image_data[1].flatten().tolist()}
    with open(os.path.relpath(new_cifar_path) + '/augmented-cifar', 'wb') as f:
        pickle.dump(raw, f)

def apply_manipulations(image_data, per_func):
    pass
