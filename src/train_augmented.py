import json
import time
import numpy as np
import neuralnetworkclassifier as nnc
import dataset_manipulations as dm
import mlutils as ml
import itertools
from csv import DictWriter

def prep_results_file(res_file, fieldnames):
    with open(res_file, 'w') as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def save_results(res_file, results_dict, fieldnames):
    print(json.dumps(results_dict))
    with open(res_file, 'a') as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results_dict)

def batched_use(nnet, Xset, Tset):
    correct = 0
    total = 0
    for img, t in zip(Xset, Tset):
        if nnet.use([img])[0] == t:
            correct += 1
        total += 1
    percent = (correct / total) * 100
    return percent

def main():
    full_start = time.time()
    res_file = './training-out.csv'
    fieldnames = ['epochs', 'batch_size', 'learning_rate', 'conv_layers',
            'conv_kernel_stride', 'max_pool_kernel_stride', 'fc_layers',
            'training_time', 'final_error', 'train_pct', 'test_pct']

    prep_results_file(res_file, fieldnames)

    # Can change which data is loaded here if you want to work with a clean dataset
    print('Loading data', flush=True)
    # Xtrain, Ttrain = dm.load_cifar_10('../notebooks/new-cifar/1var-noise-train')
    # Xtest, Ttest = dm.load_cifar_10('../notebooks/new-cifar/1var-noise-test')
    Xtrain, Ttrain = dm.load_cifar_10('../notebooks/cifar-10-batches-py/data_batch_*')
    Xtest, Ttest = dm.load_cifar_10('../notebooks/cifar-10-batches-py/test_batch')
    print('Done loading data', flush=True)

    l_epochs = [ 20 ]
    l_batch_size = [ 100 ]
    l_rho = [ 0.0005 ]
    l_conn_layers = [ [ ] ]
    l_conv_layers = [ [64, 64, 128, 128, 256, 256] ]

    l_conv_kernels = {
            '8' : [ [(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)] ],
            '6' : [ [(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)] ],
            '3' : [ [(6, 2), (3, 2), (2, 1)] ],
            '2' : [ [(4, 2), (2, 2)] ],
            '1' : [ [(4, 2)] ]
            }
    l_pool_kernels = {
            '8' : [ (), (2, 2), (), (2, 2), (), (2, 2), (), (2, 2)] ],
            '6' : [ (), (2, 2), (), (2, 2), (), (2, 2)] ],
            '3' : [ [(2, 2), (2, 1), ()] ],
            '2' : [ [(2, 1), (2, 1)] ],
            '1' : [ [(2, 1)] ]
            }

    n_trials = 0
    for v in l_conv_layers:
        n_trials += (len(l_conv_kernels[str(len(v))]) * len(l_pool_kernels[str(len(v))]))
    n_trials *= (len(l_epochs) * len(l_batch_size) * len(l_rho) * len(l_conn_layers))
    trial = 1

    for epochs, batch_size, rho, conv, conn in itertools.product(l_epochs, l_batch_size, l_rho,
                                                                 l_conv_layers, l_conn_layers):

        for conv_kernels, pool_kernels in itertools.product(l_conv_kernels[str(len(conv))],
                                                            l_pool_kernels[str(len(conv))]):

            print('\n###### Trying network structure {} out of {} ######'.format(trial, n_trials), flush=True)

            results = {
                    'epochs': epochs, 'batch_size': batch_size,
                    'learning_rate': rho, 'conv_layers': conv,
                    'conv_kernel_stride': conv_kernels,
                    'max_pool_kernel_stride': pool_kernels,
                    'fc_layers': conn
                    }

            nnet = nnc.NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                                   image_size=Xtrain.shape[2],
                                                   n_units_in_conv_layers=results['conv_layers'],
                                                   kernels_size_and_stride=results['conv_kernel_stride'],
                                                   max_pooling_kernels_and_stride=results['max_pool_kernel_stride'],
                                                   n_units_in_fc_hidden_layers=results['fc_layers'],
                                                   classes=np.unique(Ttrain), use_gpu=True, random_seed=12)

            nnet.train(Xtrain, Ttrain, n_epochs=results['epochs'], batch_size=results['batch_size'],
                       optim='Adam', learning_rate=results['learning_rate'], verbose=True)

            train_percent = batched_use(nnet, Xtrain, Ttrain)
            test_percent = batched_use(nnet, Xtest, Ttest)

            results['training_time'] = nnet.training_time
            results['final_error'] = nnet.error_trace[-1].item()
            results['train_pct'] = train_percent
            results['test_pct'] = test_percent

            save_results(res_file, results, fieldnames)
            trial += 1

    full_end = time.time()
    print('Start time: {}'.format(time.ctime(full_start)), flush=True)
    print('End time: {}'.format(time.ctime(full_end)), flush=True)
    print('Total duration: {} seconds'.format(full_end - full_start), flush=True)

if __name__ == '__main__':
    main()
