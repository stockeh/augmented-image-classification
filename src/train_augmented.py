import json
import numpy as np
import neuralnetworkclassifier as nnc
import dataset_manipulations as dm
import mlutils as ml
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

def main():
    res_file = './training-out.csv'
    fieldnames = ['epochs', 'batch_size', 'learning_rate', 'conv_layers',
            'conv_kernel_stride', 'max_pool_kernel_stride', 'fc_layers',
            'training_time', 'final_error', 'train_pct', 'test_pct']

    prep_results_file(res_file, fieldnames)
    Xtrain, Ttrain = dm.load_cifar_10('../notebooks/new-cifar/1var-noise-train')
    Xtest, Ttest = dm.load_cifar_10('../notebooks/new-cifar/1var-noise-test')

    epochs = 1000
    batch_size = 1500
    # batch_size = 10
    rho = 0.001
    conv_layers = [128, 64, 64]
    conv_kernels = [(6, 4), (3, 3), (2, 2)]
    pooling_kernels = [(4, 4)]
    fc_layers = []

    current_results = {'epochs': epochs, 'batch_size': batch_size,
            'learning_rate': rho, 'conv_layers': str(conv_layers),
            'conv_kernel_stride': conv_kernels, 'max_pool_kernel_stride':
            pooling_kernels, 'fc_layers': fc_layers}

    nnet = nnc.NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
            image_size=Xtrain.shape[2],
            n_units_in_conv_layers=conv_layers,
            kernels_size_and_stride=conv_kernels,
            max_pooling_kernels_and_stride=pooling_kernels,
            n_units_in_fc_hidden_layers=fc_layers,
            classes=np.unique(Ttrain), use_gpu=True)

    nnet.train(Xtrain, Ttrain, n_epochs=epochs, batch_size=batch_size, optim='Adam', learning_rate=rho, verbose=True)
    current_results['training_time'] = nnet.training_time
    current_results['final_error'] = nnet.error_trace[-1].item()

    train_percent = ml.percent_correct(Ttrain, nnet.use(Xtrain)[0])
    test_percent = ml.percent_correct(Ttest, nnet.use(Xtest)[0])

    current_results['train_pct'] = train_percent
    current_results['test_pct'] = test_percent

    save_results(res_file, current_results, fieldnames)

if __name__ == '__main__':
    main()
