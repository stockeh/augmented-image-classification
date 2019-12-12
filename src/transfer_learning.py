import neuralnetworkclassifier as nnc
import dataset_manipulations as dm
import mlutils as ml
import perturb as per
import numpy as np
import torch
import time

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')

def generate_increasing_noise_plot():
    full_start = time.time()
    print('Loading data', flush=True)
    Xtrain, Ttrain, Xtest, Ttest, _, _ = dm.load_mnist('../notebooks/mnist.pkl.gz')
    # Xtrain, Ttrain = dm.load_cifar_10('../notebooks/cifar-10-batches-py/data_batch_*')
    # Xtest, Ttest = dm.load_cifar_10('../notebooks/cifar-10-batches-py/test_batch')
    lessnoise_Xtrain = per.add_image_noise(Xtrain, variance=0.025) # 0.025
    lessnoise_Xtest = per.add_image_noise(Xtrain, variance=0.025)
    noise_Xtrain = per.add_image_noise(Xtrain, variance=0.05) # 0.05
    noise_Xtest = per.add_image_noise(Xtrain, variance=0.05)
    morenoise_Xtrain = per.add_image_noise(Xtrain, variance=0.1) # 0.1
    morenoise_Xtest = per.add_image_noise(Xtrain, variance=0.1)
    print('Done loading data', flush=True)

    # model = '../notebooks/pretrained_cifar_clean.pkl'
    model = '../notebooks/pretrained_mnist_clean.pkl'
    with open(model, 'rb') as f:
        nnet = torch.load(f)
        nnet.cuda()

    print('Testing loaded network', flush=True)
    clean_pct = ml.percent_correct(Ttest, ml.batched_use(nnet, Xtest, 100))
    print('Done testing loaded network', flush=True)

    print('Training transfer learning iterations in loop', flush=True)

    res_list = []
    for ds, name in zip([lessnoise_Xtrain, noise_Xtrain, morenoise_Xtrain], ['{:.3f}'.format(0.025), '{:.3f}'.format(0.05), '{:.3f}'.format(0.1)]):
        with open(model, 'rb') as f:
            nnet = torch.load(f)
            nnet.cuda()
        # nnet.transfer_learn_setup([256, 512])
        # nnet.train(ds, Ttrain, n_epochs=10, batch_size=200, optim='Adam', learning_rate=0.0005, verbose=True)
        nnet.transfer_learn_setup([256], freeze=False)
        nnet.train(ds, Ttrain, n_epochs=20, batch_size=200, optim='Adam', learning_rate=0.0005, verbose=True)

        res_list.append((name, per.run_increasing_noise(nnet, Xtest, Ttest, trials_per_step=25)))

    print('Generating plot', flush=True)
    # per.plot_increasing_noise(clean_pct, res_list, (0.001, 0.05), 5, '{}.pdf'.format(time.strftime('%H-%M-%S')))
    per.plot_increasing_noise(clean_pct, res_list, (0.001, 0.05), 5, 'mnist-fine-tune.pdf')

    full_end = time.time()
    print('Start time: {}'.format(time.ctime(full_start)), flush=True)
    print('End time: {}'.format(time.ctime(full_end)), flush=True)
    print('Total duration: {} seconds'.format(full_end - full_start), flush=True)

if __name__ == '__main__':
    generate_increasing_noise_plot()
