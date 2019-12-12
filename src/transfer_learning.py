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
    # Xtrain, Ttrain = dm.load_cifar_10('../notebooks/cifar-10-batches-py/data_batch_*')
    # Xtest, Ttest = dm.load_cifar_10('../notebooks/cifar-10-batches-py/test_batch')
    Xtrain, Ttrain, Xtest, Ttest, _, _ = dm.load_mnist('../notebooks/mnist.pkl.gz')
    lessnoise_Xtrain = dm.apply_manipulations(Xtrain, per_func=lambda x: per.add_image_noise(x, variance=0.025)) # 0.025
    lessnoise_Xtest = dm.apply_manipulations(Xtest, per_func=lambda x: per.add_image_noise(x, variance=0.025)) # 0.025
    noise_Xtrain = dm.apply_manipulations(Xtrain, per_func=lambda x: per.add_image_noise(x, variance=0.05)) # 0.05
    noise_Xtest = dm.apply_manipulations(Xtest, per_func=lambda x: per.add_image_noise(x, variance=0.05)) # 0.05
    morenoise_Xtrain = dm.apply_manipulations(Xtrain, per_func=lambda x: per.add_image_noise(x, variance=0.1)) # 0.1
    morenoise_Xtest = dm.apply_manipulations(Xtest, per_func=lambda x: per.add_image_noise(x, variance=0.1)) # 0.1
    print('Done loading data', flush=True)

    # model = '../notebooks/pretrained_cifar_clean.pkl'
    model = '../notebooks/pretrained_mnist_clean.pkl'
    with open(model, 'rb') as f:
        nnet = torch.load(f)
        nnet.cuda()

    print('Testing loaded network', flush=True)
    clean_pct = ml.percent_correct(Ttest, ml.batched_use(nnet, Xtest, 1000))
    print('Done testing loaded network', flush=True)

    print('Training transfer learning iterations in loop', flush=True)

    var_range = (0.001, 0.1)
    res_list = [('Clean', per.run_increasing_noise(nnet, Xtest, Ttest, var_range, trials_per_step=25))]
    for ds, name in zip([lessnoise_Xtrain, noise_Xtrain, morenoise_Xtrain], ['{:.3f}'.format(0.025), '{:.3f}'.format(0.05), '{:.3f}'.format(0.1)]):
        new_model = nnet
        # nnet.transfer_learn_setup([256, 512], freeze=True)
        # nnet.train(ds, Ttrain, n_epochs=10, batch_size=200, optim='Adam', learning_rate=0.0005, verbose=True)
        new_model.transfer_learn_setup([256], freeze=True)
        new_model.train(ds, Ttrain, n_epochs=20, batch_size=200, optim='Adam', learning_rate=0.0005, verbose=True)

        res_list.append((name, per.run_increasing_noise(new_model, Xtest, Ttest, var_range, trials_per_step=25)))
        print(res_list[-1])

    print('Generating plot', flush=True)
    per.plot_increasing_noise(clean_pct, res_list, var_range, 5, 'delme.pdf')

    full_end = time.time()
    print('Start time: {}'.format(time.ctime(full_start)), flush=True)
    print('End time: {}'.format(time.ctime(full_end)), flush=True)
    print('Total duration: {} seconds'.format(full_end - full_start), flush=True)

if __name__ == '__main__':
    generate_increasing_noise_plot()
