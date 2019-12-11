import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import neuralnetworkclassifier as nnc
import numpy as np
import mlutils as ml
import copy

from IPython.display import display
from ipywidgets import FloatProgress

######################################################################
# Machine Learning Utilities.
#
#  change_pixel
#  add_image_noise
#  imshow
#  classified_diff
#  change_in_pixels_plot
#  test_increasing_noise
#  train_mnist
#  train_cifar
#  augmented_training
######################################################################

COLORS = pl.cm.Set2(np.linspace(0, 1, 8))

######################################################################

def change_pixel(Xset, pixels_to_change=1, pertrub='stuck'):
    Xcopy  = copy.copy(Xset)
    bounds = Xcopy.shape[-1]
    for i in range(len(Xcopy)):
        found_set = set()
        for rounds in range(pixels_to_change):

            X = np.random.randint(bounds)
            Y = np.random.randint(bounds)

            while (X, Y) in found_set:
                X = np.random.randint(bounds)
                Y = np.random.randint(bounds)

            found_set.add((X, Y))
            for C in range(Xcopy.shape[1]):

                if pertrub == 'stuck':
                    r = np.random.random(1)
                elif pertrub == 'dead':
                    r = 0
                elif pertrub == 'hot':
                    r = 1

                Xcopy[i:i+1, C:C+1, Y:Y+1, X:X+1] = r

    return Xcopy

######################################################################

def add_image_noise(Xset, variance=0.01):
    Xcopy = copy.copy(Xset)
    noise = np.random.normal(0, variance, Xcopy.shape)
    Xcopy += noise
    return np.clip(Xcopy, 0, 1)

######################################################################

def imshow(nnet, Xset, Xcopy, Tset, same_index, model,
           display='single', name='grid.pdf'):

    if display == 'single':
        plt.figure(figsize=(9, 4))
        num  = 14
        rows = 2
        cols = 7

    else:
        plt.figure(figsize=(5, 4))
        num  = 8
        rows = 2
        cols = 4

    n_display = same_index[:num] if len(same_index) > num else same_index

    Xcopy_classes, _ = nnet.use(Xcopy[n_display])
    Xset_classes, _  = nnet.use(Xset[n_display])

    print(Xcopy.shape, Xset.shape)

    for i, val in enumerate(n_display):
        plt.subplot(rows, cols, i + 1)

        if model == 'MNIST':
            plt.imshow(Xcopy[val, :].reshape(Xset.shape[-1], Xset.shape[-1]),
                       interpolation='nearest', cmap='binary')
        if model == 'CIFAR':
            plt.imshow(np.moveaxis(Xcopy[val,...], 0, 2), interpolation='nearest')

        plt.title('$X_i$: {0}, $M_i$: {1},\n$T_i$: {2}'.format(Xset_classes[i][0],
                                                               Xcopy_classes[i][0],
                                                               Tset[val][0]))
        plt.axis('off');

    plt.tight_layout();
    plt.savefig(name, bbox_inches='tight')
    plt.show();

######################################################################

def classified_diff(nnet, Xset, Xcopy, Tset):

    try:
        Xset_classes, _  = nnet.use(Xset)
        Xcopy_classes, _ = nnet.use(Xcopy)
    except:
        Xset_classes = ml.batched_use(nnet, Xset)
        Xcopy_classes = ml.batched_use(nnet, Xcopy)

    diff_index = [ i for i in range(len(Xset_classes))
                  if Xset_classes[i] == Tset[i]
                  and Xset_classes[i] != Xcopy_classes[i] ]

    return diff_index, 100 - ml.percent_correct(Xset_classes, Xcopy_classes)

######################################################################

def change_in_pixels_plot(nnet, Xset, Tset, end_pixel_val=10, trials_per_pixel=5, name='img.pdf'):

    perturbs = ['stuck', 'dead', 'hot']

    f = FloatProgress(min=0, max=(end_pixel_val * trials_per_pixel * len(perturbs)))
    display(f)

    plt.figure(figsize=(6, 4))

    for i, perturb in enumerate(perturbs):

        change = []

        for pixels in range(end_pixel_val):
            accuracy = []
            for trial in range(trials_per_pixel):
                Xcopy = change_pixel(Xset, pixels_to_change=pixels+1, pertrub=perturb)
                try:
                    percent = ml.percent_correct(nnet.use(Xcopy)[0], Tset)
                except:
                    percent = ml.percent_correct(ml.batched_use(nnet, Xcopy), Tset)

                accuracy.append(percent)
                f.value += 1

            change.append(accuracy)

        change = np.array(change)

        x = np.arange(1, change.shape[0] + 1)
        y = np.mean(change, axis=1)
        yerr = np.std(change, axis=1)

        plt.errorbar(x, y, yerr=yerr, marker='.', lw=1, capsize=5, capthick=1.5,
                     markeredgecolor='k', label=f'{perturb}', color=COLORS[i])

    try:
        natural_per = ml.percent_correct(nnet.use(Xset)[0], Tset)
    except:
        natural_per = ml.percent_correct(ml.batched_use(nnet, Xset), Tset)

    plt.hlines(natural_per, 1, change.shape[0], label=f'natural',
               linestyle='dashed', alpha=0.3)
    plt.xlabel('Number of Pixels Changed')
    plt.ylabel('Accuracy ( \% )')
    plt.legend(loc='best', fontsize='large')
    plt.grid(True); plt.tight_layout();
    plt.savefig(name, bbox_inches='tight')
    plt.show();

######################################################################

def test_increasing_noise(nnet, Xset, Tset, var_range=(0.001, 0.05), num_steps=5,
                          trials_per_step=5, name='img.pdf'):
    change = []

    f = FloatProgress(min=0, max=(num_steps * trials_per_step))
    display(f)

    for var_step in np.linspace(var_range[0], var_range[1], num_steps):
        accuracy = []
        for trial in range(trials_per_step):
            Xcopy = add_image_noise(Xset, var_step)
            try:
                percent = ml.percent_correct(nnet.use(Xcopy)[0], Tset)
            except:
                percent = ml.percent_correct(ml.batched_use(nnet, Xcopy), Tset)

            accuracy.append(percent)
            f.value += 1

        change.append(accuracy)

    change = np.array(change)

    x = np.linspace(var_range[0], var_range[1], num_steps)
    y = np.mean(change, axis=1)
    yerr = np.std(change, axis=1)

    plt.errorbar(x, y, yerr=yerr, marker='.', lw=1, capsize=5, capthick=1.5,
                 markeredgecolor='k', color=COLORS[0])

    try:
        natural_per = ml.percent_correct(nnet.use(Xset)[0], Tset)
    except:
        natural_per = ml.percent_correct(ml.batched_use(nnet, Xset), Tset)

    plt.hlines(natural_per, var_range[0], var_range[1], label=f'natural',
               linestyle='dashed', alpha=0.3)

    plt.xticks(np.linspace(var_range[0], var_range[1], num_steps))
    plt.xlabel('Variance of Noise')
    plt.ylabel('Accuracy ( \% )')
    plt.legend(loc='best', fontsize='large')
    plt.grid(True); plt.tight_layout();
    plt.savefig(name, bbox_inches='tight')
    plt.show();

######################################################################

def train_mnist(Xtrain, Ttrain, verbose=False):
    nnet = nnc.NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                                           image_size=Xtrain.shape[2],
                                           n_units_in_conv_layers=[10],
                                           kernels_size_and_stride=[(7, 3)],
                                           max_pooling_kernels_and_stride=[(2, 2)],
                                           n_units_in_fc_hidden_layers=[],
                                           classes=np.unique(Ttrain), use_gpu=True, random_seed=12)

    nnet.train(Xtrain, Ttrain, n_epochs=50, batch_size=1500,
               optim='Adam', learning_rate=0.05, verbose=verbose)

    return nnet

def train_cifar(Xtrain, Ttrain, verbose=False):
    nnet = nnc.NeuralNetwork_Convolutional(n_channels_in_image=Xtrain.shape[1],
                               image_size=Xtrain.shape[2],
                               n_units_in_conv_layers=[64, 64, 128, 128, 256, 256, 512, 512],
                               kernels_size_and_stride=[(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
                               max_pooling_kernels_and_stride=[(), (2, 2), (), (2, 2), (), (2, 2), (), (2, 2)],
                               n_units_in_fc_hidden_layers=[],
                               classes=np.unique(Ttrain), use_gpu=True, random_seed=12)

    nnet.train(Xtrain, Ttrain, n_epochs=20, batch_size=100,
               optim='Adam', learning_rate=0.0005, verbose=verbose)

    return nnet

######################################################################
