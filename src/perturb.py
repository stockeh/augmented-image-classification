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
#  imshow
#  classified_diff
#  change_in_pixels_plot
#  add_image_noise
#  test_increasing_noise
######################################################################

COLORS = pl.cm.Set2(np.linspace(0, 1, 8))

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

def add_image_noise(Xset, variance=0.01):
    Xcopy = copy.copy(Xset)
    noise = np.random.normal(0, variance, Xcopy.shape)
    Xcopy += noise
    return np.clip(Xcopy, 0, 1)

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

def classified_diff(nnet, Xset, Xcopy, Tset):
    Xset_classes, _  = nnet.use(Xset)
    Xcopy_classes, _ = nnet.use(Xcopy)

    diff_index = [ i for i in range(len(Xset_classes))
                  if Xset_classes[i] == Tset[i]
                  and Xset_classes[i] != Xcopy_classes[i] ]

    return diff_index, 100 - ml.percent_correct(Xset_classes, Xcopy_classes)

def avg_model_change_in_pixels(nnet_struct, Xtrain, Ttrain, Xtest, Ttest, n_models=3,
                               end_pixel_val=5, trials_per_pixel=10, name='img.pdf'):
    """
        Test a number of models to get a better average result.

        nnet_struct = {
            'train_type'                        : 'clean',
            'train_pixel_change'                : 1,
            'n_units_in_conv_layers'            : [10],
            'kernels_size_and_stride'           : [(7, 3)],
            'max_pooling_kernels_and_stride'    : [(2, 2)],
            'n_units_in_fc_hidden_layers'       : [],
            'n_epochs'                          : 50,
            'batch_size'                        : 1500,
            'learning_rate'                     : 0.05,
            'random_seed'                       : 12
        },
        Xtrain              : ... data to train with,
        Ttrain              : ... data to train with
        Xtest               : ... data to test with,
        Ttest               : ... data to test with,
        n_models            : num models to train for each perturb type,
        end_pixel_val       : num final pixel change value,
        trials_per_pixel    : num trials for each model and each pixel,
        name                : name of file to save to disk
    """

    perturbs = ['stuck', 'dead', 'hot']

    f = FloatProgress(min=0, max=(n_models * end_pixel_val * trials_per_pixel * len(perturbs)))
    display(f)

    plt.figure(figsize=(6, 4))
    print('Percent Correct on Clean Data:')
    for i, perturb in enumerate(perturbs):

        change = []

        percent_correct = []

        for n in range(n_models):

            if ( nnet_struct['train_type'] != 'clean' ):
                updated_Xtrain = change_pixel(Xtrain, pixels_to_change=nnet_struct['train_pixel_change'], pertrub=perturb)
            else:
                updated_Xtrain = Xtrain

            nnet = nnc.NeuralNetwork_Convolutional(n_channels_in_image=updated_Xtrain.shape[1],
                                           image_size=updated_Xtrain.shape[2],
                                           n_units_in_conv_layers=nnet_struct['n_units_in_conv_layers'],
                                           kernels_size_and_stride=nnet_struct['kernels_size_and_stride'],
                                           max_pooling_kernels_and_stride=nnet_struct['max_pooling_kernels_and_stride'],
                                           n_units_in_fc_hidden_layers=nnet_struct['n_units_in_fc_hidden_layers'],
                                           classes=np.unique(Ttrain), use_gpu=True, random_seed=nnet_struct['random_seed'])

            nnet.train(updated_Xtrain, Ttrain, n_epochs=nnet_struct['n_epochs'],
                       batch_size=nnet_struct['batch_size'], optim='Adam',
                       learning_rate=nnet_struct['learning_rate'], verbose=False)

            percent_correct.append(ml.percent_correct(nnet.use(Xtest)[0], Ttest))

            for pixels in range(end_pixel_val):
                percent_diff_arr = []
                for trial in range(trials_per_pixel):
                    Xcopy = change_pixel(Xtest, pixels_to_change=pixels+1, pertrub=perturb)
                    percent_diff_arr.append(classified_diff(nnet, Xtest, Xcopy, Ttest)[1])

                    f.value += 1

                change.append(percent_diff_arr)

        print(f'{perturb}: {np.mean(np.array(percent_correct)):.3f}%', end=' ')

        change = np.mean(np.array(change).reshape((n_models, end_pixel_val, trials_per_pixel)), axis=0)

        x = np.arange(1, change.shape[0] + 1)
        y = np.mean(change, axis=1)
        yerr = np.std(change, axis=1)

        plt.errorbar(x, y, yerr=yerr, marker='.', lw=1, capsize=5, capthick=1.5,
                     markeredgecolor='k', label=f'{perturb}', color=COLORS[i])

    plt.xlabel('Number of Pixels Changed')
    plt.ylabel('Mean \% Misclassified')
    plt.legend(loc='best', fontsize='large')
    plt.grid(True); plt.tight_layout();
    plt.savefig(name, bbox_inches='tight')
    plt.show();

def change_in_pixels_plot(nnet, Xset, Tset, end_pixel_val=10, trials_per_pixel=5, name='img.pdf'):

    perturbs = ['stuck', 'dead', 'hot']

    f = FloatProgress(min=0, max=(end_pixel_val * trials_per_pixel * len(perturbs)))
    display(f)

    plt.figure(figsize=(6, 4))

    for i, perturb in enumerate(perturbs):

        change = []

        for pixels in range(end_pixel_val):
            percent_diff_arr = []
            for trial in range(trials_per_pixel):
                Xcopy = change_pixel(Xset, pixels_to_change=pixels+1, pertrub=perturb)
                percent_diff_arr.append(classified_diff(nnet, Xset, Xcopy, Tset)[1])

                f.value += 1

            change.append(percent_diff_arr)

        change = np.array(change)

        x = np.arange(1, change.shape[0] + 1)
        y = np.mean(change, axis=1)
        yerr = np.std(change, axis=1)

        plt.errorbar(x, y, yerr=yerr, marker='.', lw=1, capsize=5, capthick=1.5,
                     markeredgecolor='k', label=f'{perturb}', color=COLORS[i])

    plt.xlabel('Number of Pixels Changed')
    plt.ylabel('Mean \% Misclassified')
    plt.legend(loc='best', fontsize='large')
    plt.grid(True); plt.tight_layout();
    plt.savefig(name, bbox_inches='tight')
    plt.show();

def test_increasing_noise(nnet, Xset, Tset, var_range=(0.001, 0.05), num_steps=5, trials_per_step=5):
    change = []

    f = FloatProgress(min=0, max=(num_steps * trials_per_step))
    display(f)

    for var_step in np.linspace(var_range[0], var_range[1], num_steps):
        percent_diff_arr = []
        for trial in range(trials_per_step):
            Xcopy = add_image_noise(Xset, var_step)
            percent_diff_arr.append(classified_diff(nnet, Xset, Xcopy, Tset)[1])

            f.value += 1

        change.append(percent_diff_arr)

    change = np.array(change)

    x = np.linspace(var_range[0], var_range[1], num_steps)
    y = np.mean(change, axis=1)
    yerr = np.std(change, axis=1)
    return (x, y, yerr)
