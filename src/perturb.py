import matplotlib.pyplot as plt
import matplotlib.pylab as pl
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
        for rounds in range(pixels_to_change):
            X = np.random.randint(bounds)
            Y = np.random.randint(bounds)

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
