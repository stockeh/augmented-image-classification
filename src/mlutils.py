import numpy as np
import matplotlib.pyplot as plt

######################################################################
# Machine Learning Utilities.
#
#  percent_correct
#  batched_use
#  partition
#  confusion_matrix
#  print_confusion_matrix
#  show_convolutional_layer_output
#  show_convolutional_weight_output
######################################################################

def percent_correct(actual, predicted):
    return 100 * np.mean(actual == predicted)

######################################################################

def batched_use(nnet, Xset, size=100):
    correct = []
    for i in range(0, len(Xset), size):
        correct.extend(nnet.use(Xset[i : i+size, :])[0].flatten())
    return np.array(correct).reshape(-1, 1)

######################################################################

def partition(X,T,trainFraction,shuffle=False,classification=False):
    """Usage: Xtrain,Train,Xvalidate,Tvalidate,Xtest,Ttest = partition(X,T,(0.6,0.2,0.2),shuffle=False,classification=True)
      X is nSamples x nFeatures.
      fractions can have just two values, for partitioning into train and test only
      If classification=True, T is target class as integer. Data partitioned
        according to class proportions.
        """
    # Skip the validation step
    validateFraction = 0
    testFraction = 1 - trainFraction

    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(rowIndices)

    if not classification:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        nTrain = round(trainFraction * n)
        nValidate = round(validateFraction * n)
        nTest = round(testFraction * n)
        if nTrain + nValidate + nTest > n:
            nTest = n - nTrain - nValidate
        Xtrain = X[rowIndices[:nTrain],:]
        Ttrain = T[rowIndices[:nTrain],:]
        if nValidate > 0:
            Xvalidate = X[rowIndices[nTrain:nTrain+nValidate],:]
            Tvalidate = T[rowIndices[nTrain:nTrain:nValidate],:]
        Xtest = X[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        Ttest = T[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]

    else:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        trainIndices = []
        validateIndices = []
        testIndices = []
        for c in classes:
            # row indices for class c
            cRows = np.where(T[rowIndices,:] == c)[0]
            # collect row indices for class c for each partition
            n = len(cRows)
            nTrain = round(trainFraction * n)
            nValidate = round(validateFraction * n)
            nTest = round(testFraction * n)
            if nTrain + nValidate + nTest > n:
                nTest = n - nTrain - nValidate
            trainIndices += rowIndices[cRows[:nTrain]].tolist()
            if nValidate > 0:
                validateIndices += rowIndices[cRows[nTrain:nTrain+nValidate]].tolist()
            testIndices += rowIndices[cRows[nTrain+nValidate:nTrain+nValidate+nTest]].tolist()
        Xtrain = X[trainIndices,:]
        Ttrain = T[trainIndices,:]
        if nValidate > 0:
            Xvalidate = X[validateIndices,:]
            Tvalidate = T[validateIndices,:]
        Xtest = X[testIndices,:]
        Ttest = T[testIndices,:]
    if nValidate > 0:
        return Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
    else:
        return Xtrain,Ttrain,Xtest,Ttest

######################################################################

def confusion_matrix(actual, predicted, classes):
    nc = len(classes)
    confmat = np.zeros((nc, nc))
    for ri in range(nc):
        trues = (actual==classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        keep = trues
        predictedThisClassAboveThreshold = predictedThisClass
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
    print_confusion_matrix(confmat, classes)
    return confmat

def print_confusion_matrix(confmat, classes):
    print('   ',end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ',end='')
    print('{:s}'.format('------'*len(classes)))
    for i,t in enumerate(classes):
        print('{:2d} |'.format(t), end='')
        for i1,t1 in enumerate(classes):
            if confmat[i,i1] == 0:
                print('  0  ',end='')
            else:
                print('{:5.1f}'.format(100*confmat[i,i1]), end='')
        print()

######################################################################

def show_convolutional_layer_output(nnet, X_sample, layer):
    """Usage: show_convolutional_layer_output(nnet, X_sample, 0)
      nnet is a PyTorch convolutional network
      X_sample subset from X
      layer to show outputs of
      """
    outputs = []
    reg = nnet.nnet[layer * 2].register_forward_hook(
        lambda self, i, o: outputs.append(o))
    nnet.use(X_sample)
    reg.remove()
    output = outputs[0]

    n_units = output.shape[1]
    nplots = int(np.sqrt(n_units)) + 1
    for unit in range(n_units):
        plt.subplot(nplots, nplots, unit+1)
        plt.imshow(output[0, unit, :, :].detach().cpu(),cmap='binary')
        plt.axis('off')
    return output

def show_convolutional_weight_output(nnet, layer):
    """Usage: show_convolutional_weight_output(nnet, 0)
      nnet is a PyTorch convolutional network
      layer to show weight outputs of
      """
    W = nnet.nnet[layer*2].weight.detach()
    n_units = W.shape[0]
    nplots = int(np.sqrt(n_units)) + 1
    for unit in range(n_units):
        plt.subplot(nplots, nplots, unit + 1)
        plt.imshow(W[unit, 0, :, :].detach().cpu(), cmap='binary')
        plt.axis('off')
    return W

######################################################################
