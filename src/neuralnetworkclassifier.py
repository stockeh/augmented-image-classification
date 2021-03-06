import numpy as np
import random
import torch
import time
import copy
import sys

class NeuralNetwork_Convolutional():

    def __init__(self, n_channels_in_image, image_size,
                 n_units_in_conv_layers, kernels_size_and_stride,
                 max_pooling_kernels_and_stride,
                 n_units_in_fc_hidden_layers,
                 classes, use_gpu=False, random_seed=None):

        if not isinstance(n_units_in_conv_layers, list):
            raise Exception('n_units_in_conv_layers must be a list')

        if not isinstance(n_units_in_fc_hidden_layers, list):
            raise Exception('n_units_in_fc_hidden_layers must be a list')

        if use_gpu and not torch.cuda.is_available():
            print('\nGPU is not available. Running on CPU.\n')
            use_gpu = False

        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
            np.random.seed(random_seed)              # Numpy module.
            random.seed(random_seed)                 # Python random module.
            torch.manual_seed(random_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.random_seed = random_seed
        self.n_channels_in_image = n_channels_in_image
        self.image_size = image_size
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.n_units_in_fc_hidden_layers = n_units_in_fc_hidden_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.max_pooling_kernels_and_stride = max_pooling_kernels_and_stride
        self.n_outputs = len(classes)
        self.classes = np.array(classes)
        self.use_gpu = use_gpu

        self.n_conv_layers = len(self.n_units_in_conv_layers)
        self.n_fc_hidden_layers = len(self.n_units_in_fc_hidden_layers)

        # Build the net layers
        self.nnet = torch.nn.Sequential()

        # Add convolutional layers

        n_units_previous = self.n_channels_in_image
        output_size_previous = self.image_size
        n_layers = 0
        if self.n_conv_layers > 0:

            for (n_units, kernel, pool) in zip(self.n_units_in_conv_layers, self.kernels_size_and_stride,
                                               self.max_pooling_kernels_and_stride):
                n_units_previous, output_size_previous = self._add_conv2d_tanh(n_layers,
                                        n_units_previous, output_size_previous, n_units, kernel)
                if pool:
                    output_size_previous = self._add_maxpool2d(n_layers, output_size_previous, pool)

                n_layers += 1 # for text label in layer

        self.nnet.add_module('flatten', torch.nn.Flatten())  # prepare for fc layers

        n_inputs = output_size_previous ** 2 * n_units_previous
        if self.n_fc_hidden_layers > 0:
            for n_units in self.n_units_in_fc_hidden_layers:
                n_inputs = self._add_fc_tanh(n_layers, n_inputs, n_units)
                n_layers += 1

        self.nnet.add_module(f'output_{n_layers}', torch.nn.Linear(n_inputs, self.n_outputs))

        # Member variables for standardization
        self.Xmeans = None
        self.Xstds = None

        if self.use_gpu:
            self.nnet.cuda()

        self.n_epochs = 0
        self.error_trace = []

    def _add_conv2d_tanh(self, n_layers, n_units_previous, output_size_previous,
                   n_units, kernel_size_and_stride):
        if len(kernel_size_and_stride) == 2:
            kernel_size, kernel_stride = kernel_size_and_stride
            padding = 0
        else:
            kernel_size, kernel_stride, padding = kernel_size_and_stride
        self.nnet.add_module(f'conv_{n_layers}', torch.nn.Conv2d(n_units_previous, n_units, kernel_size,
                                                                 kernel_stride, padding=padding))
        self.nnet.add_module(f'norm_{n_layers}', torch.nn.BatchNorm2d(n_units))
        self.nnet.add_module(f'output_{n_layers}', torch.nn.ReLU())
        output_size_previous = (output_size_previous + 2 * padding - kernel_size) // kernel_stride + 1
        n_units_previous = n_units
        return n_units_previous, output_size_previous

    def _add_maxpool2d(self, n_layers, output_size_previous, pool):
        if len(pool) == 2:
            kernel_size, kernel_stride = pool
            padding = 0
        else:
            kernel_size, kernel_stride, padding = pool
        self.nnet.add_module(f'pool_{n_layers}', torch.nn.MaxPool2d(kernel_size, kernel_stride,
                                                                    padding=padding))
        self.nnet.add_module(f'drop_{n_layers}', torch.nn.Dropout(p=0.2))
        output_size_previous = (output_size_previous + 2 * padding - kernel_size) // kernel_stride + 1
        return output_size_previous

    def _add_fc_tanh(self, n_layers, n_inputs, n_units):
        self.nnet.add_module(f'linear_{n_layers}', torch.nn.Linear(n_inputs, n_units))
        self.nnet.add_module(f'output_{n_layers}', torch.nn.Tanh())
        n_inputs = n_units
        return n_inputs

    def __repr__(self):
        s = f'''{type(self).__name__}(
                            n_channels_in_image={self.n_channels_in_image},
                            image_size={self.image_size},
                            n_units_in_conv_layers={self.n_units_in_conv_layers},
                            kernels_size_and_stride={self.kernels_size_and_stride},
                            max_pooling_kernels_and_stride={self.max_pooling_kernels_and_stride},
                            n_units_in_fc_hidden_layers={self.n_units_in_fc_hidden_layers},
                            classes={self.classes},
                            use_gpu={self.use_gpu})'''

        s += '\n' + str(self.nnet)
        if self.n_epochs > 0:
            s += f'\n   Network was trained for {self.n_epochs} epochs with a batch size of {self.batch_size} and took {self.training_time:.4f} seconds.'
            s += f'\n   Final objective value is {self.error_trace[-1]:.3f}'
        else:
            s += '  Network is not trained.'
        return s

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _setup_standardize(self, X, T):
        self.Xmeans = X.mean(axis=0)
        self.Xstds = X.std(axis=0)
        self.Xconstant = self.Xstds == 0
        self.XstdsFixed = copy.copy(self.Xstds)
        self.XstdsFixed[self.Xconstant] = 1

    def train(self, X, T, n_epochs, batch_size, optim='Adam', learning_rate=0.01, verbose=False):

        start_time = time.time()

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if T.ndim == 1:
            T = T.reshape((-1, 1))

        _, T = np.where(T == self.classes)  # convert to labels from 0

        self._setup_standardize(X, T)
        X = self._standardizeX(X)

        X = torch.tensor(X)
        T = torch.tensor(T.reshape(-1))
        if self.use_gpu:
            X = X.cuda()
            T = T.cuda()

        loss = torch.nn.CrossEntropyLoss()

        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.nnet.parameters(), lr=learning_rate,
                                         betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(self.nnet.parameters(), lr=learning_rate,
                                        momentum=0, dampening=0, weight_decay=0, nesterov=False)
        else:
            raise Exception('Only \'Adam\' and \'SGD\' are supported optimizers.')

        n_examples = X.shape[0]
        num_batches = n_examples // batch_size

        print_every = n_epochs // 10 if n_epochs > 9 else 1
        for epoch in range(n_epochs):
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size

                Xbatch = X[start:end, ...]
                Tbatch = T[start:end, ...]

                optimizer.zero_grad()

                Y = self.nnet(Xbatch)
                error = loss(Y, Tbatch)
                self.error_trace.append(error)

                error.backward()

                optimizer.step()

            if verbose and (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch + 1} error {error:.5f}')

        if self.use_gpu:
            X = X.cpu()
            T = T.cpu()

        self.training_time = time.time() - start_time

    def train_incremental(self, X, T, M, n_epochs, batch_size, optim='Adam', learning_rate=0.01, verbose=False):
        
        start_time = time.time()

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if T.ndim == 1:
            T = T.reshape((-1, 1))

        _, T = np.where(T == self.classes)  # convert to labels from 0

        self._setup_standardize(X, T)
        M = self._standardizeX(M)
        X = self._standardizeX(X)

        M = torch.tensor(M)
        X = torch.tensor(X)
        T = torch.tensor(T.reshape(-1))
        if self.use_gpu:
            M = M.cuda()
            X = X.cuda()
            T = T.cuda()

        loss = torch.nn.CrossEntropyLoss()

        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.nnet.parameters(), lr=learning_rate,
                                         betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(self.nnet.parameters(), lr=learning_rate,
                                        momentum=0, dampening=0, weight_decay=0, nesterov=False)
        else:
            raise Exception('Only \'Adam\' and \'SGD\' are supported optimizers.')

        n_examples = X.shape[0]
        num_batches = n_examples // batch_size

        print_every = n_epochs // 10 if n_epochs > 9 else 1
        for epoch in range(n_epochs):

            numDirty = int((epoch/n_epochs) * batch_size) #0

            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size

                Tbatch =  T[start:end, ...]

                mStart = end - numDirty
                mEnd   = end
                end    = mStart

                Xbatch = torch.cat( [ X[start:end, ...], M[mStart:mEnd, ...] ] )

                optimizer.zero_grad()

                Y = self.nnet(Xbatch)
                error = loss(Y, Tbatch)
                self.error_trace.append(error)

                error.backward()

                optimizer.step()

            if verbose and (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch + 1} error {error:.5f} clean {batch_size-numDirty} dirty {numDirty}')

        if self.use_gpu:
            M = M.cpu()
            X = X.cpu()
            T = T.cpu()

        self.training_time = time.time() - start_time


    def get_error_trace(self):
        return self.error_trace

    def cpu(self):
        if self.use_gpu:
            self.nnet.cpu()
            self.use_gpu = False

    def cuda(self):
        if torch.cuda.is_available():
            self.nnet.cuda()
            self.use_gpu = True
        else:
            print('\nGPU is not available. Running on CPU.\n')

    def _softmax(self, Y):
        mx = Y.max()
        expY = np.exp(Y - mx)
        denom = expY.sum(axis=1).reshape((-1, 1)) + sys.float_info.epsilon
        return expY / denom

    def use(self, X):
        self.nnet.eval()  # turn off gradients and other aspects of training
        with torch.no_grad():
            try:
                X = self._standardizeX(X)
                X = torch.tensor(X)
                if self.use_gpu:
                    X = X.cuda()

                Y = self.nnet(X)

                if self.use_gpu:
                    X = X.cpu()
                    Y = Y.cpu()

                Y = Y.detach().numpy()
                Yclasses = self.classes[Y.argmax(axis=1)].reshape((-1, 1))

            except:
                X = X.cpu()
                raise Exception('CUDA out of memory, pass less items for X to use.')

            return Yclasses, self._softmax(Y)

    def transfer_learn_setup(self, additional_fc_layers, freeze=True, overwrite_network=True):
        if freeze:
            for p in self.nnet.parameters():
                p.requires_grad = False

        n_units_previous = self.nnet[-1].in_features

        all_layers = list(self.nnet)[:-1]
        for n_units in additional_fc_layers:
            all_layers.extend([torch.nn.Linear(n_units_previous, n_units), torch.nn.ReLU()])
            n_units_previous = n_units
        all_layers.append(torch.nn.Linear(n_units_previous, self.n_outputs))
        new_network = torch.nn.Sequential(*all_layers)
        if self.use_gpu:
            new_network.cuda()
        if overwrite_network:
            self.nnet = new_network
        return new_network

    def _get_ctor_params(self):
        return [self.n_channels_in_image, self.image_size,
                self.n_units_in_conv_layers, self.kernels_size_and_stride,
                self.max_pooling_kernels_and_stride,
                self.n_units_in_fc_hidden_layers, self.classes.tolist(),
                self.use_gpu, self.random_seed]

    def save_network(self, save_path):
        with open(save_path, 'wb') as f:
            std_params = [self.Xmeans, self.Xstds, self.Xconstant, self.XstdsFixed]
            torch.save((self._get_ctor_params(), std_params, self.nnet.state_dict()), f)

    def load_network(load_path):
        with open(load_path, 'rb') as f:
            load_items = torch.load(f)
        nnet = NeuralNetwork_Convolutional(*load_items[0])
        nnet.Xmeans = load_items[1][0]
        nnet.Xstds = load_items[1][1]
        nnet.Xconstant = load_items[1][2]
        nnet.XstdsFixed = load_items[1][3]
        nnet.nnet.load_state_dict(load_items[2])
        return nnet
