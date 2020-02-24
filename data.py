import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset as Dataset_pt
import pdb

class Dataset():
    def __init__(self, args):
        self.device = args.device
        if args.data == 'mnist':
            train = datasets.MNIST(root='./data/mnist', download=True)
            data_train = train.train_data.type(args.torchType).to(self.device)
            labels_train = train.train_labels.type(args.torchType).to(self.device)
            test = datasets.MNIST(root='./data/mnist', download=True, train=False)
            data_test = test.test_data.type(args.torchType).to(
                self.device)
            labels_test = test.test_labels.type(args.torchType).to(
                self.device)
        else:
            raise ModuleNotFoundError

        self.img_h = 28
        self.img_w = 28
        self.img_c = 1

        permute = torch.randperm(data_train.size()[0])
        data_train = data_train[permute]
        labels_train = labels_train[permute]
        n_data = data_train.shape[0]

        if max(args.val_data_size, args.batch_size_train, args.batch_size_test, args.batch_size_val) > n_data:
            raise ValueError(
                'Batch size for training, batch size for validation, batch size for test and number of data for validation should all be smaller than total data')
        data_train /= data_train.max()
        data_test /= data_test.max()

        self.validation = data_train[:args.val_data_size].data
        self.validation_labels = labels_train[:args.val_data_size].data

        self.train = data_train[args.val_data_size:].data
        self.train_labels = labels_train[args.val_data_size:].data

        self.n_IS = args.n_IS
        self.test = data_test.data
        self.test_labels = labels_test.data

        self.batch_size_train = args.batch_size_train

        train_data = []
        for i in range(self.train.shape[0]):
            train_data.append([self.train[i], self.train_labels[i]])
        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True)


        val_data = []
        for i in range(self.validation.shape[0]):
            val_data.append([self.validation[i], self.validation_labels[i]])
        self.batch_size_val = args.batch_size_val
        self.val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_val, shuffle=False)

        test_data = []
        for i in range(self.test.shape[0]):
            test_data.append([self.test[i], self.test_labels[i]])
        self.batch_size_test = args.batch_size_test
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, shuffle=False)

    def next_train_batch(self, return_labels=False):
        """
        Training batches will reshuffle every epoch and involve dynamic
        binarization
        """
        for train_batch in self.train_dataloader:
            batch = train_batch[0]
            labels = train_batch[1]
            if self.img_c == 1:
                batch = torch.distributions.Binomial(probs=batch).sample()
            batch = batch.view([self.batch_size_train, self.img_c, self.img_h, self.img_w])
            if return_labels:
                yield batch, labels
            else:
                yield batch

    def next_val_batch(self, return_labels=False):
        """
        Validation batches will be used for ELBO estimates without importance
        sampling (could change)
        """
        for val_batch in self.val_dataloader:
            batch = val_batch[0]
            labels = val_batch[1]
            batch = batch.view([self.batch_size_val, self.img_c, self.img_h, self.img_w])
            if return_labels:
                yield batch, labels
            else:
                yield batch

    def next_test_batch(self, return_labels=False):
        """
        Test batches are same as validation but with added binarization
        """
        for test_batch in self.test_dataloader:
            batch = test_batch[0]
            labels = test_batch[1]
            if self.img_c == 1:
                batch = torch.distributions.Binomial(probs=batch).sample()
                batch = batch.view([self.batch_size_test, self.img_c, self.img_h, self.img_w])
            batch = batch.repeat(self.n_IS, 1, 1, 1)
            if return_labels:
                labels = labels.repeat(self.n_IS, 1)
                yield batch, labels
            else:
                yield batch
