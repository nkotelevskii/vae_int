import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset as Dataset_pt

class Dataset():
    def __init__(self, args):
        self.device = args.device
        if args.data == 'mnist':
            data_train = datasets.MNIST(root='./data/mnist', download=True).train_data.type(args.torchType).to(self.device)
            data_test = datasets.MNIST(root='./data/mnist', download=True, train=False).test_data.type(args.torchType).to(
                self.device)
        else:
            raise ModuleNotFoundError

        self.img_h = 28
        self.img_w = 28
        self.img_c = 1

        data_train = data_train[torch.randperm(data_train.size()[0])]
        n_data = data_train.shape[0]

        if max(args.val_data_size, args.batch_size_train, args.batch_size_test, args.batch_size_val) > n_data:
            raise ValueError(
                'Batch size for training, batch size for validation, batch size for test and number of data for validation should all be smaller than total data')
        data_train /= data_train.max()
        data_test /= data_test.max()
        self.validation = data_train[:args.val_data_size].data
        self.train = data_train[args.val_data_size:].data
        self.test = data_test.data

        self.batch_size_train = args.batch_size_train

        self.train_dataloader = torch.utils.data.DataLoader(self.train,
                                                            batch_size=self.batch_size_train, shuffle=True)
        self.n_IS = args.n_IS

        self.batch_size_val = args.batch_size_val
        self.val_dataloader = torch.utils.data.DataLoader(self.validation,
                                                          batch_size=self.batch_size_val, shuffle=False)

        self.batch_size_test = args.batch_size_test
        self.test_dataloader = torch.utils.data.DataLoader(self.test,
                                                           batch_size=self.batch_size_test, shuffle=False)

    def next_train_batch(self):
        """
        Training batches will reshuffle every epoch and involve dynamic
        binarization
        """
        for batch in self.train_dataloader:
            if self.img_c == 1:
                batch = torch.distributions.Binomial(probs=batch).sample()
            batch = batch.view([self.batch_size_train, self.img_c, self.img_h, self.img_w])
            yield batch

    def next_val_batch(self):
        """
        Validation batches will be used for ELBO estimates without importance
        sampling (could change)
        """
        for batch in self.val_dataloader:
            batch = batch.view([self.batch_size_val, self.img_c, self.img_h, self.img_w])
            yield batch

    def next_test_batch(self):
        """
        Test batches are same as validation but with added binarization
        """
        for batch in self.test_dataloader:
            if self.img_c == 1:
                batch = torch.distributions.Binomial(probs=batch).sample()
                batch = batch.view([self.batch_size_test, self.img_c, self.img_h, self.img_w])
            batch = batch.repeat(self.n_IS, 1, 1, 1)
            yield batch
