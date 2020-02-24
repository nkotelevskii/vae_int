import argparse
import os
import random
import numpy as np
import torch
from training import train_vae
from utils import set_args
from plotting import plot_vardistr, plot_prior, plot_digits

torchType = torch.float32

parser = argparse.ArgumentParser(
    description='VAE intuition')

################################################### Model #######################################################
parser.add_argument('-n_samples', type=int,
                    help='How many samples to use to estimate integral in ELBO over z', default=1)
parser.add_argument('-use_skips', type=str, choices=['True', 'False'],
                    help='Whether to use skip-connections from "Avoiding Latent Variable Collapse'
                         ' with Generative Skip Models arXiv:1807.04863v2"', default='False')
parser.add_argument('-nf_prior', type=str, choices=['NAF', 'None'],
                    help='Specify, which data to use', default='None')
parser.add_argument('-num_nafs', type=int, help='How many NAF to use', default=1)
parser.add_argument('-z_dim', type=int, help='Dimensionality of hidden space', default=64)

################################################## Training ######################################################
parser.add_argument('-num_epoches', type=int, help='number of epoches (for vae)', default=2000)
parser.add_argument('-early_stopping_tolerance', type=int, help='number of epoches', default=100)
parser.add_argument('-seed', type=int, metavar='RANDOM_SEED',
                    help='Random seed. If not provided, resort to random', default=1337)
parser.add_argument('-gpu', type=int, help='If >=0 - if of device, -1 means cpu', default=-1)

################################################## Datasets ######################################################
parser.add_argument('-data', type=str, choices=['mnist', 'goodreads'],
                    help='Specify, which data to use', required=True)
parser.add_argument('-batch_size_train', type=int, help='Training batch size', default=100)
parser.add_argument('-batch_size_test', type=int, help='Test batch size', default=10)
parser.add_argument('-n_IS', type=int, help='Number of Importance samples for NLL estimation', default=1000)
parser.add_argument('-batch_size_val', type=int, help='Val batch size', default=100)

parser.add_argument('-val_data_size', type=int, help='Val data size', default=1000)

parser.add_argument('-data_c', type=int, help='How many channels we have in our data', default=1)
parser.add_argument('-data_h', type=int, help='Height of our data', default=28)
parser.add_argument('-data_w', type=int, help='Width of our data', default=28)


args = parser.parse_args()

def set_seeds(rand_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

if args.seed is None:
    args.seed = np.random.randint(1, 100000)
set_seeds(args.seed)

def main(args):
    if os.path.exists('./log.txt'): # if exists, create new
        with open("./log.txt", "w") as myfile:
            myfile.write("\n \n \n \n {}".format(args))
    else: # else, append
        with open("./log.txt", "a") as myfile:
            myfile.write("\n \n \n \n {}".format(args))
    args = set_args(args)
    best_encoder, best_decoder, best_prior, dataset = train_vae(args=args)
    with torch.no_grad():
        plot_prior(args=args, flows=best_prior)

    with open("./log.txt", "a") as myfile:
        myfile.write("!!Success!! \n \n \n \n".format(args))
    print('Success!')


if __name__ == "__main__":
    main(args)
