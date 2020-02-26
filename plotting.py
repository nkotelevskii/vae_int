import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
import pdb

def plot_vardistr():
    # plots projection of latent space on 2d space
    pass

def plot_prior(args, flows):
    # pdb.set_trace()
    if not os.path.exists('./plot_data/{}/'.format(args.data)):
        os.makedirs('./plot_data/{}/'.format(args.data))
    # plots projection of prior on 2d space
    samples = args.std_normal.sample((2000, args.z_dim))
    u = samples
    if args.nf_prior:
        for i in range(args.num_nafs_prior):
            u = flows[i](u)
    u = u.cpu().detach().numpy()
    np.savetxt(fname='./plot_data/{}/prior_data_{}_skips_{}_prior_{}_numnafs_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}.txt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs_prior,
                                                args.nf_vardistr, args.num_nafs_vardistr, args.n_samples, args.z_dim), X=u)


def plot_digits():
    # plots digits
    pass