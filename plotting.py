import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
import pdb

def plot_vardistr(args, encoder, flows, dataset):
    # plots projection of latent space on 2d space
    points = torch.tensor([])
    labels = torch.tensor([])
    for test_data, test_labels in dataset.next_test_batch(return_labels=True):
        z, _ = encoder(test_data)
        if flows:
            for flow in flows:
                z = flow(z)
        z = z.cpu().detach()
        test_labels = test_labels.cpu().detach()
        points = torch.cat([points, z])
        labels = torch.cat([labels, test_labels])
    points = points.numpy()
    labels = labels.numpy()
    np.savetxt(fname='./plot_data/{}/vardistr_points_data_{}_skips_{}_prior_{}_numnafs_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}.txt'.format(args.data,
                                                args.data, args.use_skips, args.nf_prior, args.num_nafs_prior,
                                                    args.nf_vardistr, args.num_nafs_vardistr, args.n_samples, args.z_dim), X=points)
    np.savetxt(fname='./plot_data/{}/vardistr_labels_data_{}_skips_{}_prior_{}_numnafs_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}.txt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs_prior,
                                                args.nf_vardistr, args.num_nafs_vardistr, args.n_samples, args.z_dim), X=labels)
        

def plot_prior(args, flows):
    # pdb.set_trace()
    if not os.path.exists('./plot_data/{}/'.format(args.data)):
        os.makedirs('./plot_data/{}/'.format(args.data))
    # plots projection of prior on 2d space
    samples = args.std_normal.sample((10000, args.z_dim))
    u = samples
    if args.nf_prior:
        for i in range(args.num_nafs_prior):
            u = flows[i](u)
    u = u.cpu().detach().numpy()
    np.savetxt(fname='./plot_data/{}/prior_data_{}_skips_{}_prior_{}_numnafs_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}.txt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs_prior,
                                                args.nf_vardistr, args.num_nafs_vardistr, args.n_samples, args.z_dim), X=u)

def plot_digit_samples(samples):
    """
    Plot samples from the generative network in a grid
    """

    grid_h = 8
    grid_w = 8
    data_h = 28
    data_w = 28
    data_c = 1

    # Turn the samples into one large image
    tiled_img = np.zeros((data_h * grid_h, data_w * grid_w))

    for idx, image in enumerate(samples):
        i = idx % grid_w
        j = idx // grid_w

        top = j * data_h
        bottom = (j + 1) * data_h
        left = i * data_w
        right = (i + 1) * data_w
        tiled_img[top:bottom, left:right] = image

    # Save the new image
    plt.close()
    plt.axis('off')

    plt.imshow(tiled_img, cmap='gray')
    plt.tight_layout()
    plt.show()


def get_samples(gen_network, random_code):
    samples = nn.Sigmoid()(gen_network(random_code)).view(random_code.shape[0], 28, 28)
    out = samples.cpu().detach().numpy()
    return out