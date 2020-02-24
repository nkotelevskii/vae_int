import os
import warnings
import numpy as np
from data import Dataset
from models import Decoder, Encoder
from tqdm import tqdm
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import NeuralAutoregressive

import pdb

warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn


def get_prior(args, inf_samples, prior_flow):
    if args.nf_prior:
        # Note, that here I am using T^+1 as T^-1
        log_jac_flow = 0.
        prev_v = inf_samples
        for flow_num in range(args.num_nafs):
            u = prior_flow[flow_num](prev_v)
            log_jac_flow += prior_flow[flow_num].log_abs_det_jacobian(prev_v, u)
            prev_v = u
        prior = -1. / 2 * torch.sum(u * u, 1) + log_jac_flow
    else:
        prior = -1. / 2 * torch.sum(inf_samples * inf_samples, 1)
    return prior

def get_likelihood(x_logits, x_true):
    p_x_given_z = torch.distributions.Bernoulli(logits=x_logits)
    log_likelihood = torch.sum(p_x_given_z.log_prob(x_true), [1, 2, 3])
    return log_likelihood

def get_variational(sum_log_sigma, sampled_noise):
    variational_distr = -sum_log_sigma - 0.5 * torch.sum(sampled_noise * sampled_noise, 1)
    return variational_distr

def compute_objective(args, x_logits, x_true, sampled_noise, inf_samples, sum_log_sigma, prior_flow):
    log_likelihood = get_likelihood(x_logits, x_true)
    prior = get_prior(args=args, inf_samples=inf_samples, prior_flow=prior_flow)
    variational_distr = get_variational(sum_log_sigma=sum_log_sigma, sampled_noise=sampled_noise)
    # pdb.set_trace()
    elbo = torch.mean(log_likelihood + prior - variational_distr)
    return elbo

def train_vae(args):
    best_val_elbo = -float("inf")
    prior_params = list([])
    prior_flow = None
    if args.nf_prior:
        naf = []
        for i in range(args.num_nafs):
            one_arn = AutoRegressiveNN(args.z_dim, [2 * args.z_dim], param_dims=[2 * args.z_dim] * 3).to(args.device)
            one_naf = NeuralAutoregressive(one_arn, hidden_units=256)
            naf.append(one_naf)
        prior_flow = nn.ModuleList(naf)
        prior_params = list(prior_flow.parameters())

    encoder = Encoder(args).to(args.device)
    decoder = Decoder(args).to(args.device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + prior_params
    optimizer = torch.optim.Adam(params=params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    current_tolerance = 0
    data = Dataset(args)
    # with torch.autograd.detect_anomaly():
    for ep in tqdm(range(args.num_epoches)):
        # training cycle
        for batch_num, batch_train in enumerate(data.next_train_batch()):
            batch_train_repeated = torch.repeat_interleave(batch_train, args.n_samples, dim=0)
            mu, sigma = encoder(batch_train_repeated)
            sum_log_sigma = torch.sum(torch.log(sigma), 1)
            eps = args.std_normal.sample(mu.shape)
            z = mu + sigma * eps
            logits = decoder(z)
            elbo = compute_objective(args=args, x_logits=logits, x_true=batch_train_repeated, sampled_noise=eps,
                                     inf_samples=z, sum_log_sigma=sum_log_sigma, prior_flow=prior_flow)
            (-elbo).backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation
        with torch.no_grad():
            val_elbo = validate_vae(args=args, encoder=encoder, decoder=decoder, dataset=data, prior_flow=prior_flow)
            if val_elbo > best_val_elbo:
                current_tolerance = 0
                best_val_elbo = val_elbo
                if not os.path.exists('./models/{}/'.format(args.data)):
                    os.makedirs('./models/{}/'.format(args.data))
                torch.save(encoder,
                    './models/{}/best_encoder_data_{}_skips_{}_prior_{}_numnafs_{}_samples_{}_zdim_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs, args.n_samples, args.z_dim))
                torch.save(decoder,
                    './models/{}/best_decoder_data_{}_skips_{}_prior_{}_numnafs_{}_samples_{}_zdim_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs, args.n_samples, args.z_dim))
                if args.nf_prior:
                    torch.save(prior_flow,
                        './models/{}/best_prior_data_{}_skips_{}_prior_{}_numnafs_{}_samples_{}_zdim_{}.pt'.format(args.data,
                                                args.data, args.use_skips, args.nf_prior, args.num_nafs, args.n_samples, args.z_dim))
            else:
                current_tolerance += 1
                if current_tolerance >= args.early_stopping_tolerance:
                    print("Early stopping on epoch {} (effectively trained for {} epoches)".format(ep,
                                                      ep - args.early_stopping_tolerance))
                    break
            print('Current epoch: {}'.format(ep), '\t', 'Current validation ELBO: {}'.format(val_elbo),
                  '\t', 'Best validation ELBO: {}'.format(best_val_elbo))
        # scheduler step
        scheduler.step()

    # return best models:
    encoder = torch.load('./models/{}/best_encoder_data_{}_skips_{}_prior_{}_numnafs_{}_samples_{}_zdim_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs, args.n_samples, args.z_dim))
    decoder = torch.load('./models/{}/best_decoder_data_{}_skips_{}_prior_{}_numnafs_{}_samples_{}_zdim_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_nafs, args.n_samples, args.z_dim))
    if args.nf_prior:
        prior_flow = torch.load('./models/{}/best_prior_data_{}_skips_{}_prior_{}_numnafs_{}_samples_{}_zdim_{}.pt'.format(args.data,
                                                args.data, args.use_skips, args.nf_prior, args.num_nafs, args.n_samples, args.z_dim))
    return encoder, decoder, prior_flow, data


def validate_vae(args, encoder, decoder, dataset, prior_flow):
    elbo_list = []
    for batch_num, batch_val in enumerate(dataset.next_val_batch()):
        mu, sigma = encoder(batch_val)
        sum_log_sigma = torch.sum(torch.log(sigma), 1)
        eps = args.std_normal.sample(mu.shape)
        z = mu + sigma * eps
        logits = decoder(z)
        elbo = compute_objective(args=args, x_logits=logits, x_true=batch_val, sampled_noise=eps,
                                     inf_samples=z, sum_log_sigma=sum_log_sigma, prior_flow=prior_flow)
        elbo_list.append(elbo.detach().mean().item())
    mean_val_elbo = torch.mean(torch.tensor(elbo_list, device=args.device,
                                            dtype=args.torchType)).cpu().detach().numpy()
    return mean_val_elbo
