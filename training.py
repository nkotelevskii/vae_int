import os
import warnings
import numpy as np
from data import Dataset
from models import Decoder, Encoder, Decoder_rec, Encoder_rec
from tqdm import tqdm
from pyro.nn import AutoRegressiveNN, DenseNN
from pyro.distributions.transforms import NeuralAutoregressive, AffineAutoregressive, AffineCoupling

import pdb

warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn


def get_prior(args, inf_samples, prior_flow):
    if args.nf_prior:
        # Note, that here I am using T^+1 as T^-1
        log_jac_flow = 0.
        prev_v = inf_samples
        for flow_num in range(args.num_flows_prior)[::-1]:
            u = prior_flow[flow_num](prev_v)
            log_jac_flow += prior_flow[flow_num].log_abs_det_jacobian(prev_v, u)
            prev_v = u
        prior = -1. / 2 * torch.sum(u * u, 1) + log_jac_flow
    else:
        prior = -1. / 2 * torch.sum(inf_samples * inf_samples, 1)
    return prior

def get_likelihood(x_logits, x_true):
    p_x_given_z = torch.distributions.Bernoulli(logits=x_logits)
    if len(x_true.shape) == 4:
        log_likelihood = torch.sum(p_x_given_z.log_prob(x_true), [1, 2, 3])
    elif len(x_true.shape) == 2:
        log_likelihood = torch.sum(p_x_given_z.log_prob(x_true), 1)
    return log_likelihood

def get_variational(sum_log_sigma, sampled_noise, sum_log_jacobian, args, inf_samples, mu, sigma):
    variational_distr = -sum_log_sigma - 0.5 * torch.sum(sampled_noise * sampled_noise, 1) - sum_log_jacobian
    return variational_distr

def compute_objective(args, x_logits, x_true, sampled_noise, inf_samples, sum_log_sigma, prior_flow,
                      sum_log_jacobian, mu, sigma):
    # pdb.set_trace()
    log_likelihood = get_likelihood(x_logits, x_true)
    prior = get_prior(args=args, inf_samples=inf_samples, prior_flow=prior_flow)
    variational_distr = get_variational(sum_log_sigma=sum_log_sigma,
                                        sampled_noise=sampled_noise, sum_log_jacobian=sum_log_jacobian, args=args,
                                        inf_samples=inf_samples, mu=mu, sigma=sigma)
    if args.use_reparam:
        elbo = torch.mean(log_likelihood + prior - variational_distr)
    else:
        classic_elbo = log_likelihood + prior - variational_distr
        elbo = torch.mean(log_likelihood + prior + (-sum_log_sigma - 0.5 * torch.sum((inf_samples - mu)**2 / sigma**2, 1)) * (classic_elbo.detach() - 1.))
    # pdb.set_trace()
    return elbo

def train_vae(args):
    # pdb.set_trace()
    best_metric = -float("inf")

    prior_params = list([])
    varflow_params = list([])
    prior_flow = None
    variational_flow = None

    data = Dataset(args)
    if args.data in ['goodreads', 'big_dataset']:
        args.feature_shape = data.feature_shape
    
    if args.nf_prior:
        flows = []
        for i in range(args.num_flows_prior):
            if args.nf_prior == 'IAF':
                one_arn = AutoRegressiveNN(args.z_dim, [2 * args.z_dim]).to(args.device)
                one_flow = AffineAutoregressive(one_arn)
            elif args.nf_prior == 'RNVP':
                hypernet = DenseNN(input_dim=args.z_dim // 2, hidden_dims=[2 * args.z_dim, 2 * args.z_dim],
                        param_dims=[args.z_dim - args.z_dim // 2, args.z_dim - args.z_dim // 2]).to(args.device)
                one_flow = AffineCoupling(args.z_dim // 2, hypernet).to(args.device)
            flows.append(one_flow)
        prior_flow = nn.ModuleList(flows)
        prior_params = list(prior_flow.parameters())

    if args.data == 'mnist':
        encoder = Encoder(args).to(args.device)
    elif args.data in ['goodreads', 'big_dataset']:
        encoder = Encoder_rec(args).to(args.device)

    if args.nf_vardistr:
        flows = []
        for i in range(args.num_flows_vardistr):
            one_arn = AutoRegressiveNN(args.z_dim, [2 * args.z_dim], param_dims=[2 * args.z_dim] * 3).to(args.device)
            one_flows = NeuralAutoregressive(one_arn, hidden_units=256)
            flows.append(one_flows)
        variational_flow = nn.ModuleList(flows)
        varflow_params = list(variational_flow.parameters())

    if args.data == 'mnist':
        decoder = Decoder(args).to(args.device)
    elif args.data in ['goodreads', 'big_dataset']:
        decoder = Decoder_rec(args).to(args.device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + prior_params + varflow_params
    optimizer = torch.optim.Adam(params=params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    current_tolerance = 0
    # with torch.autograd.detect_anomaly():
    for ep in tqdm(range(args.num_epoches)):
        # training cycle
        for batch_num, batch_train in enumerate(data.next_train_batch()):
            batch_train_repeated = batch_train.repeat(*[[args.n_samples] + [1] * (len(batch_train.shape) - 1)])
            mu, sigma = encoder(batch_train_repeated)
            sum_log_sigma = torch.sum(torch.log(sigma), 1)
            sum_log_jacobian = 0.
            eps = args.std_normal.sample(mu.shape)
            z = mu + sigma * eps
            if not args.use_reparam:
                z = z.detach()
            if variational_flow:
                prev_v = z
                for flow_num in range(args.num_flows_vardistr):
                    u = variational_flow[flow_num](prev_v)
                    sum_log_jacobian += variational_flow[flow_num].log_abs_det_jacobian(prev_v, u)
                    prev_v = u
                z = u
            logits = decoder(z)
            elbo = compute_objective(args=args, x_logits=logits, x_true=batch_train_repeated, sampled_noise=eps,
                                     inf_samples=z, sum_log_sigma=sum_log_sigma, prior_flow=prior_flow,
                                     sum_log_jacobian=sum_log_jacobian, mu=mu, sigma=sigma)
            (-elbo).backward()
            optimizer.step()
            optimizer.zero_grad()
        # scheduler step
        scheduler.step()

        # validation
        with torch.no_grad():
            metric = validate_vae(args=args, encoder=encoder, decoder=decoder, dataset=data, prior_flow=prior_flow,
                                    variational_flow=variational_flow)
            if (metric != metric).sum():
                print('NAN appeared!')
                raise ValueError
            if metric > best_metric:
                current_tolerance = 0
                best_metric = metric
                if not os.path.exists('./models/{}/'.format(args.data)):
                    os.makedirs('./models/{}/'.format(args.data))
                torch.save(encoder,
                    './models/{}/best_encoder_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
                torch.save(decoder,
                    './models/{}/best_decoder_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
                if args.nf_prior:
                    torch.save(prior_flow,
                        './models/{}/best_prior_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
                if args.nf_vardistr:
                    torch.save(variational_flow,
                        './models/{}/best_varflow_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
            else:
                current_tolerance += 1
                if current_tolerance >= args.early_stopping_tolerance:
                    print("Early stopping on epoch {} (effectively trained for {} epoches)".format(ep,
                                                      ep - args.early_stopping_tolerance))
                    break
            print('Current epoch: {}'.format(ep), '\t', 'Current validation {}: {}'.format(args.metric_name, metric),
                  '\t', 'Best validation {}: {}'.format(args.metric_name, best_metric))

    # return best models:
    encoder = torch.load('./models/{}/best_encoder_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
    decoder = torch.load('./models/{}/best_decoder_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
    if args.nf_prior:
        prior_flow = torch.load('./models/{}/best_prior_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
    if args.nf_vardistr:
        variational_flow = torch.load('./models/{}/best_varflow_data_{}_skips_{}_prior_{}_numflows_{}_varflow_{}_numvarflows_{}_samples_{}_zdim_{}_usereparam_{}.pt'.format(args.data,
                                            args.data, args.use_skips, args.nf_prior, args.num_flows_prior,
                                                args.nf_vardistr, args.num_flows_vardistr, args.n_samples, args.z_dim, args.use_reparam))
    return encoder, decoder, prior_flow, variational_flow, data


def validate_vae(args, encoder, decoder, dataset, prior_flow, variational_flow):
    metric_list = []
    for batch_num, batch_val in enumerate(dataset.next_val_batch()):
        mu, sigma = encoder(batch_val)
        sum_log_sigma = torch.sum(torch.log(sigma), 1)
        sum_log_jacobian = 0.
        eps = args.std_normal.sample(mu.shape)
        z = mu + sigma * eps
        if variational_flow:
            prev_v = z
            for flow_num in range(args.num_flows_vardistr):
                u = variational_flow[flow_num](prev_v)
                sum_log_jacobian += variational_flow[flow_num].log_abs_det_jacobian(prev_v, u)
                prev_v = u
            z = u
        logits = decoder(z)
        metric = args.metric(args=args, x_logits=logits, x_true=batch_val, sampled_noise=eps,
                                     inf_samples=z, sum_log_sigma=sum_log_sigma, prior_flow=prior_flow,
                                 sum_log_jacobian=sum_log_jacobian)
        metric_list.append(metric)
    mean_val_metric = np.mean(metric_list)
    return mean_val_metric
