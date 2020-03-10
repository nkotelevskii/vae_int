import torch
import numpy as np
import bottleneck as bn
import scipy

def NDCG_binary_at_k_batch(args, x_logits, x_true, sampled_noise,
                           inf_samples, sum_log_sigma, prior_flow, sum_log_jacobian):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    X_pred = scipy.sparse.csr_matrix(torch.distributions.Bernoulli(logits=x_logits).sample().cpu().detach().numpy())
    heldout_batch = scipy.sparse.csr_matrix(x_true.cpu().detach().numpy())
    k = args.k_for_cf

    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(args, x_logits, x_true, sampled_noise,
                           inf_samples, sum_log_sigma, prior_flow, sum_log_jacobian):
    X_pred = torch.distributions.Bernoulli(logits=x_logits).sample().cpu().detach().numpy()
    heldout_batch = x_true.cpu().detach().numpy()
    k = args.k_for_cf

    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def elbo(args, x_logits, x_true, sampled_noise, inf_samples, sum_log_sigma, prior_flow, sum_log_jacobian):
    def get_prior(args, inf_samples, prior_flow):
        if args.nf_prior:
            # Note, that here I am using T^+1 as T^-1
            log_jac_flow = 0.
            prev_v = inf_samples
            for flow_num in range(args.num_flows_prior):
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

    def get_variational(sum_log_sigma, sampled_noise, sum_log_jacobian):
        variational_distr = -sum_log_sigma - 0.5 * torch.sum(sampled_noise * sampled_noise, 1) - sum_log_jacobian
        return variational_distr
    log_likelihood = get_likelihood(x_logits, x_true)
    prior = get_prior(args=args, inf_samples=inf_samples, prior_flow=prior_flow)
    variational_distr = get_variational(sum_log_sigma=sum_log_sigma,
                                        sampled_noise=sampled_noise, sum_log_jacobian=sum_log_jacobian)
    elbo = torch.mean(log_likelihood + prior - variational_distr)
    return elbo.cpu().detach().numpy()