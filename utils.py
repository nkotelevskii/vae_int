import torch
from metrics import elbo, NDCG_binary_at_k_batch, Recall_at_k_batch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_args(args):
    print('\nParameters: \n', args, '\n')
    args = dotdict(vars(args))

    args.device = 'cpu' if args.gpu == -1 else 'cuda:{}'.format(args.gpu)
    args.torchType = torch.float32
    args['use_skips'] = True if args['use_skips'] == 'True' else False
    args['nf_prior'] = None if args['nf_prior'] == 'None' else args['nf_prior']
    args['nf_vardistr'] = None if args['nf_vardistr'] == 'None' else args['nf_vardistr']

    args.std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=args.device, dtype=args.torchType),
                               scale=torch.tensor(1., device=args.device, dtype=args.torchType))
    args.metric_name = args.metric
    if args.metric == 'elbo':
        args.metric = elbo
    elif args.metric == 'ndcg':
        args.metric = NDCG_binary_at_k_batch
    elif args.metric == 'recall':
        args.metric = Recall_at_k_batch
    return args