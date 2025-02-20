from torch.utils.data import Subset
import random
import torch
from torchvision.utils import DataLoader,Dataset
from torchvision import transforms

transform=transforms.Compose([
    transforms.toTensor(),
    transforms.normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def score_on_dataset(dataset: Dataset, score_fn, batch_size):
    r"""
    Args:
        dataset: an instance of Dataset
        score_fn: a batch of data -> a batch of scalars
        batch_size: the batch size
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    total_score = None
    tuple_output = None
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for idx, v in enumerate(dataloader):
        v = v.to(device)
        score = score_fn(v)
        if idx == 0:
            tuple_output = isinstance(score, tuple)
            total_score = (0.,) * len(score) if tuple_output else 0.
        if tuple_output:
            total_score = tuple([a + b.sum().detach().item() for a, b in zip(total_score, score)])
        else:
            total_score += score.sum().detach().item()
    if tuple_output:
        mean_score = tuple([a / len(dataset) for a in total_score])
    else:
        mean_score = total_score / len(dataset)
    return mean_score

def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

def run_save_ms_eps(profile):
    dataset = Dataset.CIFAR10('./datasets/cifar10',train=True,transforms=transform,download=True)
    n_samples = len(dataset)
    idxes = random.sample(list(range(len(dataset))), n_samples)
    dataset = Subset(dataset, idxes)

    logging.info("save_ms_eps with {} samples".format(len(dataset)))

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    N = len(betas) - 1
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas
    eps_model = lambda x, t: diffusion.model(x, t - 1)

    ms_eps = np.zeros(N + 1, dtype=np.float32)
    for n in range(1, N + 1):
        @torch.no_grad()
        def score_fn(x_0):
            eps = torch.randn_like(x_0)
            x_n = cum_alphas[n] ** 0.5 * x_0 + cum_betas[n] ** 0.5 * eps
            eps_pred = eps_model(x_n, _rescale_timesteps(torch.tensor([n] * x_n.size(0)).type_as(x_n), N, True))
            return mos(eps_pred)

        ms_eps[n] = score_on_dataset(dataset, score_fn, profile["batch_size"])
        logging.info("[n: {}] [ms_eps[{}]: {}]".format(n, n, ms_eps[n]))

    torch.save(ms_eps, './Gamma/cifar10_gamma.pth')
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(list(range(1, N + 1)), ms_eps[1:])
    plt.close()

