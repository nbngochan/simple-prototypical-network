import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from read_image import read_images


def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support+n_query for n_way classes
    Args:
        n_way (int): number of classes each episode
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): array of images
        datay (np.array): array of labels
    Returns:
        (dict) of:
            (torch.Tensor): sample of images with size (n_way, n_support+n_query, dim)
            (int): n_way
            (int): n_support
            (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support+n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    import pdb; pdb.set_trace()
    sample = sample.permute(0, 1, 4, 2, 3)  # change dimension from (K, N, H, W, C) to (K, N, C, H, W)
    
    return {'images': sample,
            'n_way': n_way,
            'n_support': n_support,
            'n_query': n_query}


def display_sample(sample):
    """
    Save sample in a grid
    Args:
        sample (torch.Tensor): sample of images to display
    """
    
    # get 4D tensor (K*N, C, H, W) from 5D tensor (K, N, C, H, W)
    sample_4D = sample.view(sample.shape[0]*sample.shape[1], sample.shape[2], sample.shape[3], sample.shape[4])
    
    # make a grid of (N*K, C, H, W)
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])  # make_grid() only accepts 4D tensor with [N, C, H, W]
    plt.figure(figsize=(16,7))
    plt.imshow(out.permute(1, 2, 0))
    plt.savefig('./grid.png')
    

if __name__ == "__main__":
    datax, datay = read_images('D:/mnt/omniglot/images_background')
    sample_example = extract_sample(8, 5, 5, datax, datay)
    import pdb; pdb.set_trace()
    print('hello')