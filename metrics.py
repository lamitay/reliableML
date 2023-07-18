import torch
import numpy as np

def calculate_confidences(model, dataloader):
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = torch.softmax(model(images), dim=1)
            max_confidences, _ = torch.max(outputs, dim=1)
            confidences.extend(max_confidences.cpu().numpy())
    return confidences

def calculate_average_confidence(confidences):
    return sum(confidences) / len(confidences)

def calculate_difference_of_confidences(in_confidences, out_confidences):
    return calculate_average_confidence(in_confidences) - calculate_average_confidence(out_confidences)

def calculate_MM_vector(embeddings):
    # gets a numpy array of the embeddings
    return np.mean(embeddings, axis=0)

def calculate_MMD(base_embeddings, target_embeddings):
    # gets a two 1D numpy arrays and calculates the MMD between them
    return np.linalg.norm(base_embeddings - target_embeddings)

def calculate_activation_statistics(embeddings):
    # implementation copied from here: https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py
    """Calculates the statistics used by FID
    Args:
        embeddings: 2D numpy array of embeddings of one dataset
    Returns:
        mu:     mean of the embeddings calculated between samples
        sigma:  covariance matrix of the embeddings

    """
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # implementation copied from here: https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean