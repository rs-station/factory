"""From carless: https://github.com/rs-station/careless/blob/main/careless/utils/positional_encoding.py"""

import torch

def positional_encoding(X, L):
    """
    X: metadata (batch_size, feature_size)
    L: number of sinusoidal frequency bands (output will be (batch_size, 2L))

    The positional encoding as defined in the NeRF paper https://arxiv.org/pdf/2003.08934.pdf
      gamma(p) = (sin(2**0*pi*p), cos(2**0*pi*p), ..., sin(2**(L-1)*pi*p), cos(2**(L-1)*pi*p))
    Wherein p represents an arbitrary batched set of vectors computed by normalizing X between
    between -1 and 1
    """
    # Get min and max values along the last dimension
    min_vals = X.min(dim=-1, keepdim=True)[0]
    max_vals = X.max(dim=-1, keepdim=True)[0]
    
    # Normalize between -1 and 1
    p = 2. * (X - min_vals) / (max_vals - min_vals + 1e-8) - 1.
    
    # Create frequency bands
    L_range = torch.arange(L, dtype=X.dtype, device=X.device)
    f = torch.pi * 2**L_range
    
    # Compute positional encoding
    fp = (f[..., None, :] * p[..., :, None]).reshape(p.shape[:-1] + (-1,))
    
    return torch.cat((
        torch.cos(fp),
        torch.sin(fp),
    ), dim=-1)