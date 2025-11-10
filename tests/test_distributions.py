import pytest
import torch

from factory import distributions


def test_lrmvn_distribution():
    """Test LRMVN distribution initialization and forward pass."""
    batch_size = 2
    hidden_dim = 64
    distribution = distributions.LRMVN_Distribution(hidden_dim=hidden_dim)

    # Create dummy input
    shoebox_representation = torch.randn(batch_size, hidden_dim)
    image_representation = torch.randn(batch_size, hidden_dim)

    # Test forward pass
    output = distribution(shoebox_representation, image_representation)
    assert isinstance(output, torch.distributions.LowRankMultivariateNormal)
    assert output.loc.shape[0] == batch_size


def test_dirichlet_profile():
    """Test Dirichlet profile initialization and forward pass."""
    batch_size = 2
    hidden_dim = 64
    profile = distributions.DirichletProfile(dmodel=hidden_dim)

    # Create dummy input
    representation = torch.randn(batch_size, hidden_dim)

    # Test forward pass
    output = profile(representation)
    assert isinstance(output, torch.distributions.Dirichlet)
