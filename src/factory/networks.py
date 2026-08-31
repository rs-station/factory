import dataclasses
import math

import torch
import torch.nn.functional as F
from tensordict.nn.distributions import Delta
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import AbsTransform


def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)

    return weight


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__(in_features, out_features, bias=bias)  # Set bias=False

    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)

    def forward(self, input):
        # Check for NaN values in input
        if torch.isnan(input).any():
            print(f"WARNING: NaN values in Linear layer input! Shape: {input.shape}")
            print("NaN count:", torch.isnan(input).sum().item())

        output = super().forward(input)

        # Check for NaN values in output
        if torch.isnan(output).any():
            print(f"WARNING: NaN values in Linear layer output! Shape: {output.shape}")
            print("NaN count:", torch.isnan(output).sum().item())
            print(
                "Weight stats - min:",
                self.weight.min().item(),
                "max:",
                self.weight.max().item(),
                "mean:",
                self.weight.mean().item(),
            )
            if self.bias is not None:
                print(
                    "Bias stats - min:",
                    self.bias.min().item(),
                    "max:",
                    self.bias.max().item(),
                    "mean:",
                    self.bias.mean().item(),
                )

        return output


class Constraint(torch.nn.Module):
    def __init__(self, eps=1e-12, beta=1.0):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x):
        return F.softplus(x, beta=self.beta) + self.eps


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim=64, input_dim=7, number_of_layers=2):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.hidden_dimension = hidden_dim
        self.input_dim = input_dim
        self.add_bias = True
        self._build_mlp_(number_of_layers=number_of_layers)

    def _build_mlp_(self, number_of_layers):
        mlp_layers = []
        for i in range(number_of_layers):
            mlp_layers.append(
                torch.nn.Linear(
                    in_features=self.input_dim if i == 0 else self.hidden_dimension,
                    out_features=self.hidden_dimension,
                    bias=self.add_bias,
                )
            )
            mlp_layers.append(self.activation)
        self.network = torch.nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.network(x)


@dataclasses.dataclass
class ScaleOutput:
    distribution: torch.distributions.Normal
    network: torch.Tensor


class BaseDistributionLayer(torch.nn.Module):

    def __init__():
        super.__init()

    def forward():
        pass


class DeltaDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 1
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc = hidden_representation  # torch.unbind(hidden_representation, dim=-1)
        print("shape los delta dist", loc.shape)
        loc = self.bijector(loc) + 1e-3
        return Delta(param=loc)


class NormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 2
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale) + 1e-3
        return torch.distributions.Normal(loc=loc, scale=scale)


class SoftplusNormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 2
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale) + 1e-3
        # self.normal = torch.distributions.Normal(loc=loc, scale=scale)
        return SoftplusNormal(loc=loc, scale=scale)


class SoftplusNormal(torch.nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = torch.distributions.Normal(loc=loc, scale=scale)
        self.bijector = torch.nn.Softplus()

    def rsample(self, sample_shape=[1]):
        return self.bijector(self.normal.rsample(sample_shape)).unsqueeze(-1)

    def log_prob(self, x):
        # x: sample (must be > 0), mu and sigma are parameters
        z = torch.log(torch.expm1(x))  # inverse softplus
        normal = self.normal
        log_pz = normal.log_prob(z)
        log_det_jacobian = -torch.nn.functional.softplus(-z)  # = -log(sigmoid(z))
        return log_pz + log_det_jacobian

    def forward(self, x, number_of_samples=1):
        return self.bijector(self.normal(x))


class TruncatedNormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 2
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale) + 1e-3
        # self.normal = torch.distributions.Normal(loc=loc, scale=scale)
        return PositiveTruncatedNormal(loc=loc, scale=scale)


class PositiveTruncatedNormal(torch.nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = torch.distributions.Normal(loc, scale)
        self.a = (0.0 - loc) / scale  # standardized lower bound = 0
        self.Z = torch.tensor(
            1.0 - self.normal.cdf(0.0), device=loc.device, dtype=loc.dtype
        )  # normalization constant

    def rsample(self, sample_shape=torch.Size()):
        u = torch.rand(
            sample_shape + self.a.shape, device=self.a.device, dtype=self.a.dtype
        )
        u = u * self.Z + self.normal.cdf(0.0)  # map [0,1] to [cdf(0), 1]
        z = self.normal.icdf(u)
        return z

    def log_prob(self, x):
        logp = self.normal.log_prob(x)
        return logp - torch.log(self.Z)


class NormalIRSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loc, scale, samples, dFdmu, dFdsig, q):
        dzdmu = -dFdmu / q
        dzdsig = -dFdsig / q
        ctx.save_for_backward(dzdmu, dzdsig)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        (
            dzdmu,
            dzdsig,
        ) = ctx.saved_tensors
        return grad_output * dzdmu, grad_output * dzdsig, None, None, None, None


class FoldedNormal(torch.distributions.Distribution):
    """
    Folded Normal distribution class

    Args:
        loc (float or Tensor): location parameter of the distribution
        scale (float or Tensor): scale parameter of the distribution (must be positive)
        validate_args (bool, optional): Whether to validate the arguments of the distribution.
        Default is None.
    """

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.nonnegative

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.distributions.utils.broadcast_all(loc, scale)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, validate_args=validate_args)
        self._irsample = NormalIRSample().apply

    def log_prob(self, value):
        """
        Compute the log-probability of the given values under the Folded Normal distribution

        Args:
            value (Tensor): The values at which to evaluate the log-probability

        Returns:
            Tensor: The log-probabilities of the given values
        """
        if self._validate_args:
            self._validate_sample(value)
        loc = self.loc
        scale = self.scale
        log_prob = torch.logaddexp(
            torch.distributions.Normal(loc, scale).log_prob(value),
            torch.distributions.Normal(-loc, scale).log_prob(value),
        )
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        """
        Generate random samples from the Folded Normal distribution

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        samples = torch.abs(eps * self.scale + self.loc)

        return samples

    @property
    def mean(self):
        """
        Compute the mean of the Folded Normal distribution

        Returns:
            Tensor: The mean of the distribution.
        """
        loc = self.loc
        scale = self.scale
        return scale * torch.sqrt(torch.tensor(2.0) / torch.pi) * torch.exp(
            -0.5 * (loc / scale) ** 2
        ) + loc * (1 - 2 * torch.distributions.Normal(0, 1).cdf(-loc / scale))

    @property
    def variance(self):
        """
        Compute the variance of the Folded Normal distribution

        Returns:
            Tensor: The variance of the distribution
        """
        loc = self.loc
        scale = self.scale
        return loc**2 + scale**2 - self.mean**2

    def cdf(self, value):
        """
        Args:
            value (Tensor): The values at which to evaluate the CDF

        Returns:
            Tensor: The CDF values at the given values
        """
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        # return dist.Normal(loc, scale).cdf(value) - dist.Normal(-loc, scale).cdf(-value)
        return 0.5 * (
            torch.erf((value + self.loc) / (self.scale * np.sqrt(2.0)))
            + torch.erf((value - self.loc) / (self.scale * np.sqrt(2.0)))
        )

    def dcdfdmu(self, value):
        return torch.exp(
            torch.distributions.Normal(-self.loc, self.scale).log_prob(value)
        ) - torch.exp(torch.distributions.Normal(self.loc, self.scale).log_prob(value))

    def dcdfdsigma(self, value):
        A = (-(value + self.loc) / self.scale) * torch.exp(
            torch.distributions.Normal(-self.loc, self.scale).log_prob(value)
        )
        B = (-(value - self.loc) / self.scale) * torch.exp(
            torch.distributions.Normal(self.loc, self.scale).log_prob(value)
        )
        return A + B

    def pdf(self, value):
        return torch.exp(self.log_prob(value))

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate differentiable random samples from the Folded Normal distribution.
        Gradients are implemented using implicit reparameterization (https://arxiv.org/abs/1805.08498).

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        samples = self.sample(sample_shape)
        # F = self.cdf(samples)
        q = self.pdf(samples)
        dFdmu = self.dcdfdmu(samples)
        dFdsigma = self.dcdfdsigma(samples)
        samples.requires_grad_(True)
        return self._irsample(self.loc, self.scale, samples, dFdmu, dFdsigma, q)


class FoldedNormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 2
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale) + 1e-3
        return FoldedNormal(loc=loc.unsqueeze(-1), scale=scale.unsqueeze(-1))


class LogNormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 2
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale) + 1e-3
        print("lognormal loc, scale", loc.shape, scale.shape)
        dist = torch.distributions.LogNormal(loc=loc, scale=scale)
        print(
            "torch.distributions.LogNormal(loc=loc, scale=scale)", dist.rsample().shape
        )
        dist = torch.distributions.LogNormal(
            loc=loc.unsqueeze(-1), scale=scale.unsqueeze(-1)
        )
        print(
            "torch.distributions.LogNormal(loc=loc, scale=scale) unsqueeze",
            dist.rsample().shape,
        )

        return dist


class GammaDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.len_params = 2
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        concentration, rate = torch.unbind(hidden_representation, dim=-1)
        rate = self.bijector(rate) + 1e-3
        concentration = self.bijector(concentration) + 1e-3
        return torch.distributions.Gamma(
            concentration=concentration.unsqueeze(-1), rate=rate.unsqueeze(-1)
        )


class MLPScale(torch.nn.Module):
    def __init__(
        self,
        input_dimension=64,
        scale_distribution=FoldedNormalDistributionLayer,
        hidden_dimension=64,
        number_of_layers=1,
        initial_scale_guess=2 / 140,
    ):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.hidden_dimension = hidden_dimension
        self.input_dimension = input_dimension
        self.add_bias = True
        self.initial_scale_guess = initial_scale_guess
        self.scale_distribution_layer = scale_distribution()
        self._build_mlp_(number_of_layers=number_of_layers)

    def _build_mlp_(self, number_of_layers):
        mlp_layers = []
        for i in range(number_of_layers):
            mlp_layers.append(
                torch.nn.Linear(
                    in_features=(
                        self.input_dimension if i == 0 else self.hidden_dimension
                    ),
                    out_features=self.hidden_dimension,
                    bias=self.add_bias,
                )
            )
            mlp_layers.append(self.activation)
        self.network = torch.nn.Sequential(*mlp_layers)

        map_to_distribution_layers = []
        final_linear = torch.nn.Linear(
            in_features=self.hidden_dimension,
            out_features=self.scale_distribution_layer.len_params,
            bias=self.add_bias,
        )

        if self.add_bias:
            with torch.no_grad():
                for i in range(self.scale_distribution_layer.len_params):
                    if i == self.scale_distribution_layer.len_params - 1:
                        final_linear.bias[i] = torch.log(
                            torch.tensor(self.initial_scale_guess)
                        )
                    else:
                        final_linear.bias[i] = torch.tensor(0.1)

        map_to_distribution_layers.append(final_linear)
        map_to_distribution_layers.append(self.scale_distribution_layer)

        self.distribution = torch.nn.Sequential(*map_to_distribution_layers)

    def forward(self, x) -> ScaleOutput:
        h = self.network(x)
        print(
            "MLP output:",
            h.mean().item(),
            h.min().item(),
            h.max().item(),
            "any nan?",
            torch.isnan(h).any().item(),
        )

        return ScaleOutput(distribution=self.distribution(h), network=h)
