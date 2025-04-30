import torch
import math
import torch.nn.functional as F

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

class Constraint(torch.nn.Module):
    def __init__(self, eps=1e-12, beta=1.0):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x):
        return F.softplus(x, beta=self.beta) + self.eps

class NormalDistributionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bijector = torch.nn.Softplus()

    def forward(self, hidden_representation):
        loc, scale = torch.unbind(hidden_representation, dim=-1)
        scale = self.bijector(scale)
        return torch.distributions.Normal(loc=loc, scale=scale)
    
class MLP(torch.nn.Module):
    def __init__(self, hidden_dim=64, input_dim=7, number_of_layers=2):
        super().__init__()
        activation = torch.nn.ReLU()
        self.hidden_dimension = hidden_dim
        self.input_dim = input_dim
        self.add_bias = True
        self._build_mlp_(number_of_layers=number_of_layers)
        
    def _build_mlp_(self, number_of_layers):
        mlp_layers = []
        for i in range(number_of_layers):
            mlp_layers.append(torch.nn.Linear(
                in_features=self.input_dim if i == 0 else self.hidden_dimension,
                out_features=self.hidden_dimension,
                bias=self.add_bias,
            ))
            mlp_layers.append(self.activation)
        self.network = torch.nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        return self.network(x)

class MLPScale(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension=64, number_of_layers=2):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.hidden_dimension = hidden_dimension
        self.input_dimension = input_dimension
        self.add_bias = True
        self._build_mlp_(number_of_layers=number_of_layers)
        
    def _build_mlp_(self, number_of_layers):
        mlp_layers = []
        for i in range(number_of_layers):
            mlp_layers.append(torch.nn.Linear(
                in_features=self.input_dimension if i == 0 else self.hidden_dimension,
                out_features=self.hidden_dimension,
                bias=self.add_bias,
            ))
            mlp_layers.append(self.activation)
        self.network = torch.nn.Sequential(*mlp_layers)

        map_to_distribution_layers = []
        map_to_distribution_layers.append(torch.nn.Linear(
            in_features=self.hidden_dimension,
            out_features=2,
            bias=self.add_bias,
        ))
        map_to_distribution_layers.append(NormalDistributionLayer())

        self.distribution = torch.nn.Sequential(*map_to_distribution_layers)

    def forward(self, x) -> torch.distributions.Normal:
        return self.distribution(self.network(x))