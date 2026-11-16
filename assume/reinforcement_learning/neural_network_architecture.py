# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch as th
from torch import nn
from torch.nn import functional as F


class CriticTD3(nn.Module):
    """Initialize parameters and build model.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of each state
        act_dim (int): Dimension of each action
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int,
        use_popart: bool = False,
        popart_beta: float = 3e-4,
    ):
        super().__init__()

        self.obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        self.act_dim = act_dim * n_agents
        self.use_popart = use_popart

        # Select proper architecture based on `n_agents`
        if n_agents <= 20:
            hidden_sizes = [256, 128]  # Shallow network for small `n_agents`
        elif n_agents <= 50:
            hidden_sizes = [512, 256, 128]  # Medium network
        else:
            hidden_sizes = [1024, 512, 256, 128]  # Deeper network for large `n_agents`

        # First Q-network (Q1)
        self.q1_layers = self._build_q_network(hidden_sizes, float_type)

        # Second Q-network (Q2) for double Q-learning
        self.q2_layers = self._build_q_network(hidden_sizes, float_type)

        # PopArt output layers (replace final linear layers)
        self.q1_popart = None
        self.q2_popart = None
        if use_popart:
            from assume.reinforcement_learning.popart import PopArtLayer

            last_hidden = hidden_sizes[-1]
            # Replace the final linear layer with PopArtLayer
            self.q1_popart = PopArtLayer(last_hidden, 1, beta=popart_beta)
            self.q2_popart = PopArtLayer(last_hidden, 1, beta=popart_beta)
            # Remove the original final layers (they're replaced by PopArt)
            self.q1_layers[-1] = self.q1_popart.linear
            self.q2_layers[-1] = self.q2_popart.linear

        # Initialize weights properly
        self._init_weights()

    def _build_q_network(self, hidden_sizes, float_type):
        """
        Dynamically creates a Q-network given the chosen hidden layer sizes.
        """
        layers = nn.ModuleList()
        input_dim = (
            self.obs_dim + self.act_dim
        )  # Input includes all observations and actions

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h, dtype=float_type))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1, dtype=float_type))  # Output Q-value

        return layers

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""

        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_layer)

    def forward(self, obs, actions):
        """
        Forward pass through both Q-networks.
        """
        xu = th.cat([obs, actions], dim=1)  # Concatenate obs & actions

        # Compute Q1
        x1 = xu
        for layer in self.q1_layers[:-1]:  # All hidden layers
            x1 = F.relu(layer(x1))
        x1 = self.q1_layers[-1](x1)  # Output layer (no activation)

        # Compute Q2
        x2 = xu
        for layer in self.q2_layers[:-1]:  # All hidden layers
            x2 = F.relu(layer(x2))
        x2 = self.q2_layers[-1](x2)  # Output layer (no activation)

        return x1, x2

    def q1_forward(self, obs, actions):
        """
        Compute only Q1 (used during actor updates).
        """
        x = th.cat([obs, actions], dim=1)

        for layer in self.q1_layers[:-1]:  # All hidden layers
            x = F.relu(layer(x))

        x = self.q1_layers[-1](x)  # Output layer (no activation)

        return x


class DistributionalCriticTD3(nn.Module):
    """Twin Q-networks with quantile regression output.

    Instead of Q(s,a) → scalar, each Q-network outputs N quantile values
    of the return distribution Z(s,a), preserving the full reward structure
    (e.g., bimodal distribution at merit-order cliffs).

    The actor uses mean(quantiles) as a drop-in replacement for scalar Q.

    Reference: Dabney et al. (2018) "Distributional Reinforcement Learning
    with Quantile Regression"
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int,
        n_quantiles: int = 32,
    ):
        super().__init__()

        self.obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        self.act_dim = act_dim * n_agents
        self.n_quantiles = n_quantiles

        # Interface compat with CriticTD3 — PopArt guard clauses evaluate False
        self.use_popart = False
        self.q1_popart = None
        self.q2_popart = None

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N)
        taus = (2 * th.arange(n_quantiles, dtype=float_type) + 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)

        if n_agents <= 20:
            hidden_sizes = [256, 128]
        elif n_agents <= 50:
            hidden_sizes = [512, 256, 128]
        else:
            hidden_sizes = [1024, 512, 256, 128]

        # Output dim = n_quantiles instead of 1
        self.q1_layers = self._build_q_network(hidden_sizes, float_type)
        self.q2_layers = self._build_q_network(hidden_sizes, float_type)
        self._init_weights()

    def _build_q_network(self, hidden_sizes, float_type):
        layers = nn.ModuleList()
        input_dim = self.obs_dim + self.act_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h, dtype=float_type))
            input_dim = h
        layers.append(nn.Linear(input_dim, self.n_quantiles, dtype=float_type))
        return layers

    def _init_weights(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_layer)

    def forward(self, obs, actions):
        """Returns (q1_quantiles, q2_quantiles), each shape (batch, N)."""
        xu = th.cat([obs, actions], dim=1)

        x1 = xu
        for layer in self.q1_layers[:-1]:
            x1 = F.relu(layer(x1))
        x1 = self.q1_layers[-1](x1)

        x2 = xu
        for layer in self.q2_layers[:-1]:
            x2 = F.relu(layer(x2))
        x2 = self.q2_layers[-1](x2)

        return x1, x2  # (batch, N), (batch, N)

    def q1_forward(self, obs, actions):
        """Returns mean of Q1 quantiles as (batch, 1) — drop-in for actor loss."""
        x = th.cat([obs, actions], dim=1)
        for layer in self.q1_layers[:-1]:
            x = F.relu(layer(x))
        x = self.q1_layers[-1](x)  # (batch, N)
        return x.mean(dim=1, keepdim=True)  # (batch, 1)

    def quantile_means(self, obs, actions):
        """Returns (q1_mean, q2_mean) each shape (batch, 1)."""
        q1, q2 = self.forward(obs, actions)
        return q1.mean(dim=1, keepdim=True), q2.mean(dim=1, keepdim=True)


class Actor(nn.Module):
    """
    Parent class for actor networks.
    """

    activation_function_limit = {
        "softsign": (-1, 1),
        "tanh": (-1, 1),
        "sigmoid": (0, 1),
        "relu": (0, float("inf")),
    }

    activation_function_map = {
        "softsign": F.softsign,
        "tanh": th.tanh,
        "sigmoid": th.sigmoid,
        "relu": F.relu,
    }

    def __init__(self):
        super().__init__()

        self.activation = "softsign"  # or "tanh", "sigmoid", "relu"

        if self.activation not in self.activation_function_limit:
            raise ValueError(
                f"Activation '{self.activation}' not supported! Supported: {list(self.activation_function_limit.keys())}"
            )
        self.min_output, self.max_output = self.activation_function_limit[
            self.activation
        ]

        self.activation_function = self.activation_function_map.get(self.activation)
        if self.activation_function is None:
            raise ValueError(
                f"Activation '{self.activation}' not implemented in forward pass!"
            )


class MLPActor(Actor):
    """
    The neurnal network for the MLP actor.
    """

    def __init__(self, obs_dim: int, act_dim: int, float_type, *args, **kwargs):
        super().__init__()

        h1 = kwargs.get("actor_hidden_1", 256)
        h2 = kwargs.get("actor_hidden_2", 128)

        self.FC1 = nn.Linear(obs_dim, h1, dtype=float_type)
        self.FC2 = nn.Linear(h1, h2, dtype=float_type)
        self.FC3 = nn.Linear(h2, act_dim, dtype=float_type)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""

        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_layer)

    def forward(self, obs):
        """Forward pass for action prediction."""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        x = self.activation_function(self.FC3(x))

        return x


class StochasticActor(nn.Module):
    """Stochastic actor for SAC — outputs mean and log_std for a Normal distribution.

    Uses the reparameterization trick: sample = tanh(mean + std * noise),
    then squash into (-1, 1) with tanh and correct the log_prob.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, float_type, *args, **kwargs):
        super().__init__()
        self.act_dim = act_dim
        # Compatibility with EnergyLearningStrategy.get_actions() which clamps
        # noise to [min_output, max_output].  tanh output range is (-1, 1).
        self.min_output = -1.0
        self.max_output = 1.0

        h1 = kwargs.get("actor_hidden_1", 256)
        h2 = kwargs.get("actor_hidden_2", 128)

        self.FC1 = nn.Linear(obs_dim, h1, dtype=float_type)
        self.FC2 = nn.Linear(h1, h2, dtype=float_type)
        self.mean_head = nn.Linear(h2, act_dim, dtype=float_type)
        self.log_std_head = nn.Linear(h2, act_dim, dtype=float_type)
        self._init_weights()

    def _init_weights(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_layer)

    def forward(self, obs):
        """Deterministic forward (mean action, squashed via tanh)."""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mean = self.mean_head(x)
        return th.tanh(mean)

    def sample(self, obs):
        """Sample action via reparameterization trick. Returns (action, log_prob)."""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        normal = th.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = th.tanh(x_t)

        # Log-prob with tanh squashing correction
        log_prob = normal.log_prob(x_t) - th.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class DistActor(nn.Module):
    """Stochastic actor for PPO — outputs a Normal distribution with learnable log_std."""

    def __init__(self, obs_dim: int, act_dim: int, float_type, *args, **kwargs):
        super().__init__()
        # Compatibility with EnergyLearningStrategy.get_actions() which clamps
        # noise to [min_output, max_output].  softsign output range is (-1, 1).
        self.min_output = -1.0
        self.max_output = 1.0

        h1 = kwargs.get("actor_hidden_1", 256)
        h2 = kwargs.get("actor_hidden_2", 128)

        self.FC1 = nn.Linear(obs_dim, h1, dtype=float_type)
        self.FC2 = nn.Linear(h1, h2, dtype=float_type)
        self.FC3 = nn.Linear(h2, act_dim, dtype=float_type)
        self.log_std = nn.Parameter(th.zeros(act_dim, dtype=float_type))
        self._init_weights()

    def _init_weights(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_layer)

    def forward(self, obs):
        """Deterministic forward (mean action, squashed via softsign)."""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        return F.softsign(self.FC3(x))

    def get_distribution(self, obs):
        """Return Normal distribution over actions."""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mean = self.FC3(x)
        std = self.log_std.exp().expand_as(mean)
        return th.distributions.Normal(mean, std)

    def evaluate(self, obs, actions):
        """Evaluate log_prob and entropy for given obs-action pairs."""
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class CriticPPO(nn.Module):
    """Single value network V(s) for PPO (not twin-Q like TD3/SAC)."""

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int, float_type, unique_obs_dim: int = 0):
        super().__init__()
        input_dim = obs_dim + unique_obs_dim * (n_agents - 1) + act_dim * n_agents
        self.FC1 = nn.Linear(input_dim, 256, dtype=float_type)
        self.FC2 = nn.Linear(256, 128, dtype=float_type)
        self.FC3 = nn.Linear(128, 1, dtype=float_type)
        self._init_weights()

    def _init_weights(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_layer)

    def forward(self, obs, actions):
        x = th.cat([obs, actions], dim=1)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        return self.FC3(x)


class LSTMActor(Actor):
    """
    The LSTM recurrent neurnal network for the actor.

    Based on "Multi-Period and Multi-Spatial Equilibrium Analysis in Imperfect Electricity Markets"
    by Ye at al. (2019)

    Note: the original source code was not available, therefore this implementation was derived from the published paper.
    Adjustments to resemble final layers from MLPActor:
    - dense layer 2 was omitted
    - single output layer with softsign activation function to output actions directly instead of two output layers for mean and stddev
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int,
        num_timeseries_obs_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.float_type = float_type
        self.unique_obs_dim = unique_obs_dim
        self.num_timeseries_obs_dim = num_timeseries_obs_dim

        try:
            self.timeseries_len = int(
                (obs_dim - unique_obs_dim) / num_timeseries_obs_dim
            )
        except Exception as e:
            raise ValueError(
                f"Using LSTM but not providing correctly shaped timeseries: Expected integer as unique timeseries length, got {(obs_dim - unique_obs_dim) / num_timeseries_obs_dim} instead."
            ) from e

        self.LSTM1 = nn.LSTMCell(num_timeseries_obs_dim, 8, dtype=float_type)
        self.LSTM2 = nn.LSTMCell(8, 16, dtype=float_type)

        # input size defined by forecast horizon and concatenated with capacity and marginal cost values
        self.FC1 = nn.Linear(self.timeseries_len * 16 + 2, 128, dtype=float_type)
        self.FC2 = nn.Linear(128, act_dim, dtype=float_type)

    def forward(self, obs):
        if obs.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {obs.dim()}D instead"
            )

        is_batched = obs.dim() == 2
        if not is_batched:
            obs = obs.unsqueeze(0)

        x1, x2 = obs.split(
            [obs.shape[1] - self.unique_obs_dim, self.unique_obs_dim], dim=1
        )
        x1 = x1.reshape(-1, self.num_timeseries_obs_dim, self.timeseries_len)

        h_t = th.zeros(x1.size(0), 8, dtype=self.float_type, device=obs.device)
        c_t = th.zeros(x1.size(0), 8, dtype=self.float_type, device=obs.device)

        h_t2 = th.zeros(x1.size(0), 16, dtype=self.float_type, device=obs.device)
        c_t2 = th.zeros(x1.size(0), 16, dtype=self.float_type, device=obs.device)

        outputs = []

        for time_step in x1.split(1, dim=2):
            time_step = time_step.reshape(-1, self.num_timeseries_obs_dim)
            h_t, c_t = self.LSTM1(time_step, (h_t, c_t))
            h_t2, c_t2 = self.LSTM2(h_t, (h_t2, c_t2))
            outputs += [h_t2]

        outputs = th.cat(outputs, dim=1)
        x = th.cat((outputs, x2), dim=1)

        x = F.relu(self.FC1(x))
        x = self.activation_function(self.FC2(x))

        if not is_batched:
            x = x.squeeze(0)

        return x
