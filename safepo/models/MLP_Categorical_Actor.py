import torch
import numpy as np
from torch.distributions.categorical import Categorical
from safepo.models.Actor import Actor
from safepo.models.model_utils import build_mlp_network


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,
                 weight_initialization, shared=None):
        super().__init__(obs_dim, act_dim, weight_initialization, shared=shared)
        if shared is not None:
            raise NotImplementedError
        self.net = build_mlp_network(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            weight_initialization=weight_initialization
        )

    def dist(self, obs):
        logits = self.net(obs)
        # inf_tensor = torch.as_tensor(np.asarray([float("-inf")] * len(logits)), dtype=torch.float32)
        # logits_with_mask = torch.where(action_mask > 0, logits, inf_tensor)
        return Categorical(logits=logits)

    def log_prob_from_dist(self, pi, act):
        return pi.log_prob(act)

    def sample(self, obs, action_mask):
        dist = self.dist(obs)
        logits = dist.logits
        inf_tensor = torch.as_tensor(np.asarray([float("-inf")] * len(logits)), dtype=torch.float32)
        logits_with_mask = torch.where(action_mask > 0, logits, inf_tensor)
        dist = Categorical(logits=logits_with_mask)
        a = dist.sample()
        logp_a = self.log_prob_from_dist(dist, a)

        return a, logp_a

    def predict(self, obs, action_mask):
        dist = self.dist(obs)
        logits = dist.logits
        inf_tensor = torch.as_tensor(np.asarray([float("-inf")] * len(logits)), dtype=torch.float32)
        logits_with_mask = torch.where(action_mask > 0, logits, inf_tensor)
        probs = Categorical(logits=logits_with_mask).probs
        a = torch.argmax(probs)
        logp_a = self.log_prob_from_dist(dist, a)

        return a, logp_a