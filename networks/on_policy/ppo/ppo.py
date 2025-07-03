import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal
from CILv2.models.architectures.CIL_multiview.CIL_multiview import CIL_multiview


class ActorCritic(nn.Module):
    def __init__(self, policy_params, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")

        # Create our variable for the matrix.
        # Note that I chose 0.2 for stdev arbitrarily.
        self.action_var = torch.full((action_dim,), action_std * action_std).to(self.device)

        # Create the covariance matrix
        # self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        # actor
        self.actor = CIL_multiview(policy_params).to(self.device)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

    def forward(self):
        raise NotImplementedError

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def get_value(self, s, s_d, s_s):
        with torch.no_grad():
            in_memory, _, _ = self.actor.foward_eval(s, s_d, s_s)  # 输出是 [B, 1, action_dim]
            critic_input = in_memory.squeeze(1)  # [B, 512]
            return self.critic(critic_input)  # [B, 1]

    def get_action_and_log_prob(self, s_img, s_cmd, s_spd, B):
        # print("s_img 类型:", type(s_img), "长度:", len(s_img))
        # print("s_img[0] 类型:", type(s_img[0]), "长度:", len(s_img[0]))
        # print("s_img[0][0] shape:", s_img[0][0].shape)
        try:
            action_out = self.actor(s_img, s_cmd, s_spd, B)  # [B, 1, act_dim]
            # print("→ actor forward 成功")
        except Exception as e:
            print("❌ actor forward 失败:", e)
            raise

        action_mean = action_out.squeeze(1)
        B = action_mean.shape[0]
        epsilon = 1e-6
        cov_mat = torch.diag(self.action_var + epsilon).unsqueeze(0).expand(B, -1, -1)  # [B, action_dim, action_dim]

        dist = MultivariateNormal(action_mean, covariance_matrix=cov_mat)
        # print("action_mean shape:", action_mean.shape)
        # print("cov_mat shape:", cov_mat.shape)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, s_img, s_cmd, s_spd, actions, B):
        action_out = self.actor(s_img, s_cmd, s_spd, B).squeeze(1)
        cov_mat = torch.diag(self.action_var).unsqueeze(0).expand(action_out.size(0), -1, -1)  # shape [B, 3, 3]
        dist = MultivariateNormal(action_out, covariance_matrix=cov_mat)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        with torch.no_grad():  # 可选
            _, in_memory, _ = self.actor.foward_eval(s_img, s_cmd, s_spd)  # [B, 1, 512]
            critic_input = in_memory.squeeze(1)  # [B, 512]
        state_value = self.critic(critic_input)
        # state_value = self.critic(action_out)  # ****
        return action_logprobs, state_value.squeeze(), dist_entropy