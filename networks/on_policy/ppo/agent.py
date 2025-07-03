import os
import numpy as np
import yaml
import torch
import torch.nn as nn
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import *
from configs.attribute_dict import AttributeDict
from configs import g_conf
from configs._global import _merge_a_into_b

device = torch.device("cpu")


class Buffer:
    def __init__(self):
        # Batch data
        self.observation = []  # list of (s, s_d, s_s)
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.observation[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]


class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        self.device = torch.device("cpu")
        # self.env = env
        self.obs_dim = 100
        self.action_dim = 3
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        self.checkpoint_file_no = 0
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(BASE_DIR, "CILv2_3cam_smalltest.yaml")
        # yaml_path = 'CILv2_3cam_smalltest.yaml'
        self.load_policy_params_from_yaml(yaml_path)
        policy_params = g_conf.MODEL_CONFIGURATION
        self.policy = ActorCritic(policy_params, self.action_dim, self.action_std)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(policy_params, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def load_policy_params_from_yaml(self, yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("YAML 内容不是字典结构")
        if 'MODEL_CONFIGURATION' not in config:
            raise KeyError("YAML 文件中缺少 'MODEL_CONFIGURATION' 字段。")

        if not isinstance(config['MODEL_CONFIGURATION'], dict):
            raise TypeError(" MODEL_CONFIGURATION 不是一个字典")

        g_conf.MODEL_CONFIGURATION = AttributeDict(config['MODEL_CONFIGURATION'])

    def get_action(self, obs, train):
        s, s_d, s_s = obs  # 解包三元组
        s = [[img.unsqueeze(0).to(self.device) for img in frame] for frame in s]  # [S][cam][1, 3, H, W]
        s_d = [x.unsqueeze(0).to(self.device) if x.dim() == 1 else x.to(self.device) for x in s_d]
        s_s = [x.unsqueeze(0).to(self.device) if x.dim() == 1 else x.to(self.device) for x in s_s]
        B = s_s[0].shape[0]
        with torch.no_grad():
            action, logprob = self.old_policy.get_action_and_log_prob(s, s_d, s_s, B)
        if train:
            s_clean = [[img for img in frame] for frame in s]  # 保持 [S][cam][C,H,W] 结构
            s_d_clean = s_d[0]  # Tensor(1,4)
            s_s_clean = s_s[0]  # Tensor(1,1)
            self.memory.observation.append((s_clean, s_d_clean, s_s_clean))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        return action.detach().cpu().numpy().flatten()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std

    def learn(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        s_list, s_d_list, s_s_list = zip(*self.memory.observation)
        old_s = [[torch.stack([
            s_item[0][0].squeeze(0).to(device)  # s_item 是 [[Tensor(3, H, W)]]
            for s_item in s_list
        ], dim=0)]]  # → [B, 3, H, W]
        old_s_d = [torch.cat(s_d_list, dim=0).to(device)]  # → old_s_d: [B, 4]
        old_s_s = [torch.cat(s_s_list, dim=0).to(device)]  # → old_s_s: [B, 1]
        B = old_s_d[-1].shape[0]
        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        # old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        # old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)
        old_actions = torch.stack(self.memory.actions, dim=0).detach().to(device)
        if old_actions.dim() == 3 and old_actions.shape[1] == 1:
            old_actions = old_actions.squeeze(1)  # [B, 1, action_dim] → [B, action_dim]
        elif old_actions.dim() == 1:
            old_actions = old_actions.unsqueeze(1)  # [B] → [B, 1] if action_dim == 1
        old_logprobs = torch.stack(self.memory.log_probs, dim=0).detach().to(device)
        if old_logprobs.dim() == 2 and old_logprobs.shape[1] == 1:
            old_logprobs = old_logprobs.squeeze(1)  # [B, 1] → [B]
        elif old_logprobs.dim() > 2:
            old_logprobs = old_logprobs.view(-1)  # flatten to [B]

        # Optimize policy for K epochs
        for _ in range(self.n_updates_per_iteration):
            # Evaluating old actions and values
            logprobs, values, dist_entropy = self.policy.evaluate(old_s, old_s_d, old_s_s, old_actions, B)

            # match values tensor dimensions with rewards tensor
            values = torch.squeeze(values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR + self.town + "/ppo_policy_" + str(self.checkpoint_file_no) + "_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2])
        if self.checkpoint_file_no != 0:
            self.checkpoint_file_no -= 1
        checkpoint_file = PPO_CHECKPOINT_DIR + self.town + "/ppo_policy_" + str(self.checkpoint_file_no) + "_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def load(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2]) - 1
        checkpoint_file = PPO_CHECKPOINT_DIR + self.town + "/ppo_policy_" + str(self.checkpoint_file_no) + "_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))
