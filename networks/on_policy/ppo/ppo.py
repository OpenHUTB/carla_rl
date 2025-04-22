import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from configs import g_conf
from torch.distributions import MultivariateNormal
from CILv2.models.architectures.CIL_multiview.CIL_multiview import CIL_multiview


class ActorCritic(nn.Module):
    def __init__(self, cil_params, action_dim, action_std_init):
        super().__init__()
        self.action_dim = action_dim   # 动作维度（如转向、油门）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化 CIL_multiview 作为共享特征提取器
        self.cil = CIL_multiview(cil_params).to(self.device)

        # 动态调整的协方差矩阵
        self.cov_var = nn.Parameter(
            torch.full((action_dim,), action_std_init, device=self.device)
        )

        # Critic 专用价值头（共享 CIL 的 Transformer 特征）
        self.value_head = nn.Linear(cil_params['TxEncoder']['d_model'], 1)
        # 初始化权重
        for m in [self.value_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self):
        raise NotImplementedError

    def set_action_std(self, new_action_std):
        """动态调整探索幅度"""
        self.cov_var.data = torch.full_like(self.cov_var, new_action_std)

    def get_value(self, s, s_d, s_s):
        """Critic：预测状态价值"""
        # 共享 CIL 的特征提取
        in_memory = self._get_shared_features(s, s_d, s_s)  # [B, 512]
        return self.value_head(in_memory)  # [B, 1]

    def _to_device(self, x):
        """统一的设备转换方法"""
        if isinstance(x, (list, tuple)):
            return [t.to(self.device) for t in x]
        elif isinstance(x, dict):
            return {k: v.to(self.device) for k, v in x.items()}
        return x.to(self.device)

    def _get_shared_features(self, s, s_d, s_s):
        """
            特征提取阶段
        """
        s = self._to_device(s)
        s_d = self._to_device(s_d)
        s_s = self._to_device(s_s)
        """共享 CIL 的 Transformer 特征提取"""
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d.shape[0] if s_d.dim() > 1 else 1
        if not isinstance(s, (list, tuple)) or len(s) == 0:
            raise ValueError("输入s必须是包含至少一个元素的序列")

        # 1. 图像序列处理
        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * len(g_conf.DATA_USED), *g_conf.IMAGE_SHAPE)  # [B*S*cam, 3, H, W]

        # 2. ResNet提取图像特征（相当于人眼视觉处理）
        e_p, _ = self.cil.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S * len(g_conf.DATA_USED), self.cil.res_out_dim, -1)
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.cil.res_out_dim)

        # 3. 融合命令和速度特征（相当于结合驾驶意图）
        e_d = self.cil.command(s_d[-1]).unsqueeze(1)  # [B, 1, 512]
        e_s = self.cil.speed(s_s[-1]).unsqueeze(1)  # [B, 1, 512]
        encoded_obs = encoded_obs + e_d + e_s  # [B, S*cam*h*w, 512]

        # 4. 位置编码,Transformer时空建模（相当于大脑综合分析）
        if self.cil.params['TxEncoder']['learnable_pe']:
            pe = encoded_obs + self.cil.positional_encoding
        else:
            pe = self.cil.positional_encoding(encoded_obs)

        # 5. Transformer编码
        in_memory, _ = self.cil.tx_encoder(pe)  # [B, S*cam*h*w, 512]

        return torch.mean(in_memory, dim=1)  # [B, 512]

    def get_action_and_log_prob(self, s, s_d, s_s):

        """
        决策计算阶段
           Actor：采样动作并计算概率
        """
        # 获取动作均值（通过 CIL 的 action_output）
        action_mean = self.cil(s, s_d, s_s).squeeze(1)  # [B, action_dim]
        print(f"action_mean shape: {action_mean.shape}")  # 调试
        action_mean = torch.tanh(action_mean)  # 限制到 [-1, 1]

        # 定义高斯分布,动作分布（带探索噪声）
        cov_mat = torch.diag_embed(self.cov_var.expand_as(action_mean))
        cov_mat = cov_mat + torch.eye(self.action_dim, device=self.device) * 1e-6
        dist = MultivariateNormal(action_mean, cov_mat)

        # 采样动作
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def evaluate(self, s, s_d, s_s, action):
        """
           控制输出阶段
          PPO 训练阶段评估
          """
        self.cil.eval()
        try:
            with torch.no_grad():
                action_mean = self.cil(s, s_d, s_s).squeeze(1)
                action_mean = torch.tanh(action_mean)

                cov_mat = torch.diag_embed(self.cov_var.expand_as(action_mean))
                dist = MultivariateNormal(action_mean, cov_mat)

                logprobs = dist.log_prob(action)
                entropy = dist.entropy()
                values = self.get_value(s, s_d, s_s)
            return logprobs, values, entropy
        finally:
            self.cil.train()
            torch.cuda.empty_cache()  # 如果是GPU环境
