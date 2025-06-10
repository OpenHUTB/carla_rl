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
        self.device = torch.device("cpu")

        self.action_dim = action_dim   # ['steer', 'throttle', 'brake']

        # 特征提取主干网
        # 初始化 CIL_multiview 作为共享特征提取器
        self.cil = CIL_multiview(cil_params).to(self.device)
        # 正确初始化最后一个Linear层（不是Tanh）
        for name, module in self.cil.action_output.named_modules():
            if isinstance(module, nn.Linear):
                if name == '6':  # 根据实际结构调整这个索引
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    nn.init.constant_(module.bias, 0.0)

        self.projection = nn.Linear(256, 512)  # 将 256 维 -> 512 维
        self._convert_image = self._create_image_converter()
        # Create our variable for the matrix.
        # Note that I chose 0.2 for stdev arbitrarily.
        self.cov_var = torch.full((self.action_dim,), action_std_init, device=self.device)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        # Critic 专用价值头
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, 0.0)

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

    def _create_image_converter(self):
        """创建统一的图像转换方法"""

        def convert(img):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            elif isinstance(img, torch.Tensor):
                img = img.float()
            else:
                raise TypeError(f"不支持的图像类型: {type(img)}")

            # 确保图像是CHW格式
            if img.dim() == 3 and img.size(0) == 3:  # 已经是CHW
                return img
            elif img.dim() == 3:  # HWC转CHW
                return img.permute(2, 0, 1)
            else:
                raise ValueError(f"非法图像维度: {img.dim()}")

        return convert

    def _preprocess_input(self, s):
        """统一输入预处理"""
        if isinstance(s, list):
            # 处理列表输入
            if isinstance(s[0], list):
                # 双层列表 [[img1, img2,...]]
                return torch.stack([torch.stack(
                    [self._convert_image(img) for img in s[0]],
                    dim=0
                )])
            else:
                # 单层列表 [img]
                return torch.stack([self._convert_image(s[0])])
        elif isinstance(s, torch.Tensor):
            # 张量输入
            if s.dim() == 3:  # [C,H,W]
                return s.unsqueeze(0)
            elif s.dim() == 4:  # [B,C,H,W]
                return s
            else:
                raise ValueError(f"非法张量维度: {s.dim()}")
        else:
            raise TypeError(f"不支持的输入类型: {type(s)}")

    def _get_shared_features_with_mean(self, s, s_d, s_s):
        """整合特征提取和动作均值计算的完整实现"""
        # 1. 图像预处理
        x = self._preprocess_input(s)  # [B,C,H,W]

        # 2. 添加序列和相机维度 (S=1, cam=1)
        B = x.size(0)
        x = x.unsqueeze(1).unsqueeze(1)  # [B,1,1,C,H,W]
        x = x.reshape(B * 1 * 1, *g_conf.IMAGE_SHAPE)  # [B*1 * 1,C,H,W]

        # 3. 提取视觉特征
        e_p, _ = self.cil.encoder_embedding_perception(x)  # [B*1 * 1, dim, h, w]
        encoded_obs = e_p.view(B, 1 * 1, self.cil.res_out_dim, -1)  # [B,1,dim,h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.cil.res_out_dim)  # [B,1*h*w,dim]

        # 4. 投影层（如果存在）
        if hasattr(self, 'projection'):
            encoded_obs = self.projection(encoded_obs)  # [B,1*h*w,512]

        # 5. 处理命令和速度
        d = s_d[-1] if isinstance(s_d, (list, tuple)) else s_d
        s = s_s[-1] if isinstance(s_s, (list, tuple)) else s_s

        e_d = self.cil.command(d).unsqueeze(1)  # [B,1,512]
        e_s = self.cil.speed(s).unsqueeze(1)  # [B,1,512]
        encoded_obs = encoded_obs + e_d + e_s  # [B,1*h*w,512]

        # 6. 位置编码
        if self.cil.params['TxEncoder']['learnable_pe']:
            pe = encoded_obs + self.cil.positional_encoding
        else:
            pe = self.cil.positional_encoding(encoded_obs)

        # 7. Transformer编码
        in_memory, _ = self.cil.tx_encoder(pe)  # [B,1*h*w,512]
        in_memory = torch.mean(in_memory, dim=1)  # [B,512]

        # 8. 计算动作均值
        action_mean = self.cil.action_output(in_memory).squeeze(1)  # [B,action_dim]

        return in_memory, action_mean

    def get_action_and_log_prob(self, s, s_d, s_s):
        # 修改1：强制FP32计算
        with torch.cuda.amp.autocast(enabled=False):
            in_memory, action_mean = self._get_shared_features_with_mean(s, s_d, s_s)
            action_mean = torch.tanh(action_mean) * 1.5  # 扩大输出范围

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(action_mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach(), log_prob.detach()

    def evaluate(self, s, s_d, s_s, action):
        """
           评估动作概率和状态价值
        """
        self.cil.eval()  # 将网络切换到评估模式（关闭dropout等）

        # 确保输入在正确设备上
        s = s.to(self.device)
        action = action.to(self.device)

        # 处理命令输入
        if isinstance(s_d, (list, tuple)):
            s_d = [x.to(self.device) if torch.is_tensor(x) else x for x in s_d]
            if torch.is_tensor(s_d[-1]) and s_d[-1].dim() == 1:
                s_d[-1] = s_d[-1].unsqueeze(0).expand(s.size(0), -1)
        else:
            s_d = s_d.to(self.device)

        with torch.no_grad():
            # 禁用自动混合精度，改为手动控制
            with torch.cuda.amp.autocast(enabled=False):
                # action_mean = self.cil(s, s_d, s_s).squeeze(1)
                in_memory, action_mean = self._get_shared_features_with_mean(s, s_d, s_s)
                # in_memory: 状态特征 [B,512]
                # action_mean: 动作均值 [B,action_dim]
                # 协方差矩阵参数
                cov_var = self.cov_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(cov_var)
                action_mean = torch.tanh(action_mean) * 1.5  # 扩大输出范围
                dist = MultivariateNormal(action_mean, cov_mat)
                # 计算对数概率和熵
                logprobs = dist.log_prob(action)
                entropy = dist.entropy()
        # 单独计算需要梯度的值函数
        with torch.enable_grad():
            values = self.value_head(in_memory)   # 状态价值估计
        return logprobs, values, entropy
