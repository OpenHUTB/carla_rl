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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # 动态调整的协方差矩阵
        self.cov_var = nn.Parameter(
            torch.full((action_dim,),
                       min(max(action_std_init, 0.1), 0.5),  # 限制初始标准差范围
                       device=self.device)
        )

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
    #
    # def _get_shared_features(self, s, s_d, s_s):
    #     """
    #         特征提取阶段
    #     """
    #     s = self._to_device(s)
    #     s_d = self._to_device(s_d)
    #     s_s = self._to_device(s_s)
    #     """共享 CIL 的 Transformer 特征提取"""
    #     # 输入统一处理（同时兼容Tensor和numpy输入）
    #     # 1. 图像序列处理
    #     if isinstance(s, list):
    #         if isinstance(s[0], list):
    #             # 处理双层列表输入 [[img1, img2,...]]
    #             frame_list = []
    #             for img in s[0]:
    #                 if isinstance(img, np.ndarray):
    #                     # numpy数组处理
    #                     frame_list.append(torch.from_numpy(img).float().permute(2, 0, 1))
    #                 elif isinstance(img, torch.Tensor):
    #                     # 已经是张量的情况
    #                     if img.dim() == 3 and img.size(0) == 3:  # 已经是CHW格式
    #                         frame_list.append(img.float())
    #                     else:
    #                         frame_list.append(img.float().permute(2, 0, 1))
    #                 else:
    #                     raise TypeError(f"不支持的图像类型: {type(img)}")
    #             s = torch.stack(frame_list, dim=0)  # [N,C,H,W]
    #         else:
    #             # 处理单层列表输入 [img]
    #             img = s[0]
    #             if isinstance(img, np.ndarray):
    #                 s = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    #             elif isinstance(img, torch.Tensor):
    #                 s = img.float().permute(2, 0, 1).unsqueeze(0) if img.dim() == 3 else img.unsqueeze(0)
    #     elif isinstance(s, torch.Tensor):
    #         # 已经是张量输入
    #         if s.dim() == 3:  # [C,H,W]
    #             s = s.unsqueeze(0)  # [1,C,H,W]
    #         elif s.dim() == 4:  # [B,C,H,W]
    #             pass
    #         else:
    #             raise ValueError(f"非法张量维度: {s.dim()}")
    #     else:
    #         raise TypeError(f"不支持的输入类型: {type(s)}")
    #
    #     # 添加序列和相机维度 (S=1, cam=1)
    #     x = s.unsqueeze(1).unsqueeze(1)  # [B,1,1,C,H,W]
    #
    #     # 获取批量大小
    #     B = x.size(0)
    #
    #     # 重塑图像数据
    #     x = x.reshape(B * 1 * 1, *g_conf.IMAGE_SHAPE)  # [B,C,H,W]
    #     # 处理速度和命令输入
    #     d = s_d[-1]
    #     if d.dim() == 1:  # 如果是 [4]
    #         d = d.unsqueeze(0).expand(B, -1)  # -> [1,4] -> [B,4]
    #     elif d.dim() == 2:  # 如果是 [B,4]
    #         assert d.size(0) == B, f"命令batch不匹配: {d.size(0)} vs {B}"
    #     else:
    #         raise ValueError(f"非法命令维度: {d.shape}")
    #
    #     s = s_s[-1]
    #     if s.dim() == 1:  # 如果是 [4]
    #         s = s.unsqueeze(0).expand(B, -1)  # -> [1,4] -> [B,4]
    #     elif s.dim() == 2:  # 如果是 [B,4]
    #         assert s.size(0) == B, f"命令batch不匹配: {s.size(0)} vs {B}"
    #     else:
    #         raise ValueError(f"非法命令维度: {s.shape}")
    #
    #     # 2. ResNet提取图像特征（相当于人眼视觉处理）
    #     e_p, _ = self.cil.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
    #     encoded_obs = e_p.view(B, 1 * 1, self.cil.res_out_dim, -1)
    #     encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.cil.res_out_dim)
    #     # 投影层（如果存在）
    #     if hasattr(self, 'projection'):
    #         encoded_obs = self.projection(encoded_obs)
    #
    #     # 3. 融合命令和速度特征（相当于结合驾驶意图）
    #     e_d = self.cil.command(d).unsqueeze(1)  # [B, 1, 512]
    #     e_s = self.cil.speed(s).unsqueeze(1)  # [B, 1, 512]
    #     encoded_obs = encoded_obs + e_d + e_s  # [B, S*cam*h*w, 512]
    #
    #     # 4. 位置编码,Transformer时空建模（相当于大脑综合分析）
    #     if self.cil.params['TxEncoder']['learnable_pe']:
    #         pe = encoded_obs + self.cil.positional_encoding
    #     else:
    #         pe = self.cil.positional_encoding(encoded_obs)
    #
    #     # 5. Transformer编码
    #     in_memory, _ = self.cil.tx_encoder(pe)  # [B, S*cam*h*w, 512]
    #
    #     return torch.mean(in_memory, dim=1)  # [B, 512]
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

    # def get_action_and_log_prob(self, s, s_d, s_s):
    #
    #     """
    #     ---->> 策略网络
    #     决策计算阶段
    #        Actor：采样动作并计算概率
    #     """
    #     # 获取动作均值（通过 CIL 的 action_output）
    #     action_mean = self.cil(s, s_d, s_s).squeeze(1)  # [B, action_dim]
    #     # print(f"action_mean shape: {action_mean.shape}")  # 调试
    #     # 分维度处理动作范围
    #     steering = torch.tanh(action_mean[:, 0])  # 方向盘
    #     throttle = torch.tanh(action_mean[:, 1])  # 油门
    #     brake = torch.tanh(action_mean[:, 2])  # 刹车
    #
    #     # 重新组合动作
    #     processed_mean = torch.stack([steering, throttle, brake], dim=1)
    #
    #     # 定义高斯分布,动作分布（带探索噪声）， 通过多元高斯分布增加随机性，实现策略探索
    #     cov_mat = torch.diag_embed(self.cov_var.expand_as(processed_mean))
    #     cov_mat = cov_mat + torch.eye(self.action_dim, device=self.device) * 1e-6
    #     dist = MultivariateNormal(processed_mean, cov_mat)
    #
    #     # 采样动作（保留梯度）
    #     with torch.no_grad():  # 只在采样时阻断梯度
    #         action = dist.sample()
    #         action = torch.clamp(action, -1, 1)  # 全部限制到[-1,1]
    #     log_prob = dist.log_prob(action)  # 计算当前动作在策略分布下的对数概率，用于PPO重要性采样,后续训练时会用这个评分判断：当前决策是该奖励还是惩罚
    #     return action, log_prob
    def get_action_and_log_prob(self, s, s_d, s_s):
        # 修改1：强制FP32计算
        with torch.cuda.amp.autocast(enabled=False):
            in_memory, action_mean = self._get_shared_features_with_mean(s, s_d, s_s)
            action_mean = torch.tanh(action_mean) * 1.5  # 扩大输出范围

        # 修改2：更稳定的协方差矩阵
        action_std = torch.clamp(self.cov_var, min=0.05, max=0.3)  # 限制标准差
        cov_mat = torch.diag_embed(action_std) + 1e-4 * torch.eye(self.action_dim, device=self.device)

        # 修改3：使用更稳定的分布
        dist = MultivariateNormal(action_mean.float(), cov_mat.float())

        with torch.no_grad():
            action = dist.sample().clamp(-1, 1)

        return action, dist.log_prob(action.float())  # 确保FP32

    def evaluate(self, s, s_d, s_s, action):
        """
        评估动作概率和状态价值（安全版本）
        """
        self.cil.eval()  # 将网络切换到评估模式

        # 1. 输入数据强制标准化处理
        with torch.no_grad():
            # 确保所有输入在相同设备且为float32
            s = s.to(self.device).float()
            action = action.to(self.device).float()

            # 处理命令输入
            if isinstance(s_d, (list, tuple)):
                s_d = [x.to(self.device).float() if torch.is_tensor(x) else torch.tensor(x, device=self.device).float()
                       for x in s_d]
                if s_d[-1].dim() == 1:
                    s_d[-1] = s_d[-1].unsqueeze(0).expand(s.size(0), -1)
            else:
                s_d = s_d.to(self.device).float()

        # 2. 安全获取特征和均值
        with torch.no_grad():
            in_memory, action_mean = self._get_shared_features_with_mean(s, s_d, s_s)

            # 3. 安全构建分布（关键修改）
            cov_var = self.cov_var.expand_as(action_mean)
            cov_var = torch.clamp(cov_var, min=0.01, max=1.0)  # 更严格的截断

            # 双重保证协方差矩阵正定性
            eye_matrix = torch.eye(action_mean.size(-1), device=self.device) * 1e-4
            cov_mat = torch.diag_embed(cov_var) + eye_matrix

            # 4. 安全创建分布（强制CPU计算+无梯度）
            try:
                # 先尝试GPU计算
                dist = MultivariateNormal(
                    action_mean,
                    cov_mat,
                    validate_args=False
                )
                logprobs = dist.log_prob(action)
                entropy = dist.entropy()
            except RuntimeError:
                # GPU失败时回退到CPU
                dist = MultivariateNormal(
                    action_mean.cpu(),
                    cov_mat.cpu(),
                    validate_args=False
                )
                logprobs = dist.log_prob(action.cpu()).to(self.device)
                entropy = dist.entropy().to(self.device)

        # 5. 单独计算值函数（保持梯度）
        with torch.enable_grad():
            values = self.value_head(in_memory.detach())  # 阻断特征提取梯度

        return logprobs, values, entropy