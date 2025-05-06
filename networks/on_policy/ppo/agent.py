import os
import numpy as np
import torch
import torch.nn as nn
from parameters import *
from configs import g_conf
from torch.distributions import MultivariateNormal
from networks.on_policy.ppo.ppo import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.frames = []  # List[List[torch.Tensor]] 图像序列 [S][B,C,H,W]
        self.commands = []  # List[torch.Tensor] 控制命令 [B,4]
        self.speeds = []  # List[torch.Tensor] 速度信息 [B,1]
        self.actions = []  # List[torch.Tensor] 执行动作 [B,action_dim]
        self.log_probs = []  # List[torch.Tensor] 动作概率 [B]
        self.rewards = []  # List[float] 即时奖励
        self.dones = []  # List[bool] 终止标志

    def clear(self):
        self.frames.clear()
        self.commands.clear()
        self.speeds.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return self.max_size is not None and len(self) >= self.max_size


class PPOAgent:
    def __init__(self, town, action_std_init=0.4, buffer_size=None):
        # 参数验证
        assert action_std_init > 0, "动作标准差必须大于0"

        # 环境参数
        self.town = town
        self.action_dim = 3  # ['steer', 'throttle', 'brake']

        # 超参数
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.batch_size = BATCH_SIZE

        # 初始化网络
        cil_params = {
            'encoder_embedding': {
                'perception': {'res': {'name': 'resnet34', 'layer_id': 3}}
            },
            'TxEncoder': {
                'd_model': 512,
                'n_head': 8,
                'num_layers': 3,
                'norm_first': True,
                'learnable_pe': False
            },
            'action_output': {
                'fc': {'neurons': [256, 128], 'dropouts': [0.1, 0.1]}
            }
        }

        # 模型初始化
        self.policy = ActorCritic(cil_params, self.action_dim, self.action_std).to(device)
        self.old_policy = ActorCritic(cil_params, self.action_dim, self.action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.lr,
            eps=1e-5
        )

        self.MseLoss = nn.MSELoss()
        self.memory = Buffer(buffer_size)
        self.checkpoint_file_no = 0

    def store_reward_done(self, reward, done):
        """存储奖励和终止标志"""
        assert not self.memory.is_empty(), "必须先调用get_action()存储观测数据"
        self.memory.rewards.append(float(reward))
        self.memory.dones.append(bool(done))

    def get_action(self, obs_dict, train=True):
        """
        处理单摄像头的单帧图像输入
        """
        # 输入验证
        assert len(obs_dict['frames']) == 1, "只支持单摄像头输入"
        assert len(obs_dict['frames'][0]) == 1, "只支持单帧图像"

        with torch.no_grad():
            # 提取唯一的单帧图像
            img = obs_dict['frames'][0][0]  # 获取单帧 (H,W,C)

            # 转换为PyTorch张量
            img = np.ascontiguousarray(img)  # 确保内存连续
            if img.shape[-1] == 3:  # HWC格式
                img = img.transpose(2, 0, 1)  # 转为CHW (C,H,W)
            frame_tensor = torch.from_numpy(img).float().to(device)  # 最终形状 (C,H,W)

            # 处理其他数据（command/speed等）
            command = torch.as_tensor(obs_dict['command'], dtype=torch.float32).to(device)
            speed = torch.as_tensor(obs_dict['speed'], dtype=torch.float32).to(device)

            # 获取动作
            action, logprob = self.old_policy.get_action_and_log_prob(
                [frame_tensor],  # 注意：保持输入为列表形式以兼容网络接口
                command.unsqueeze(0) if command.dim() == 1 else command,
                speed.unsqueeze(-1) if speed.dim() == 1 else speed
            )

        # 存储经验（如果需要）
        if train and not self.memory.is_full():
            self._store_transition(frame_tensor, command, speed, action, logprob)

        return action.cpu().numpy().flatten()

    def _store_transition(self, frames, command, speed, action, logprob):
        """存储单步转移数据"""
        assert command.shape[-1] == 4, f"命令维度错误，应为4，得到{command.shape[-1]}"
        self.memory.frames.append(frames)
        if command.dim() == 1:
            command = command.unsqueeze(0)  # [4] -> [1, 4]
        if speed.dim() == 1:
            speed = speed.unsqueeze(-1)  # [1] -> [1,1]
        elif speed.dim() == 0:
            speed = speed.unsqueeze(0).unsqueeze(-1)  # 标量 -> [1,1]

        self.memory.commands.append(command)
        self.memory.speeds.append(speed)
        self.memory.actions.append(action)
        self.memory.log_probs.append(logprob)

    def learn(self):
        # 计算优势估计，返回的蒙特卡洛估计
        """执行PPO学习更新"""
        if len(self.memory) < self.batch_size:
            return

        # 1. 计算折扣奖励
        rewards = self._compute_discounted_rewards().detach()

        # 2. 用于从经验回放缓冲区（memory）中提取数据并进行预处理
        # batch = self._prepare_batch()
        # 修改数据准备部分，显式启用梯度
        old_f = torch.stack(self.memory.frames, dim=0).to(device).requires_grad_(True)
        old_commands = torch.cat(self.memory.commands, dim=0).to(device).requires_grad_(True)
        old_speeds = torch.cat(self.memory.speeds, dim=0).to(device).requires_grad_(True)
        old_actions = torch.stack(self.memory.actions, dim=0).to(device).requires_grad_(True)

        # 只有old_logprobs不需要梯度（因为是旧策略产生的）
        old_logprobs = torch.stack(self.memory.log_probs, dim=0).to(device).detach()

        # 确保模型在训练模式
        self.policy.train()
        # 3. PPO优化,  # 对K个时期epoch优化策略
        for _ in range(self.n_updates_per_iteration):

            # loss = self._compute_ppo_loss(old_f, old_commands, old_speeds, old_actions)
            # 估计旧的动作和值
            logprobs, values, dist_entropy = self.policy.evaluate(old_f, old_commands, old_speeds, old_actions)
            print(f"网络输出梯度: logprobs={logprobs.requires_grad}, values={values.requires_grad}")
            # 以奖励张量匹配值张量的维度
            values = torch.squeeze(values)
            # 找到比例 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            # 找到代理损失 Surrogate Loss
            advantages = rewards - values.detach()
            print(f"中间变量梯度: ratios={ratios.requires_grad}, advantages={advantages.requires_grad}")
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy
            print(f"最终loss梯度: {loss.requires_grad}")
            try:
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            except RuntimeError as e:
                print(f"梯度更新异常: {e}")
                torch.cuda.empty_cache()  # 清理GPU缓存
        # 4. 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def _compute_discounted_rewards(self):
        """计算折扣奖励并归一化"""
        rewards = np.zeros(len(self.memory), dtype=np.float32)
        discounted = 0.0

        for i in reversed(range(len(self.memory))):
            if self.memory.dones[i]:
                discounted = 0.0
            discounted = self.memory.rewards[i] + self.gamma * discounted
            rewards[i] = discounted

        rewards = torch.as_tensor(rewards, device=device)
        return (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    def _prepare_batch(self):
        """准备训练批量数据"""
        # 按时间步重组帧数据
        # 使用列表推导式+并行stack
        # frames = [torch.stack([seq[t] for seq in self.memory.frames[0]])
        #           for t in range(g_conf.ENCODER_INPUT_FRAMES_NUM)]
        # 从单个相机的帧序列中提取第一帧，并将其转换为PyTorch张量，最后放入一个列表中。
        frames = [torch.stack([seq[0] for seq in self.memory.frames[0]])]
        return {
            'frames': frames,
            'commands': torch.cat(self.memory.commands),
            'speeds': torch.cat(self.memory.speeds),
            'old_actions': torch.cat(self.memory.actions),
            'old_logprobs': torch.cat(self.memory.log_probs)
        }

    # def _compute_ppo_loss(self, old_f, old_commands, old_speeds, old_actions):
    #     """计算PPO损失"""
    #     # 估计旧的动作和值
    #     logprobs, values, entropy = self.policy.evaluate(
    #         batch['frames'],
    #         batch['commands'],
    #         batch['speeds'],
    #         batch['old_actions']
    #     )
    #
    #     # 维度处理
    #     values = values.view(-1)
    #     advantages = rewards - values.detach()
    #     ratios = torch.exp(logprobs - batch['old_logprobs'].detach())
    #
    #     # 计算PPO损失
    #     surr1 = ratios * advantages
    #     surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
    #
    #     policy_loss = -torch.min(surr1, surr2).mean()
    #     value_loss = 0.5 * self.MseLoss(values, rewards)
    #     entropy_bonus = -0.01 * entropy.mean()
    #
    #     return policy_loss + value_loss + entropy_bonus

    def set_action_std(self, new_action_std):
        """设置新的动作标准差"""
        assert new_action_std > 0, f"动作标准差必须大于0，得到{new_action_std}"
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, decay_rate, min_std=0.1):
        """衰减探索噪声"""
        self.action_std = max(self.action_std - decay_rate, min_std)
        self.set_action_std(self.action_std)
        return self.action_std

    def save(self, path=None, is_checkpoint=False):
        """保存模型状态"""
        if path is None:
            save_dir = os.path.join(PPO_CHECKPOINT_DIR, self.town)
            os.makedirs(save_dir, exist_ok=True)

            existing = [f for f in os.listdir(save_dir) if f.startswith('ppo_') and f.endswith('.pth')]
            if is_checkpoint and existing:
                # 检查点模式使用最大编号
                self.checkpoint_file_no = max(
                    int(f.split('_')[1].split('.')[0]) for f in existing
                )
            else:
                # 新文件使用当前数量作为编号
                self.checkpoint_file_no = len(existing)

            path = os.path.join(save_dir, f"ppo_{self.checkpoint_file_no}.pth")

        torch.save({
            'model_state': self.old_policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'action_std': self.action_std,
            'config': {
                'clip': self.clip,
                'gamma': self.gamma,
                'lr': self.lr,
                'town': self.town,
                'n_updates': self.n_updates_per_iteration,
                'batch_size': self.batch_size
            }
        }, path)

    def chkpt_save(self):
        """保存检查点(覆盖最新)"""
        self.save(is_checkpoint=True)

    def load(self, path=None):
        """加载模型状态"""
        if path is None:
            save_dir = os.path.join(PPO_CHECKPOINT_DIR, self.town)
            existing = [f for f in os.listdir(save_dir) if f.startswith('ppo_') and f.endswith('.pth')]
            if not existing:
                raise FileNotFoundError(f"No model found in {save_dir}")

            # 自动加载最新模型
            latest = max(
                existing,
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )
            path = os.path.join(save_dir, latest)

        checkpoint = torch.load(path, map_location=device)

        # 加载模型参数
        self.old_policy.load_state_dict(checkpoint['model_state'])
        self.policy.load_state_dict(checkpoint['model_state'])

        # 加载优化器状态并转移到当前设备
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        # 恢复配置
        config = checkpoint.get('config', {})
        self.action_std = config.get('action_std', 0.4)
        self.set_action_std(self.action_std)
        self.n_updates_per_iteration = config.get('n_updates', 7)
        self.batch_size = config.get('batch_size', 64)