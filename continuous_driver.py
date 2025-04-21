import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *

import carla


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town07", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()

    return args


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def runner():
    # ========================================================================
    #                           基本参数 & 日志设置
    # ========================================================================
    args = parse_args()
    # exp_name = args.exp_name
    # train = args.train
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init

    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """
            这里的功能可以扩展到不同的算法。
            """
            sys.exit()
    except Exception as e:
        print(e.message)
        sys.exit()

    if train is True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))

    # 设置随机数种子以复现结果
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 在默认的非确定性模式下，多次执行相同的程序可能会因为内部并行计算的差异而产生略微不同的结果。
    torch.backends.cudnn.deterministic = args.torch_deterministic

    action_std_decay_rate = 0.05
    min_action_std = 0.05
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    # ========================================================================
    #                           创建仿真
    # ========================================================================

    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world, town)
    else:
        env = CarlaEnvironment(client, world, town, checkpoint_frequency=None)
    encode = EncodeState(LATENT_DIM)

    # 调整场景的视角到俯视（便于查看训练时车辆的完整运动）
    if env.map.name == "Town07":
        spectator = env.world.get_spectator()
        location = carla.Location(x=-66.9, y=-34.1, z=270)  # 通过选中虚幻编辑器中场景中间对象的坐标来获取（需要除以100，将厘米转成米）
        rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)  # 俯仰角-90表示俯视
        new_transform = carla.Transform(location, rotation)
        spectator.set_transform(new_transform)
        pass

    # ========================================================================
    #                           算法
    # ========================================================================
    try:
        # time.sleep(0.5)

        if checkpoint_load:
            # 获取最新的检查点文件编号
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
            # 构建检查点文件路径
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_' + str(chkt_file_nums) + '.pickle'
            # 加载检查点数据
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']  # 训练轮次
                timestep = data['timestep']  # 时间步数
                cumulative_score = data['cumulative_score']  # 累计得分
                action_std_init = data['action_std_init']  # 动作标准差初始值
            # 使用检查点中的参数初始化智能体
            agent = PPOAgent(town, action_std_init)
            # 加载模型参数
            agent.load()
        else:
            if train is False:
                # 初始化智能体
                agent = PPOAgent(town, action_std_init)
                agent.load()  # 加载预训练模型
                # 冻结actor网络参数（推理时不更新）
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
            else:
                # 全新训练 (默认情况)
                # 全新初始化智能体
                agent = PPOAgent(town, action_std_init)
        if train:
            # 执行训练, total_timesteps：预设的总训练步数。
            while timestep < total_timesteps:
                print(timestep)
                # 修改两个位置
                # 1. 观测数据修改
                observation = env.reset()  # 重置环境
                # observation = encode.process(observation)  # 编码观测数据（如图像预处理）,对原始观测（如图像）进行标准化/归一化处理。

                current_ep_reward = 0
                t1 = datetime.now()  # 记录回合开始时间
                # 回合内交互循环
                for t in range(args.episode_length):
                    # 选择带有策略的动作
                    action = agent.get_action(observation, train=True)  # 通过策略网络选择动作,使用当前策略（带探索噪声）选择动作。

                    observation, reward, done, info = env.step(action)  # 执行动作并返回新状态、奖励、终止标志和额外信息（如距离、偏离中心程度）。
                    if observation is None:  # 无效观测处理
                        break
                    # observation = encode.process(observation)
                    # 存储经验数据, 将(reward, done)存入PPO的回放缓冲区。
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    # 更新统计量
                    timestep += 1
                    current_ep_reward += reward
                    # 动态调整探索噪声
                    if timestep % action_std_decay_freq == 0:
                        action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep == total_timesteps - 1:
                        agent.chkpt_save()

                    # break; if the episode is over终止条件处理
                    if done:  # 回合提前终止（如碰撞、超时）
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2 - t1

                        episodic_length.append(abs(t3.total_seconds()))  # 记录回合耗时,（用于分析效率）。
                        break
                # 性能指标计算

                deviation_from_center += info[1]  # 累计偏离车道中心程度
                distance_covered += info[0]  # 累计行驶距离

                scores.append(current_ep_reward)  # 存储本轮总奖励

                # 计算滑动平均奖励
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep),
                      ', Reward:  {:.2f}'.format(current_ep_reward),
                      ', Average Reward:  {:.2f}'.format(cumulative_score))

                # 定期学习与检查点保存, 每10回合学习一次
                if episode % 10 == 0:
                    agent.learn()  # 从经验中更新策略
                    agent.chkpt_save()  # 快速保存检查点
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -= 1
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_' + str(chkt_file_nums) + '.pickle'
                    # 更新检查点元数据
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep,
                                'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)

                # 每5回合记录TensorBoard日志
                if episode % 5 == 0:
                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center / 5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered / 5, timestep)
                    # 重置统计量
                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                # 每100回合完整保存
                if episode % 100 == 0:
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_' + str(chkt_file_nums) + '.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep,
                                'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)

            print("Terminating the run.")
            sys.exit()
        else:
            # Testing
            while timestep < args.test_timesteps:
                observation = env.reset()
                # observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    # select action with policy
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    # observation = encode.process(observation)

                    timestep += 1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2 - t1

                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]

                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep),
                      ', Reward:  {:.2f}'.format(current_ep_reward),
                      ', Average Reward:  {:.2f}'.format(cumulative_score))

                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()

    finally:
        sys.exit()


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
