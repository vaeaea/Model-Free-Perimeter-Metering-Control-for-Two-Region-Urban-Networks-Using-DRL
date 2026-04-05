
"""
RL 训练器，整合数据采集、训练、测试流程
适配标准 Gym 环境，统一入口
支持并行环境采样
"""
import torch
import numpy as np
import random
import os
import sys
from typing import Dict, List
import gymnasium as gym
from util.RL.GYM_DDPG_S_mau.TrafficSignalEnv import TrafficSignalEnv
from util.RL.GYM_DDPG_S_mau.DDPG import DDPGAgent
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_env(
        config: Dict,
        seed: int,
        idx: int,
):
    """
    创建单个环境的工厂函数（thunk）

    Args:
        config: 环境配置字典
        seed: 随机种子（每个环境不同）
        idx: 环境索引（用于分配不同端口）

    Returns:
        thunk: 延迟执行的函数，返回初始化好的环境
    """

    def thunk():
        # ========== 1. 创建基础环境 ==========
        # 为每个环境分配不同的端口，避免 SUMO 冲突
        env = TrafficSignalEnv(config=config)

        # ========== 2. 应用基础包装器 ==========
        # ClipAction: 限制动作空间范围（防止数值溢出）
        # print(f"ClipAction: 截断动作空间范围（防止数值溢出）")
        env = gym.wrappers.ClipAction(env)

        # ========== 3. 记录 Episode 统计信息 ==========
        # print(f"记录 Episode 统计信息")

        # ========== 4. 观测值裁剪（防止异常值） ==========
        # env = NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(
        #     env,
        #     lambda obs: np.clip(obs, -10., 10.).astype(np.float32)
        # )
        #
        # # ========== 5. 奖励裁剪（防止梯度爆炸） ==========
        # env = NormalizeReward(env)
        # env = gym.wrappers.TransformReward(
        #     env,
        #     lambda reward: np.clip(reward, -10., 10.).astype(np.float32)
        # )
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # ========== 6. 设置随机种子 ==========
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)

        return env

    return thunk

class RLTrainer:
    """
    RL训练器，统一管理训练、数据收集、测试流程
    """
    def __init__(self, config: Dict):
        self.config = config

        # Windows 系统需要特殊处理
        if sys.platform == 'win32':
            print("⚠️  检测到 Windows 系统，将使用 spawn 启动方式")

        print("初始化训练器...")
        # 初始化环境（用于获取维度信息）
        temp_env = TrafficSignalEnv(config)
        self.num_junctions = 2
        self.state_dim = temp_env.observation_space.shape[0]
        self.action_dim = temp_env.action_space.shape[0]
        temp_env.close()
        print(f"路口数量：{self.num_junctions}")
        print(f"状态维度：{self.state_dim} | 动作维度：{self.action_dim}")
        print(f"{'='*70}\n")

        # 初始化智能体
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            mode='train',
            num_junctions=self.num_junctions
        )

        num_envs = self.config.get('num_workers', 4)
        base_seed = self.config.get('seed', 42)
        print(f"\n创建 {num_envs} 个并行环境...")
        self.envs = self._create_parallel_envs(num_envs, base_seed)
        self.test_env = self._get_test_env()
        print("✅ 初始化完成！")
        print(f"{'='*70}\n")

    def _create_parallel_envs(self, num_envs: int, seed: int):
        """
        创建并行环境集合

        Args:
            num_envs: 并行环境数量
            seed: 基础随机种子

        Returns:
            envs: AsyncVectorEnv 实例
        """
        # 创建环境工厂列表
        env_fns = [
            make_env(
                config=self.config,
                seed=seed + i,
                idx=i,
            )
            for i in range(num_envs)
        ]

        # 根据操作系统选择上下文
        context = "spawn" if sys.platform == 'win32' else "fork"
        envs = gym.vector.AsyncVectorEnv(env_fns, context=context)
        return envs

    def train_policy(self, iteration: int):
        """
        一体化训练方法：数据采集 + 模型更新
        Args:
            iteration: 当前迭代次数
        Returns:
            stats: 训练统计信息
        """
        print(f"🎯 迭代 {iteration} / {self.config.get('max_iterations', 100)}")

        num_envs = self.config.get('num_workers', 4)
        num_steps = self.config.get('simulation_steps', 360)

        # ========== 阶段 1：数据采集 ==========
        print(f"\n📊 阶段 1: 数据采集 ({num_envs}个环境 × {num_steps}步)")

        # 重置环境（获取初始观测/info
        obs, infos = self.envs.reset()

        # 预分配存储空间（用于统计，不存储完整数据）
        episode_rewards = []
        episode_lengths = []
        episode_times = []
        raw_rewards = []
        tts = []
        M11_LIST = []
        M22_LIST = []
        total_samples = 0

        old_buffer_len = len(self.agent.replay_buffer)
        self.agent.actor.eval()
        self.agent.critic.eval()
        # 采集数据并按时间步批量存储
        with torch.no_grad():
            for step in range(num_steps):
                # 选择动作（批量）
                action = self.agent.select_actions(obs, deterministic=False)
                # 执行动作（并行）
                next_obs, reward, terminateds, truncateds, infos = self.envs.step(action)
                dones = terminateds | truncateds

                # ========== 按时间步批量 push 到 buffer（核心改动）==========
                # 一次性存入整个时间步的所有环境数据
                self.agent.replay_buffer.batch_push(
                    states=obs,  # shape: (num_envs, state_dim)
                    actions=action,  # shape: (num_envs, action_dim)
                    rewards=reward,  # shape: (num_envs,)
                    next_states=next_obs,  # shape: (num_envs, state_dim)
                    dones=dones,  # shape: (num_envs,)
                )

                total_samples += num_envs  # 每个时间步有 num_envs 条数据

                # 处理 episode 结束`
                if any(dones):
                    print(f"⚠️  某一环境的 Episode 在{step}步结束！")
                    for i, (term, trunc) in enumerate(zip(terminateds, truncateds)):
                        if term or trunc:
                            if 'final_info' in infos and infos['final_info'][i]['episode'] is not None:
                                ep_info = infos['final_info'][i]['episode']
                                episode_rewards.append(ep_info['r'])
                                episode_lengths.append(ep_info['l'])
                                episode_times.append(ep_info['t'])
                            if 'final_info' in infos and infos['final_info'][i]['raw_rewards'] is not None:
                                raw_rewards.append(infos['final_info'][i]['raw_rewards'])
                            if 'final_info' in infos and infos['final_info'][i]['tts'] is not None:
                                tts.append(infos['final_info'][i]['tts'])
                            if 'final_info' in infos and infos['final_info'][i]['M11_LIST'] is not None:
                                M11_LIST.append(infos['final_info'][i]['M11_LIST'])
                                M22_LIST.append(infos['final_info'][i]['M22_LIST'])

                # 更新观测
                obs = next_obs
                # 进度显示
                if (step + 1) % 10 == 0:
                    print(f"   采集进度：{step + 1}/{num_steps} 步 | 已存储 {total_samples} 条样本")

        # ========== 数据采集完成 ==========
        print(f"\n📦 数据采集完成...")
        print(f"   总样本数：{total_samples}条")
        print(f"   Buffer 大小由 {old_buffer_len} 步变为 {len(self.agent.replay_buffer)} 步")

        if episode_rewards:
            print(f"   平均 Episode 奖励：{np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"   平均 Raw Rewards：{np.mean(raw_rewards):.2f} ± {np.std(raw_rewards):.2f}")
            print(f"   平均 TTS：{(np.mean(tts)/3600):.2f} ± {(np.std(tts)/3600):.2f}")
            print(f"   平均 CTC：{(np.sum(M11_LIST)+np.sum(M22_LIST))/num_envs:.2f}")
            print(f"   平均 Episode 长度：{np.mean(episode_lengths):.2f}")
            print(f"   平均 Episode 耗时：{np.mean(episode_times):.2f} ± {np.std(episode_times):.2f}")

        # ========== 阶段 2：训练更新 ==========
        print(f"\n🔄 阶段 2: 模型训练...")
        self.agent.actor.train()
        self.agent.critic.train()

        # 执行训练（直接从 buffer 读取）
        train_stats = self.agent.update()

        print(f"   ✅ 训练完成")
        print(f"   Actor Loss: {train_stats.get('actor_loss', 0):.4f}")
        print(f"   Critic Loss: {train_stats.get('critic_loss', 0):.4f}")

        # ========== 阶段 3：清理与保存 ==========
        # 清空 buffer（on-policy 算法）
        self.agent.noise_decay()

        # 保存模型
        if self.config.get('save_model', True):
            model_path = f"{self.config['pretrained_path_prefix']}_iter{iteration}.pth"
            self.agent.save_model(model_path)
            print(f"   💾 模型已保存至：{model_path}")

        # 合并统计信息
        stats = {
            "iteration": iteration,
            "samples_collected": total_samples,
            "episode_rewards": episode_rewards,
            "raw_rewards": raw_rewards,
            "tts": tts,
            "M11_LIST": M11_LIST,
            "M22_LIST": M22_LIST,
            "episode_lengths": episode_lengths,
            "episode_times": episode_times,
            "train_stats": train_stats
        }

        print(f"\n✅ 迭代 {iteration} 完成！")
        print(f"{'=' * 70}\n")

        return stats

    def _get_test_env(self):
        """创建/获取带归一化包装器的测试环境（用于norm参数加载）"""
        port = self.config.get('base_port', 8877) + 999  # 独立端口避免冲突
        env = TrafficSignalEnv(config=self.config, )
        # 应用与训练环境完全一致的包装器
        env = gym.wrappers.ClipAction(env)
        # env = NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(
        #     env,
        #     lambda obs: np.clip(obs, -10., 10.).astype(np.float32)
        # )
        # env = NormalizeReward(env)
        # env = gym.wrappers.TransformReward(
        #     env,
        #     lambda reward: np.clip(reward, -10., 10.).astype(np.float32)
        # )
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(self.config.get('seed', 42))
        env.observation_space.seed(self.config.get('seed', 42))
        return env

    def test(self, render: bool = False):
        """
        测试智能体
        Args:
            render: 是否渲染 GUI
        Returns:
            rewards: 奖励历史
            actions: 动作历史
        """
        print(f"\n🧪 开始测试...")
        # 创建环境

        obs, _ = self.test_env.reset(seed=self.config['seed'])
        obs = obs.reshape(1, -1)

        step = 0
        rewards = []
        all_actions = []  # 存储所有时间步的动作
        tts = []
        M11_LIST = []
        M22_LIST = []
        self.agent.actor.eval()
        self.agent.critic.eval()
        while step < self.config['simulation_steps']:
            # 选择动作
            action = self.agent.select_actions(obs, deterministic=True)
            # 环境交互
            next_obs, reward, terminated, truncated, info = self.test_env.step(action[0])
            dones = terminated | truncated
            # 记录
            rewards.append(reward)
            all_actions.append(action[0])
            # 更新
            obs = next_obs.reshape(1, -1)
            step += 1
            print(f"   Step {info['step_count']} | Reward: {reward:.2f}")

            if terminated or truncated:
                if info['tts'] is not None:
                    tts.append(info['tts'])
                    M11_LIST.append(info['M11_LIST'])
                    M22_LIST.append(info['M22_LIST'])
        self.test_env.close()
        # ========== 新增：统计分析 ==========
        # print(f"\n📊 观测归一化参数统计:")
        # print(f"{'=' * 70}")
        # norm_obs_wrapper = self.test_env
        # while not isinstance(norm_obs_wrapper, NormalizeObservation):
        #     norm_obs_wrapper = norm_obs_wrapper.env
        #     if not hasattr(norm_obs_wrapper, 'env'):
        #         break
        #
        # obs_mean = norm_obs_wrapper.obs_rms.mean
        # obs_var = norm_obs_wrapper.obs_rms.var
        # obs_count = norm_obs_wrapper.obs_rms.count
        # print(f"   观测均值：{obs_mean}")
        # print(f"   观测方差：{obs_var}")
        # print(f"   观测样本数：{obs_count}")

        print(f"\n✅ 测试完成 | 平均奖励：{np.mean(rewards):.2f}")
        print(f"{'='*70}")

        # 计算总奖励
        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        print(f"\n📊 奖励统计:")
        print(f"   总奖励：{total_reward:.2f}")
        print(f"   平均奖励：{avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   平均TTS：{(np.mean(tts)/3600):.2f}")
        print(f"   CTC：{np.sum(M11_LIST)+np.sum(M22_LIST):.2f}")
        print(f"   最大单步奖励：{np.max(rewards):.2f}")
        print(f"   最小单步奖励：{np.min(rewards):.2f}")
        print(f"   总步数：{len(rewards)}")

        # 分析各 junction 的动作统计
        all_actions_array = np.array(all_actions)  # shape: (steps, num_junctions)
        num_junctions = all_actions_array.shape[1]

        print(f"\n🚦 各路口动作统计 (共 {num_junctions} 个路口):")
        print(f"{'='*70}")
        print(f"{'路口 ID':<10s} | {'最小值':<10s} | {'最大值':<10s} | "
              f"{'平均值':<10s} | {'标准差':<10s}")
        print(f"{'-'*70}")

        for junction_idx in range(num_junctions):
            junction_actions = all_actions_array[:, junction_idx]
            min_val = np.min(junction_actions)
            max_val = np.max(junction_actions)
            mean_val = np.mean(junction_actions)
            std_val = np.std(junction_actions)

            print(f"{junction_idx:<10d} | {min_val:<10.4f} | {max_val:<10.4f} | "
                  f"{mean_val:<10.4f} | {std_val:<10.4f}")

        print(f"{'='*70}")

        # 全局统计
        all_actions_flat = all_actions_array.flatten()
        print(f"\n🌐 全局动作统计:")
        print(f"   所有路口动作范围：[{np.min(all_actions_flat):.4f}, {np.max(all_actions_flat):.4f}]")
        print(f"   所有路口平均动作：{np.mean(all_actions_flat):.4f} ± {np.std(all_actions_flat):.4f}")
        print(f"   动作空间理论范围：[0.1, 0.9]")

        # 检查是否有超出边界的情况
        out_of_bounds_ratio = ((all_actions_flat < 0.1) | (all_actions_flat > 0.9)).sum() / len(all_actions_flat) * 100
        if out_of_bounds_ratio > 0:
            print(f"   ⚠️  超出边界比例：{out_of_bounds_ratio:.2f}%")
        else:
            print(f"   ✅ 所有动作均在合法范围内")

        print(f"{'='*70}\n")
        return rewards, all_actions, tts, np.sum(M11_LIST)+np.sum(M22_LIST)

    def save_model(self, iteration: int):
        """保存模型"""
        model_path = f"{self.config['pretrained_path_prefix']}_iter{iteration}.pth"
        self.agent.save_model(model_path)
        print(f"💾 模型已保存至：{model_path}")

    def load_model(self, iteration: int):
        """加载模型"""
        model_path = f"{self.config['pretrained_path_prefix']}_iter{iteration}.pth"
        self.agent.load_model(model_path)

    def close(self):
        """关闭训练器，释放资源"""
        print("\n🔒 关闭训练器，释放资源...")
        if hasattr(self, 'envs') and self.envs is not None:
            self.envs.close()
            print("   ✅ 并行环境已关闭")
        print("   ✅ 训练器已完全关闭\n")

