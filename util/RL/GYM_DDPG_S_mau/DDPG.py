
"""
PPO算法实现，适配标准Gym环境
重构自原有PPO_global代码，统一接口
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import os
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from util.RL.GYM_DDPG_S_mau.Model import Actor, Critic
from util.RL.GYM_DDPG_S_mau.Buffer import ReplayBuffer

# ==================== DDPG智能体 ====================
class DDPGAgent:
    """
    DDPG智能体，适配标准Gym环境
    支持集中式多交叉口控制，带自注意力机制
    """
    def __init__(self, state_dim: int, action_dim: int, config: dict, mode: str, worker_id: int = 0,
                 num_junctions: int = 1, num_iterations: int = 0):
        self.config = config
        self.mode = mode
        self.worker_id = worker_id
        self.num_junctions = num_junctions
        self.device = torch.device(config["device"])

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 网络初始化
        self.actor = Actor(state_dim, config["hidden_dim"], num_junctions).to(self.device)
        self.actor_target = Actor(state_dim, config["hidden_dim"], num_junctions).to(self.device)
        self.critic = Critic(state_dim, config["hidden_dim"], num_junctions).to(self.device)
        self.critic_target = Critic(state_dim, config["hidden_dim"], num_junctions).to(self.device)

        # 同步目标网络初始参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        # self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config["lr_actor"], eps=1e-5)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config["lr_actor"], eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config["lr_critic"], eps=1e-5)

        self.actor_scheduler = ExponentialLR(self.actor_optim, gamma=config.get("lr_decay_actor", 0.99))
        self.critic_scheduler = ExponentialLR(self.critic_optim, gamma=config.get("lr_decay_critic", 0.99))
        # self.optim_scheduler = ExponentialLR(self.optim, gamma=config.get("lr_decay", 0.99))

        # 轨迹缓冲区，使用预分配数组的RolloutBuffer
        num_env = self.config.get("num_workers", 1) if mode == 'train' else 1
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.get("max_trajectory_steps", 10000),
            device=self.device
        )

        # PPO参数
        self.gamma = config.get("gamma", 0.99)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.DDPG_epochs = config.get("DDPG_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.noise  = self.config['noise_base']
        self.tau = config.get("tau", 0.005)

    def select_actions(self, state, deterministic: bool = False) -> tuple:
        """
        选择动作，标准Gym接口
        Args:
            state: 环境返回的观测
            deterministic: 是否确定性动作
        Returns:
            actions: 动作数组
            log_prob: 对数概率
        """
        # 重塑为 ( state_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config['device'])
        # 前向传播
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()

        if not deterministic:
            action += np.random.uniform(-self.noise, self.noise, size=action.shape)

        return action

    def noise_decay(self):
        """
        Noise decay
        """
        self.noise *= self.config['noise_decay']


    def update(self):
        """DDPG 更新，执行多个 epoch 的训练"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # 拉取所有数据，一次性转换为 Tensor todo 确认shape
        states, actions, rewards, next_states, dones = self.replay_buffer.pull(self.config['sample_size'])
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        actor_losses = []
        critic_losses = []

        # 进行batch划分
        n_samples = len(states)
        for epoch in range(self.DDPG_epochs):
            indices = torch.randperm(n_samples)
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_target_q = target_q[batch_idx]

                batch_current_q = self.critic(batch_states, batch_actions)
                batch_critic_loss = nn.MSELoss()(batch_current_q, batch_target_q)
                # 优化 Critic
                self.critic_optim.zero_grad()
                batch_critic_loss.backward()
                self.critic_optim.step()
                self.soft_update(self.critic, self.critic_target)
                critic_losses.append(batch_critic_loss.item())

                # 降低actor的更新频率
                if epoch % self.config['actor_update_freq'] == 0:
                    batch_predicted_actions = self.actor(batch_states)
                    batch_actor_loss = -self.critic(batch_states, batch_predicted_actions).mean()
                    # 优化 Actor
                    self.actor_optim.zero_grad()
                    batch_actor_loss.backward()
                    self.actor_optim.step()
                    self.soft_update(self.actor, self.actor_target)
                    actor_losses.append(batch_actor_loss.item())

            # 转回原始维度并展平
            # 学习率衰减
            self.critic_scheduler.step()
            self.actor_scheduler.step()

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
        }

    def soft_update(self, source, target):
        """软更新: target = tau * source + (1 - tau) * target"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        # self.optim.load_state_dict(checkpoint['optim_state_dict'])
        print(f"Loaded model from {path}")