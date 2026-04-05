
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
from util.RL.GYM_PPO_S_mau.Model import Actor, Critic
from util.RL.GYM_PPO_S_mau.Buffer import RolloutBuffer

# ==================== PPO智能体 ====================
class PPOAgent:
    """
    PPO智能体，适配标准Gym环境
    支持集中式多交叉口控制，带自注意力机制
    """
    def __init__(self, state_dim: int, config: dict, mode: str, worker_id: int = 0,
                 num_junctions: int = 1, num_iterations: int = 0):
        self.config = config
        self.mode = mode
        self.worker_id = worker_id
        self.num_junctions = num_junctions
        self.device = torch.device(config["device"])

        self.state_dim = state_dim

        # 网络初始化
        self.actor = Actor(state_dim, config["hidden_dim"], num_junctions).to(self.device)
        self.critic = Critic(state_dim, config["hidden_dim"], num_junctions).to(self.device)

        # 优化器
        # self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config["lr_actor"], eps=1e-5)
        self.policy_optim = torch.optim.Adam(self.actor.parameters(), lr=config["lr_actor"], eps=1e-5)
        self.value_optim = torch.optim.Adam(self.critic.parameters(), lr=config["lr_critic"], eps=1e-5)

        self.policy_scheduler = ExponentialLR(self.policy_optim, gamma=config.get("lr_decay", 0.99))
        self.value_scheduler = ExponentialLR(self.value_optim, gamma=config.get("lr_decay", 0.99))
        # self.optim_scheduler = ExponentialLR(self.optim, gamma=config.get("lr_decay", 0.99))
        # 学习率下限配置
        self.min_lr_actor = config.get("min_lr_actor", 1e-5)
        self.min_lr_critic = config.get("min_lr_critic", 1e-5)

        # 轨迹缓冲区，使用预分配数组的RolloutBuffer
        num_env = self.config.get("num_workers", 1) if mode == 'train' else 1
        self.trajectory_buffer = RolloutBuffer(
            num_env=num_env,
            steps=self.config.get("max_trajectory_steps", 10000),
            state_shape=(self.state_dim,),
            action_shape=(self.num_junctions,),
            device=self.device
        )

        # PPO参数
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lam", 0.95)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 10)
        self.batch_size = config.get("batch_size", 64)

        self.vf_clip_coef = config.get("vf_clip_coef", 0.5)
        self.kl_threshold = config.get("kl_threshold", 0.07)

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
        mean, std = self.actor(state_tensor)
        if deterministic:
            action = mean
            log_prob = torch.tensor([0.0]).reshape(1, -1)
        else:
            # 采样动作
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        # 转换为numpy
        action = torch.clip(action, -1, 1)
        action_np = action.cpu().detach().numpy()
        action_np = action_np * 0.4 + 0.5
        # log_prob_np = log_prob.cpu().detach().item()
        log_prob_np = log_prob.cpu().detach().numpy().flatten()

        return action_np, log_prob_np


    def update(self):
        """PPO 更新，执行多个 epoch 的训练"""
        if len(self.trajectory_buffer) == 0:
            return {}

        # 拉取所有数据，一次性转换为 Tensor
        states, actions, old_log_probs, rewards, next_states, dones = self.trajectory_buffer.pull()

        # 转置维度：(steps, num_env, ...) -> (num_env, steps, ...) todo shape是否正确
        states = states.transpose(0, 1)
        next_states = next_states.transpose(0, 1)
        rewards = rewards.transpose(0, 1)
        dones = dones.transpose(0, 1)
        actions = actions.transpose(0, 1)
        old_log_probs = old_log_probs.transpose(0, 1)


        num_env = states.size(0)
        steps = states.size(1)

        with torch.no_grad():
            # 批量计算 values 和 next_values（替代循环，提升效率）
            # reshape: (num_env, steps, state_dim) -> (num_env*steps, state_dim)
            all_states = states.reshape(-1, self.state_dim)
            all_next_states = next_states.reshape(-1, self.state_dim)

            # 批量前向传播
            all_values = self.critic(all_states)  # (num_env*steps, 1)
            all_next_values = self.critic(all_next_states)  # (num_env*steps, 1)
            # print("all_states:", all_states[5:10, :], "\n all_next_states:", all_next_states[5:10, :])
            # print("all_values:", all_values[5:10, :], "\n all_next_values:", all_next_values[5:10, :])
            # reshape 回 (num_env, steps, 1)
            values = all_values.view(num_env, steps)
            next_values = all_next_values.view(num_env, steps)

            # 计算 GAE 优势函数
            advantages = torch.zeros_like(values).to(self.device)
            gae = 0
            for t in reversed(range(steps)):
                delta = rewards[:, t] + self.gamma * (1 - dones[:, t]) * next_values[:, t] - values[:, t]
                gae = delta + self.gamma * self.lam * gae * (1 - dones[:, t])
                advantages[:, t] = gae
            # 计算 returns
            returns = advantages + values
            # 归一化优势函数
            norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转回原始维度并展平
        states = states.transpose(0, 1).contiguous().view(-1, self.state_dim)
        actions = actions.transpose(0, 1).contiguous().view(-1, self.num_junctions)
        old_log_probs = old_log_probs.transpose(0, 1).contiguous().view(-1, 1)
        returns = returns.transpose(0, 1).contiguous().view(-1, 1)
        norm_adv = norm_adv.transpose(0, 1).contiguous().view(-1, 1)

        # ========== 新增：保存旧的 value 用于 clipping ==========
        with torch.no_grad():
            old_values = self.critic(states).detach()

        # 统计
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        clip_ratios = []
        kl_divs = []

        # 多 epoch 训练
        n_samples = len(states)
        print(self.actor.log_std_param)
        for epoch in range(self.ppo_epochs):
            # 随机打乱样本
            indices = torch.randperm(n_samples)
            epoch_kl_divs = []
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]

                # 取 batch 数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = norm_adv[batch_idx]

                # 前向传播
                mean, std = self.actor(batch_states)

                normal = torch.distributions.Normal(mean, std)
                new_log_prob = normal.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                values = self.critic(batch_states)
                entropy = normal.entropy().sum(dim=-1, keepdim=True)
                ratio = torch.exp(new_log_prob - batch_old_log_probs)

                with torch.no_grad():
                    approx_kl_div = batch_old_log_probs - new_log_prob
                    kl_value = approx_kl_div.mean().item()
                    epoch_kl_divs.append(kl_value)

                # PPO 损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                policy_loss = actor_loss + self.config['entropy_coef'] * entropy_loss

                # Critic 损失
                # critic_loss = 0.5 * nn.MSELoss()(values, batch_returns)
                # 获取 batch 对应的 old_values
                batch_old_values = old_values[batch_idx]
                clip_v = batch_old_values + torch.clamp(values - batch_old_values, -self.vf_clip_coef, self.vf_clip_coef)
                unclipped_v_loss = (values - batch_returns) ** 2
                clipped_v_loss = (clip_v - batch_returns) ** 2
                v_max = 0.5 * torch.max(unclipped_v_loss, clipped_v_loss)
                critic_loss = v_max.mean()

                # 反向传播 - Actor
                self.policy_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['grad_clip'])
                self.policy_optim.step()

                # 反向传播 - Critic
                self.value_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['grad_clip'])
                self.value_optim.step()

                # 统计
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_ratios.append((ratio > 1 + self.clip_coef).float().mean().item()) # 衡量有多少比例的动作被 PPO 的裁剪机制"截断"

            # ========== 新增：检查 epoch 平均 KL 散度 ==========
            avg_epoch_kl = np.mean(epoch_kl_divs)
            kl_divs.append(avg_epoch_kl)

            print(f"   Epoch {epoch+1}/{self.ppo_epochs} | KL: {avg_epoch_kl:.4f} | Target: {self.kl_threshold}")

            # KL 散度早停判断
            if avg_epoch_kl > self.kl_threshold:
                print(f"   ⚠️  KL 散度 ({avg_epoch_kl:.4f}) 超过阈值 ({self.kl_threshold})，提前终止本次更新")
                break
            # 学习率衰减
            self.policy_scheduler.step()
            self.value_scheduler.step()
            # self.optim.step()
            # 限制学习率下限
            current_policy_lr = self.policy_optim.param_groups[0]['lr']
            current_value_lr = self.value_optim.param_groups[0]['lr']

            if current_policy_lr < self.min_lr_actor:
                for param_group in self.policy_optim.param_groups:
                    param_group['lr'] = self.min_lr_actor

            if current_value_lr < self.min_lr_critic:
                for param_group in self.value_optim.param_groups:
                    param_group['lr'] = self.min_lr_critic


        # 清空缓冲区
        self.trajectory_buffer.clear()

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": -np.mean(entropy_losses),
            "clip_ratio": np.mean(clip_ratios)
        }

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'value_optim_state_dict': self.value_optim.state_dict(),
            # 'optim_state_dict': self.optim.state_dict(),
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.value_optim.load_state_dict(checkpoint['value_optim_state_dict'])
        # self.optim.load_state_dict(checkpoint['optim_state_dict'])
        print(f"Loaded model from {path}")