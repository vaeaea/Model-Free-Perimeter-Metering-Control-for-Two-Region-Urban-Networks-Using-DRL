import torch
import numpy as np
from typing import Tuple, List, Deque
from collections import deque
from dataclasses import dataclass

@dataclass
class Experience:
    """单条经验的数据结构"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float

# ==================== 轨迹缓冲区（模仿预分配数组写法重构） ====================
class ReplayBuffer(object):
    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device

        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存入一条经验到队列末尾
        当队列满时，自动丢弃最旧的经验（deque 的 maxlen 机制）

        Args:
            state: 当前状态 (numpy array)
            action: 动作 (numpy array)
            reward: 奖励 (float)
            next_state: 下一状态 (numpy array)
            done: 是否终止 (float: 0 或 1)
        """
        experience = Experience(
            state=state.copy(),  # 复制避免外部修改
            action=action.copy(),
            reward=float(reward),
            next_state=next_state.copy(),
            done=float(done)
        )
        self.buffer.append(experience)

    def batch_push(self, states, actions, rewards, next_states, dones):
        """
        批量存入多条经验到队列末尾

        Args:
            states: 状态数组，形状 (batch_size, state_dim)
            actions: 动作数组，形状 (batch_size, num_junctions)
            rewards: 奖励数组，形状 (batch_size,)
            next_states: 下一状态数组，形状 (batch_size, state_dim)
            dones: 终止标志数组，形状 (batch_size,)

        Note:
            - 按顺序将每条经验加入队列
            - 当队列满时，超出的旧经验会被自动丢弃
            - 所有输入数组的第一维大小必须一致
            - 如果输入是 Tensor，会自动转换为 numpy 数组
        """
        # 获取 batch_size
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(next_states, torch.Tensor):
            next_states = next_states.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()

        batch_size = len(states)

        # 验证所有数组的 batch_size 一致
        assert len(actions) == batch_size, f"actions length {len(actions)} != states length {batch_size}"
        assert len(rewards) == batch_size, f"rewards length {len(rewards)} != states length {batch_size}"
        assert len(next_states) == batch_size, f"next_states length {len(next_states)} != states length {batch_size}"
        assert len(dones) == batch_size, f"dones length {len(dones)} != states length {batch_size}"

        # 批量创建 Experience 对象
        for i in range(batch_size):
            experience = Experience(
                state=states[i].copy(),
                action=actions[i].copy(),
                reward=float(rewards[i]),
                next_state=next_states[i].copy(),
                done=float(dones[i])
            )
            self.buffer.append(experience)

    def pull(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        从缓冲区随机抽取固定数量的样本，并转换为 tensor

        Args:
            batch_size: 需要抽取的样本数量

        Returns:
            states: 状态张量，形状 (batch_size, state_dim)
            actions: 动作张量，形状 (batch_size, num_junctions)
            rewards: 奖励张量，形状 (batch_size,)
            next_states: 下一状态张量，形状 (batch_size, state_dim)
            dones: 终止标志张量，形状 (batch_size,)

        Note:
            - 使用无放回抽样（除非当前样本数不足 batch_size）
            - 所有返回的 tensor 都已移动到指定设备（device）
            - 如果当前样本数为 0，抛出异常
        """
        current_size = len(self.buffer)

        if current_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")

        # 如果请求的 batch_size 大于当前数据量，调整为当前数据量
        actual_batch_size = min(batch_size, current_size)

        # 随机生成索引，始终使用无放回抽样
        indices = np.random.choice(current_size, actual_batch_size, replace=False)

        # 将 deque 转换为 list 以支持随机访问
        buffer_list = list(self.buffer)

        # 按索引提取经验
        sampled_experiences = [buffer_list[i] for i in indices]

        # 批量提取各字段并转换为 tensor
        states = torch.tensor(
            np.array([exp.state for exp in sampled_experiences]),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            np.array([exp.action for exp in sampled_experiences]),
            dtype=torch.float32,
            device=self.device
        )

        rewards = torch.tensor(
            np.array([exp.reward for exp in sampled_experiences]),
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            np.array([exp.next_state for exp in sampled_experiences]),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            np.array([exp.done for exp in sampled_experiences]),
            dtype=torch.float32,
            device=self.device
        )

        return states, actions, rewards, next_states, dones

    @property
    def size(self) -> int:
        """当前已存储的经验数量"""
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        """判断缓冲区是否已满"""
        return len(self.buffer) >= self.capacity

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

    def __len__(self) -> int:
        """当前已存储的经验数量（支持 len(buffer) 调用）"""
        return len(self.buffer)