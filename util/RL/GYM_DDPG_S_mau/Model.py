import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def initialize_weights(mod, initialization_type, scale=np.sqrt(2)):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                torch.nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_head: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
        query, key, value = self._split(query), self._split(key), self._split(value)
        attn_score, attn = self.attention(query, key, value)
        attn = self._concat(attn)
        out = self.w_o(attn)
        return out

    def _concat(self, tensor: torch.Tensor):
        # input shape: (batch size, num_head, length, d_block)
        tensor = tensor.transpose(1, 2)  # (batch size, length, num_head, d_block)
        tensor = tensor.reshape(tensor.size()[0], tensor.size()[1], self.model_dim)  # (batch size, length, d_model)
        return tensor

    def _split(self, tensor: torch.Tensor):
        batch_size, length, d_model = tensor.size()
        d_block = d_model // self.num_head
        # (batch size, length, d_model) -> (batch size, length, num_head, d_block)
        tensor = tensor.view(batch_size, length, self.num_head, d_block)
        # (batch size, num_head, length, d_block)
        tensor.transpose_(1, 2)
        return tensor

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # scale dot product attention
        d_block = query.size()[-1]
        key_t = key.transpose(2, 3)
        # attention score = softmax(q * k^T / sqrt(d_block)). shape: (batch size, num_head, length, length)
        attn_score = torch.softmax(torch.matmul(query, key_t) / math.sqrt(d_block), dim=-1)
        attn = torch.matmul(attn_score, value)  # (batch size, num_head, length, d_block)
        return attn_score, attn


class Actor(nn.Module):
    def __init__(self, state_dim: int,  hidden_dim: int = 64, num_junctions: int = None):
        """
        初始化 Actor 网络

        参数:
            state_dim: 输入状态的维度
            action_dim: 动作空间维度 (默认为 2)
            hidden_dim: 隐藏层神经元数量 (默认为 64)
            lr: 学习率
        """
        super().__init__()

        self.state_dim = state_dim

        # 构建网络层
        self.dense1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 输出层 - 为每个动作维度创建独立的输出
        self.action_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 1),
                nn.Tanh(),
                ScaleLayer(scale=0.4, shift=0.5)  # 映射到 [0.1, 0.9]
            ) for _ in range(num_junctions)
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入状态张量，形状为 (batch_size, state_dim)

        返回:
            动作张量，形状为 (batch_size, action_dim)
        """
        # 第一隐藏层
        h = self.dense1(x)
        # 第二隐藏层
        h = self.dense2(h)

        # 为每个动作维度计算输出
        actions = []
        for action_net in self.action_outputs:
            action = action_net(h)
            actions.append(action)

        # 拼接所有动作维度
        out = torch.cat(actions, dim=-1)
        return out


class ScaleLayer(nn.Module):
    """自定义缩放层，实现 y = x * scale + shift"""

    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


# ==================== Critic网络 ====================


class Critic(nn.Module):
    """Critic 网络，评估状态 - 动作对的价值"""

    def __init__(self, state_dim: int, hidden_dim: int, num_junctions: int = None):
        """
        初始化 Critic 网络

        参数:
            state_dim: 状态空间的维度
            action_dim: 动作空间的维度
            hidden_dim: 隐藏层神经元数量 (默认为 64)
            lr: 学习率
        """
        super().__init__()

        self.state_dim = state_dim

        # 状态 - 动作拼接后的全连接层
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + num_junctions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 Q 值
        )


    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            state: 状态张量，形状为 (batch_size, state_dim)
            action: 动作张量，形状为 (batch_size, action_dim)

        返回:
            Q 值张量，形状为 (batch_size, 1)
        """
        # 拼接状态和动作
        state_action = torch.cat([state, action], dim=-1)

        # 通过 Q 网络
        q_values = self.q_network(state_action)
        return q_values