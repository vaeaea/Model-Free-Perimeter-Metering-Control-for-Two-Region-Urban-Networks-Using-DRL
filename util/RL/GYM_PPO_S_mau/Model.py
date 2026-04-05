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



# ==================== Actor网络 ====================
class Actor(nn.Module):
    """
    Actor 网络（策略网络）
    参考 DDPG 的简单全连接结构，输出确定性策略（均值）和对数标准差
    适用于两区域交通模型，动作维度为 2（u_12, u_21）
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_junctions: int = 2):
        """
        初始化 Actor 网络
        Args:
            state_dim: 状态空间维度（对于两区域模型为 9）
            hidden_dim: 隐藏层维度
            num_junctions: 动作维度（默认 2，对应 u_12 和 u_21）
        """
        super().__init__()
        self.num_junctions = num_junctions

        # 构建网络结构（参考 Keras 实现）
        # 输入层 -> 64 维隐藏层 -> 64 维隐藏层 -> 输出层
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 输出层：分别输出均值和对数标准差
        # 均值层：tanh 激活，范围 [-1, 1]，后续会线性变换到 [0.2, 0.8]
        self.mu_layer = nn.Linear(64, num_junctions)

        # 对数标准差参数（可学习参数）
        self.log_std_param = nn.Parameter(torch.zeros(1, num_junctions))
        self.min_log_std = -2.0
        self.max_log_std = 2.0

        # 初始化权重（使用 random_normal，对应 Keras 的 kernel_initializer='random_normal'）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            # Keras 的 random_normal 默认 mean=0, stddev=0.05
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        Args:
            x: 输入状态 (batch_size, state_dim)
        Returns:
            mean: 动作均值 (batch_size, num_junctions)
            std: 动作标准差 (batch_size, num_junctions)
        """
        # 共享隐藏层
        h = self.network(x)

        # 输出均值（tanh 激活，范围 [-1, 1]）
        mean = torch.tanh(self.mu_layer(h))

        # 输出对数标准差（裁剪到合理范围）
        log_std = torch.clamp(
            self.log_std_param.expand_as(mean),
            min=self.min_log_std,
            max=self.max_log_std
        )
        std = torch.exp(log_std)

        return mean, std


# class Actor(nn.Module):
#     """Actor网络，带自注意力机制的全局策略网络"""
#     def __init__(self, state_dim: int, hidden_dim: int, num_junctions: int = None):
#         super().__init__()
#         # print(state_dim, hidden_dim, num_junctions)
#         self.num_junctions = num_junctions
#         # 输入嵌入
#         self.mu_layer = nn.Sequential(
#             nn.Linear(state_dim, 4*hidden_dim),
#             nn.ReLU(),
#             nn.Linear(4*hidden_dim, 2*hidden_dim),
#             nn.ReLU(),
#             nn.Linear(2*hidden_dim, num_junctions)
#         )
#         self.log_std_param = nn.Parameter(torch.zeros(1, num_junctions))
#         self.min_log_std = -2
#         self.max_log_std = 2
#
#     def forward(self, x: torch.Tensor) -> tuple:
#         batch_size, _ = x.shape
#         mean = self.mu_layer(x)
#         log_std = torch.clamp(self.log_std_param.expand_as(mean), min=self.min_log_std, max=self.max_log_std)
#         std = torch.exp(log_std)
#         return mean, std

# ==================== Critic网络 ====================
class Critic(nn.Module):
    """
    Critic 网络（价值网络）
    输入状态，输出状态价值估计
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_junctions: int = 2):
        """
        初始化 Critic 网络
        Args:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
            num_junctions: 动作维度（仅用于兼容性，不影响网络结构）
        """
        super().__init__()
        self.num_junctions = num_junctions

        # 构建网络结构（与 Actor 类似的隐藏层结构）
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个价值值
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 输入状态 (batch_size, state_dim)
        Returns:
            value: 状态价值 (batch_size, 1)
        """
        return self.network(state)