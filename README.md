# 双区域城市网络周长控制 C-RL 智能体

本代码库是对论文 **"Model-Free Perimeter Metering Control for Two-Region Urban Networks Using Deep Reinforcement Learning"** 的复现实现。

## 项目概述

该项目使用深度强化学习算法对双区域城市网络的周长流量控制进行建模和优化。实现了两种强化学习算法：
- **DDPG (Deep Deterministic Policy Gradient)**
- **PPO (Proximal Policy Optimization)**

## 目录结构

```
双区域RL/
├── Train_model_DDPG_MAU.py      # DDPG 算法训练入口
├── Train_model_PPO_MAU.py        # PPO 算法训练入口
└── util/
    └── RL/
        ├── GYM_DDPG_S_mau/       # DDPG 算法实现
        │   ├── DDPG.py            # DDPG 智能体
        │   ├── Model.py           # 神经网络模型
        │   ├── Buffer.py          # 经验回放缓冲区
        │   ├── ENV.py             # 环境封装
        │   ├── RLTrainer.py       # 训练器
        │   └── TrafficSignalEnv.py # 交通信号环境
        └── GYM_PPO_S_mau/        # PPO 算法实现
            ├── PPO.py             # PPO 智能体
            ├── Model.py           # 神经网络模型
            ├── Buffer.py          # 轨迹缓冲区
            ├── ENV.py             # 环境封装
            ├── RLTrainer.py       # 训练器
            └── TrafficSignalEnv.py # 交通信号环境
```

## 环境要求

- Python 3.8+
- PyTorch
- NumPy

## 使用方法

### 训练 DDPG 模型

```bash
python Train_model_DDPG_MAU.py
```

### 训练 PPO 模型

```bash
python Train_model_PPO_MAU.py
```

## 配置说明

主要配置参数位于训练脚本的 `config` 字典中，包括：
- 仿真参数
- 强化学习算法超参数
- 网络结构参数
- 训练和测试间隔

## 参考

本项目的灵感来源于：https://github.com/wingsweihua/colight

## 许可证

请参考原论文和相关项目的许可证。
