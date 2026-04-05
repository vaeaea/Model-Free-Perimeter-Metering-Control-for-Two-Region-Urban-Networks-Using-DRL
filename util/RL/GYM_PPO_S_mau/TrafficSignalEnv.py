"""
路网多交叉口信号控制的标准Gymnasium环境实现
重构自原有代码，统一Gymnasium接口，支持标准RL算法调用
"""
import gymnasium as gym
import numpy as np
import traci
from traci import constants
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict
import os
import random
import time
import queue

# 导入拆分的路网解析模块
from util.RL.GYM_PPO_S_mau.ENV import TwoRegionTrafficModel

U_ACCUMULATION_1 = 34000.0  # 区域 1 最大容量 (veh)
U_ACCUMULATION_2 = 17000.0  # 区域 2 最大容量 (veh)
L_ACCUMULATION = 0
MAX_COMPLETION = 33168 / 3600

class TrafficSignalEnv(gym.Env):
    """
    路网多交叉口信号控制的标准Gym环境
    兼容标准RL算法接口，支持PPO等算法直接调用
    """

    def __init__(self, config: Dict,  ):
        """
        初始化环境
        Args:
            config: 配置字典，包含sumo配置、控制参数等
            render_mode: 渲染模式，human为启动SUMO GUI
        """
        super().__init__()
        self.config = config

        # 用拆分的路网解析器解析路网
        self.TrafficModel = TwoRegionTrafficModel(step_length=self.config['control_period'])

        # 定义标准观测空间和动作空间 todo 需要确认low
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape= (8, ),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.1,
            high=0.9,
            shape= (2, ),
            dtype=np.float32
        )

        # SUMO相关
        self.step_count = 0
        self.raw_rewards = 0
        self.veh_count = 0

        # 使用队列记录控制周期内的观测和奖励（便于管理和内存控制）
        self.control_period = config.get('control_period', 90)
        self.obs_queue = []  # 观测队列
        self.reward_queue = []  # 奖励队列

    def reset(self, seed=None, options=None):
        """
        重置环境，标准Gymnasium接口
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super(TrafficSignalEnv, self).reset(seed=seed)

        self.step_count = 0
        self.raw_rewards = 0
        self.veh_count = 0
        self.TrafficModel.reset()

        # 获取初始观测
        obs = self._get_observation()
        info = self._get_info()

        self.obs_queue = []  # 观测队列
        self.reward_queue = []  # 奖励队列
        return obs, info

    def step(self, action):
        """
        执行一步交互，标准 Gymnasium 接口
        Args:
            action: 智能体的动作，每个交叉口的相位时长
        Returns:
            observation: 新的观测
            reward: 奖励
            terminated: 是否结束
            truncated: 是否截断
            info: 额外信息
        """
        # 执行完整的控制周期（多个仿真步）
        state = self.TrafficModel.step(action)
        obs = self._get_observation()
        reward = self._get_reward(state)

        self.step_count += self.control_period

        # 检查是否结束
        terminated = False
        sumo_end_time = self.control_period * self.config['simulation_steps']
        if self.step_count >= sumo_end_time or (state['n1']<=0 and state['n2'] <= 0):
            terminated = True

        truncated = False

        info = self._get_info()
        # 统计原始 reward 数据并加入 info
        self.raw_rewards += reward
        info['raw_rewards'] = self.raw_rewards
        self.veh_count += (state['n1'] + state['n2']) * self.control_period
        info['tts'] = self.veh_count
        info['M11_LIST'] = self.TrafficModel.M11_LIST
        info['M22_LIST'] = self.TrafficModel.M22_LIST
        return obs, reward, terminated, truncated, info


    def _get_observation(self):
        """获取观测"""
        # 下一时刻需求和当前累积量
        return np.array(self.TrafficModel.get_state(), dtype=np.float32)

    def _get_reward(self, state):
        """获取奖励，包含总等待时间和路网总车辆数的加权组合"""
        reward = (state['M11'] + state['M12']) / (MAX_COMPLETION*2*5)
        if state['n1'] > U_ACCUMULATION_1 or state['n2'] > U_ACCUMULATION_2:
            reward -= 5
        return reward

    def _get_info(self):
        """获取额外信息，包含 episode 统计（供 RecordEpisodeStatistics 使用）"""
        import time

        info = {'step_count': self.step_count}
        # print(info)

        return info

    def render(self):
        """渲染，SUMO GUI会自动渲染"""
        pass

    def close(self):
        """关闭环境"""
        try:
            self.TrafficModel.reset()
        except Exception as e:
            # print(f"   ⚠️  关闭环境时出错：{e}")
            pass
