"""
两区域宏观交通流量转移模型
灵感来源于 C-D-RL-main 的 rlplant.py，纯交通流仿真模型
可嵌入到任意 RL 框架中使用
"""
import numpy as np
import random
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import torch
from util.RL.GYM_DDPG_S_mau.DDPG import DDPGAgent

from util.RL.GYM_PPO_S_mau.PPO import PPOAgent

# ==================== 交通需求函数 ====================
class TwoRegionDemandModel:
    """
    两区域交通需求模型
    定义 4 个 OD 对的时变交通需求函数
    """

    def __init__(self, sigma: float = 0.1):
        """
        初始化需求模型
        Args:
            sigma: 随机扰动标准差
        """
        self.sigma = sigma

    # 定义fig4.a中的交通需求
    @staticmethod
    def q_11(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 300:
            return max(0.25 * (1 + randomness), 0) * scaler
        elif t <= 1300:
            return max((0.00065 * t + 0.055) * (1 + randomness), 0) * scaler
        elif t <= 2200:
            return max(0.9 * (1 + randomness), 0) * scaler
        elif t <= 3200:
            return max((-0.00065 * t + 2.33) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def q_12(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 200:
            return max((0.015 * t + 0.25) * (1 + randomness), 0) * scaler
        elif t <= 3000:
            return max(3.25 * (1 + randomness), 0) * scaler
        elif t <= 3600:
            return max((18.25 - t / 200) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def q_21(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 300:
            return max(0.25 * (1 + randomness), 0) * scaler
        elif t <= 1800:
            return max((t / 1500 + 0.05) * (1 + randomness), 0) * scaler
        elif t <= 3200:
            return max(1.25 * (1 + randomness), 0) * scaler
        elif t <= 3600:
            return max((9.25 - t / 400) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def q_22(sigma, t, scaler=1):
        randomness = random.gauss(0, sigma)
        if t <= 100:
            return max(0.25 * (1 + randomness), 0) * scaler
        elif t <= 900:
            return max((1.25 / 800 * t + 0.09375) * (1 + randomness), 0) * scaler
        elif t <= 2700:
            return max(1.5 * (1 + randomness), 0) * scaler
        elif t <= 3500:
            return max((-1.25 / 800 * t + 5.71875) * (1 + randomness), 0) * scaler
        else:
            return max(0.25 * (1 + randomness), 0) * scaler

    @staticmethod
    def MFD(n, alpha=0):  # f(n) + error R1区域
        # 作用：实现宏观基本图（Macroscopic Fundamental Diagram）模型
        # 分段表示，小于14000时为三次函数，大于14000时为线性函数
        error = random.uniform(-alpha, alpha) * n
        if n < 14000:
            return (2.28e-8 * n ** 3 - 8.62e-4 * n ** 2 + 9.58 * n + error) / 3600
        elif n < 34000:
            return (27731 - 1.38655 * (n - 14000) + error) / 3600
        else:
            return 0

    def inner_MFD(self, n, alpha=0): # f(n) + error R2区域
        return self.MFD(2 * n, alpha) * 0.5

U_ACCUMULATION_1 = 34000.0  # 区域 1 最大容量 (veh)
U_ACCUMULATION_2 = 17000.0  # 区域 2 最大容量 (veh)
L_ACCUMULATION = 0.0         # 最小容量 (veh)
# ==================== 两区域宏观交通模型 ====================
class TwoRegionTrafficModel:
    """
    两区域宏观交通流量转移模型
    基于 MFD（宏观基本图）理论，模拟两个区域之间的车辆流动 dynamics

    状态变量：
        n1: 区域 1 的车辆累积量 (veh)
        n2: 区域 2 的车辆累积量 (veh)
        c1: 区域 1 完成的出行数 (veh)
        c2: 区域 2 完成的出行数 (veh)
        t:  当前仿真时间 (s)

    控制输入：
        u1: 区域 1 的流出控制系数 (0.2~0.8)
        u2: 区域 2 的流出控制系数 (0.2~0.8)

    状态方程：
        dn1/dt = (q11 + q21) - (outflow_1_internal + outflow_1_transfer)
        dn2/dt = (q22 + q12) - (outflow_2_internal + outflow_2_transfer)
    """

    # 常量定义

    def __init__(
        self,
        sigma: float = 0.1,
        step_length: float = 60.0,
        initial_n1: float = 6000.0,
        initial_n2: float = 5000.0,
        mfd_alpha: float = 0.1
    ):
        """
        初始化两区域交通模型
        Args:
            sigma: 需求随机扰动强度
            step_length: 仿真步长 (秒)
            initial_n1: 区域 1 初始车辆数
            initial_n2: 区域 2 初始车辆数
        """
        self.sigma = sigma
        self.dt = step_length
        self.mfd_alpha = mfd_alpha

        # 初始状态分解（按 OD 拆分）
        # 假设初始时内部出行和跨区域出行各占一半
        self.initial_n_11 = initial_n1 * 0.5
        self.initial_n_12 = initial_n1 * 0.5
        self.initial_n_21 = initial_n2 * 0.5
        self.initial_n_22 = initial_n2 * 0.5
        self.M11_LIST = []
        self.M22_LIST = []

        # 初始化需求模型
        self.demand_model = TwoRegionDemandModel(sigma=self.sigma)

        # 重置模型状态
        self.reset()

    def reset(self):
        """
        重置模型到初始状态
        """
        self.n_11 = self.initial_n_11
        self.n_12 = self.initial_n_12
        self.n_21 = self.initial_n_21
        self.n_22 = self.initial_n_22
        self.n1 = self.n_11 + self.n_12  # 区域 1 总量
        self.n2 = self.n_21 + self.n_22  # 区域 2 总量
        self.t = 0.0
        self.M11_LIST = []
        self.M22_LIST = []

        # 统计信息
        self.total_demand = 0.0
        self.total_completion = 0.0

    def get_state(self):
        # 假设交通需求已知、累积量已知
        q_11 = self.demand_model.q_11(self.sigma, self.t + self.dt//2)
        q_12 = self.demand_model.q_12(self.sigma, self.t + self.dt//2)
        q_21 = self.demand_model.q_21(self.sigma, self.t + self.dt//2)
        q_22 = self.demand_model.q_22(self.sigma, self.t + self.dt//2)
        # 下一时刻需求和当前累积量
        return [self.n_11/ U_ACCUMULATION_1, self.n_12/ U_ACCUMULATION_1, self.n_21/ U_ACCUMULATION_2, self.n_22/ U_ACCUMULATION_2, q_11, q_12, q_21, q_22]

    def step(self, action) -> Dict:
        # 动作限幅
        u_12, u_21 = action
        u_12 = np.clip(u_12, 0.1, 0.9)
        u_21 = np.clip(u_21, 0.1, 0.9)

        # ========== 计算 MFD 完成量（按 OD 比例分配） ==========
        # 区域 1 的 MFD 完成量分配
        if self.n1 > 1e-6:
            M11 = self.n_11 / self.n1 * self.demand_model.MFD(self.n1, alpha=self.mfd_alpha)
            M12 = self.n_12 / self.n1 * self.demand_model.MFD(self.n1, alpha=self.mfd_alpha)
        else:
            M11 = 0.0
            M12 = 0.0

        # 区域 2 的 MFD 完成量分配
        if self.n2 > 1e-6:
            M21 = self.n_21 / self.n2 * self.demand_model.inner_MFD(self.n2, alpha=self.mfd_alpha)
            M22 = self.n_22 / self.n2 * self.demand_model.inner_MFD(self.n2, alpha=self.mfd_alpha)
        else:
            M21 = 0.0
            M22 = 0.0
        self.M11_LIST.append(M11*self.dt)
        self.M22_LIST.append(M22*self.dt)

        # ========== 计算交通需求 ==========
        # 使用步长中点时刻的需求（提高数值精度）
        q11 = self.demand_model.q_11(self.sigma, self.t + self.dt / 2, scaler=1.0)
        q12 = self.demand_model.q_12(self.sigma, self.t + self.dt / 2, scaler=1.0)
        q21 = self.demand_model.q_21(self.sigma, self.t + self.dt / 2, scaler=1.0)
        q22 = self.demand_model.q_22(self.sigma, self.t + self.dt / 2, scaler=1.0)

        # 记录总需求
        current_demand = q11 + q12 + q21 + q22
        self.total_demand += current_demand * self.dt
        # ========== 状态更新（欧拉法积分） ==========
        # 区域 1 内部出行：新增需求 + 从区域 2 流入 - 完成出行
        self.n_11 += (q11 + u_21 * M21 - M11) * self.dt
        # 区域 1 到区域 2：新增需求 - 流出到区域 2
        self.n_12 += (q12 - u_12 * M12) * self.dt
        # 区域 2 到区域 1：新增需求 - 流出到区域 1
        self.n_21 += (q21 - u_21 * M21) * self.dt
        # 区域 2 内部出行：新增需求 + 从区域 1 流入 - 完成出行
        self.n_22 += (q22 + u_12 * M12 - M22) * self.dt

        # 更新区域总量
        self.n1 = self.n_11 + self.n_12
        self.n2 = self.n_21 + self.n_22

        # 更新时间
        self.t += self.dt
        # ========== 返回状态 ==========
        state = {
            'n_11': self.n_11,
            'n_12': self.n_12,
            'n_21': self.n_21,
            'n_22': self.n_22,
            'n1': self.n1,
            'n2': self.n2,
            't': self.t,
            'q11': q11,
            'q12': q12,
            'q21': q21,
            'q22': q22,
            'M11': M11,
            'M12': M12,
            'M21': M21,
            'M22': M22,
        }
        return state

    def run_without_control(
            self,
            simulation_time: float = 3600.0,
            fixed_u_12: float = 0.5,
            fixed_u_21: float = 0.5,
            verbose: bool = False,
            plot: bool = True
    ) -> Dict:
        """
        在不控制（固定控制系数）情况下运行仿真
        用于基线对比，模拟传统的固定配时策略
        Args:
            simulation_time: 仿真总时长 (s)
            fixed_u_12: 固定的区域 1→2 流出控制系数 (0.1~0.9)
            fixed_u_21: 固定的区域 2→1 流出控制系数 (0.1~0.9)
            verbose: 是否打印进度
            plot: 是否绘制累计完成量图
        Returns:
            results: 仿真结果字典
                - time_series: 时间序列数据
                - statistics: 统计指标
        """
        # 初始化记录器
        time_series = {
            't': [],
            'n_11': [],
            'n_12': [],
            'n_21': [],
            'n_22': [],
            'n1': [],
            'n2': [],
            'q11': [],
            'q12': [],
            'q21': [],
            'q22': [],
            'M11': [],
            'M12': [],
            'M21': [],
            'M22': [],
            'u_12': [],
            'u_21': []
        }

        # 重置模型
        self.reset()

        # 仿真主循环
        num_steps = int(simulation_time / self.dt)

        for step_idx in range(num_steps):
            # 使用固定控制系数（无智能控制）
            u_12 = fixed_u_12
            u_21 = fixed_u_21

            # 执行一步
            next_state = self.step([u_12, u_21])

            # 记录数据
            time_series['t'].append(next_state['t'])
            time_series['n_11'].append(next_state['n_11'])
            time_series['n_12'].append(next_state['n_12'])
            time_series['n_21'].append(next_state['n_21'])
            time_series['n_22'].append(next_state['n_22'])
            time_series['n1'].append(next_state['n1'])
            time_series['n2'].append(next_state['n2'])
            time_series['q11'].append(next_state['q11'])
            time_series['q12'].append(next_state['q12'])
            time_series['q21'].append(next_state['q21'])
            time_series['q22'].append(next_state['q22'])
            time_series['M11'].append(next_state['M11'])
            time_series['M12'].append(next_state['M12'])
            time_series['M21'].append(next_state['M21'])
            time_series['M22'].append(next_state['M22'])
            time_series['u_12'].append(u_12)
            time_series['u_21'].append(u_21)

            if verbose and (step_idx + 1) % 100 == 0:
                current_time = (step_idx + 1) * self.dt
                print(f"t={current_time:.0f}s | n1={self.n1:.0f} | n2={self.n2:.0f} | "
                      f"M11={next_state['M11']:.2f} | M22={next_state['M22']:.2f}")

        # 计算统计指标
        total_completion_M11 = sum(self.M11_LIST)
        total_completion_M22 = sum(self.M22_LIST)
        total_completion = total_completion_M11 + total_completion_M22

        statistics = {
            'final_n1': self.n1,
            'final_n2': self.n2,
            'total_demand': self.total_demand,
            'total_completion': total_completion,
            'completion_rate': total_completion / (self.total_demand + 1e-6),
            'avg_n1': np.mean(time_series['n1']),
            'avg_n2': np.mean(time_series['n2']),
            'max_n1': np.max(time_series['n1']),
            'max_n2': np.max(time_series['n2']),
            'avg_completion_rate': total_completion / (self.total_demand + 1e-6),
            'fixed_u_12': fixed_u_12,
            'fixed_u_21': fixed_u_21
        }

        results = {
            'time_series': time_series,
            'statistics': statistics
        }

        # ========== 绘制累计完成量图 ==========
        if plot:
            try:
                import matplotlib.pyplot as plt

                # 计算累计完成量
                cumulative_M11 = np.cumsum(self.M11_LIST)
                cumulative_M22 = np.cumsum(self.M22_LIST)
                cumulative_total = cumulative_M11 + cumulative_M22
                time_array = np.array(time_series['t'])

                # 计算累计车辆 - 时间（VHT）
                # (n1 + n2) * dt 的累加，表示总的车辆滞在时间
                accumulation_array = np.array(time_series['n1']) + np.array(time_series['n2'])
                vht_per_step = accumulation_array * self.dt  # 每步的车辆 - 秒
                cumulative_vht = np.cumsum(vht_per_step)  # 累计车辆 - 秒
                cumulative_vht_hours = cumulative_vht / 3600  # 转换为车辆 - 小时

                # 创建图形（2x3 布局）
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                # 子图 1：累计完成量（总量 + 分区域）
                ax1 = axes[0, 0]
                ax1.plot(time_array / 3600, cumulative_M11, 'b-', linewidth=2, label='Region 1 (M11)')
                ax1.plot(time_array / 3600, cumulative_M22, 'g-', linewidth=2, label='Region 2 (M22)')
                ax1.plot(time_array / 3600, cumulative_total, 'r-', linewidth=2.5, label='Total')
                ax1.set_xlabel('Time (hour)', fontsize=12)
                ax1.set_ylabel('Cumulative Completion (veh)', fontsize=12)
                ax1.set_title('Cumulative Vehicle Completions', fontsize=14)
                ax1.legend(loc='best', fontsize=11)
                ax1.grid(True, alpha=0.3)

                # 子图 2：累计车辆 - 时间（VHT）
                ax2 = axes[0, 1]
                ax2.plot(time_array / 3600, cumulative_vht_hours, 'm-', linewidth=2.5)
                ax2.set_xlabel('Time (hour)', fontsize=12)
                ax2.set_ylabel('Cumulative VHT (veh-hour)', fontsize=12)
                ax2.set_title('Cumulative Vehicle-Hours Traveled', fontsize=14)
                ax2.grid(True, alpha=0.3)

                # 添加 VHT 增长率（导数）
                ax2_twin = ax2.twinx()
                vht_rate = vht_per_step / 3600  # veh-hour/step
                ax2_twin.plot(time_array / 3600, vht_rate, 'm--', linewidth=1, alpha=0.5, label='VHT rate per step')
                ax2_twin.set_ylabel('VHT per Step (veh-hour)', fontsize=11)
                ax2_twin.legend(loc='upper right', fontsize=10)

                # 子图 3：区域累积量变化
                ax3 = axes[0, 2]
                ax3.plot(time_array / 3600, time_series['n1'], 'b-', linewidth=2, label='Region 1 (n1)')
                ax3.plot(time_array / 3600, time_series['n2'], 'r-', linewidth=2, label='Region 2 (n2)')
                ax3.plot(time_array / 3600, accumulation_array, 'k--', linewidth=1.5, label='Total (n1+n2)')
                ax3.set_xlabel('Time (hour)', fontsize=12)
                ax3.set_ylabel('Accumulation (veh)', fontsize=12)
                ax3.set_title('Vehicle Accumulation in Regions', fontsize=14)
                ax3.legend(loc='best', fontsize=11)
                ax3.grid(True, alpha=0.3)

                # 子图 4：累计完成量增长率（导数）
                ax4 = axes[1, 0]
                if len(self.M11_LIST) > 1:
                    dM11 = np.diff(self.M11_LIST) * 3600  # veh/h
                    dM22 = np.diff(self.M22_LIST) * 3600
                    time_diff = time_array[:-1] / 3600
                    ax4.plot(time_diff, dM11, 'b-', linewidth=1.5, label='Region 1 (M11)', alpha=0.7)
                    ax4.plot(time_diff, dM22, 'g-', linewidth=1.5, label='Region 2 (M22)', alpha=0.7)
                    ax4.set_xlabel('Time (hour)', fontsize=12)
                    ax4.set_ylabel('Completion Rate (veh/h)', fontsize=12)
                    ax4.set_title('Completion Rate Over Time', fontsize=14)
                    ax4.legend(loc='best', fontsize=11)
                    ax4.grid(True, alpha=0.3)

                # 子图 5：VHT 与完成量的关系
                ax5 = axes[1, 1]
                efficiency = cumulative_total / (cumulative_vht_hours + 1e-6)  # veh / veh-hour = 1/hour
                ax5.plot(time_array / 3600, efficiency, 'c-', linewidth=2)
                ax5.set_xlabel('Time (hour)', fontsize=12)
                ax5.set_ylabel('Efficiency (1/hour)', fontsize=12)
                ax5.set_title('System Efficiency (Completion/VHT)', fontsize=14)
                ax5.grid(True, alpha=0.3)

                # 子图 6：控制系数和关键统计
                ax6 = axes[1, 2]
                ax6.axis('off')

                # 显示控制参数和统计信息
                stats_text = f"Control Parameters:\n"
                stats_text += f"  u_12 = {fixed_u_12:.2f}\n"
                stats_text += f"  u_21 = {fixed_u_21:.2f}\n\n"
                stats_text += f"Statistics:\n"
                stats_text += f"  Total Demand: {statistics['total_demand']:.0f} veh\n"
                stats_text += f"  Total Completion: {statistics['total_completion']:.0f} veh\n"
                stats_text += f"  Completion Rate: {statistics['completion_rate'] * 100:.2f}%\n\n"
                stats_text += f"  Region 1 Completion: {total_completion_M11:.0f} veh\n"
                stats_text += f"  Region 2 Completion: {total_completion_M22:.0f} veh\n\n"
                stats_text += f"  Total VHT: {cumulative_vht_hours[-1]:.1f} veh-hour\n"
                stats_text += f"  Avg Accumulation: {np.mean(accumulation_array):.0f} veh\n"
                stats_text += f"  Max Accumulation: {np.max(accumulation_array):.0f} veh\n\n"
                stats_text += f"  Avg n1: {statistics['avg_n1']:.0f} veh\n"
                stats_text += f"  Avg n2: {statistics['avg_n2']:.0f} veh\n"
                stats_text += f"  Max n1: {statistics['max_n1']:.0f} veh\n"
                stats_text += f"  Max n2: {statistics['max_n2']:.0f} veh"

                ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.suptitle(
                    f'Two-Region Traffic Simulation Results (Fixed Control: u₁₂={fixed_u_12:.2f}, u₂₁={fixed_u_21:.2f})',
                    fontsize=16, y=1.02)
                plt.tight_layout()

                # 保存图片
                save_path = f"results_fixed_control_u12_{fixed_u_12:.2f}_u21_{fixed_u_21:.2f}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✅ 累计完成量图已保存至：{save_path}")

                plt.show()

            except ImportError:
                print("⚠️  警告：matplotlib 未安装，无法绘制图表")
            except Exception as e:
                print(f"⚠️  绘图时出错：{e}")

        return results

    def run_with_ddpg_control(
            self,
            ddpg_agent,
            model_path: str,
            simulation_time: float = 3600.0,
            verbose: bool = False,
            plot: bool = True
    ) -> Dict:
        """
        使用训练好的 DDPG 模型进行控制仿真
        加载预训练的 DDPG 智能体，通过 Actor 网络生成实时控制动作

        Args:
            ddpg_agent: DDPGAgent 实例（需先初始化）
            model_path: 训练好的模型参数文件路径 (.pth 或 .pt)
            simulation_time: 仿真总时长 (s)
            verbose: 是否打印进度
            plot: 是否绘制累计完成量图

        Returns:
            results: 仿真结果字典
                - time_series: 时间序列数据
                - statistics: 统计指标
                - control_info: 控制信息（包括使用的模型路径）
        """
        # ========== 加载训练好的 DDPG 模型 ==========
        print(f"\n正在加载 DDPG 模型：{model_path}")
        ddpg_agent.load_model(model_path)
        ddpg_agent.actor.eval()  # 设置为评估模式
        print("✅ DDPG 模型加载完成，切换到评估模式 (eval)\n")

        # 初始化记录器
        time_series = {
            't': [],
            'n_11': [],
            'n_12': [],
            'n_21': [],
            'n_22': [],
            'n1': [],
            'n2': [],
            'q11': [],
            'q12': [],
            'q21': [],
            'q22': [],
            'M11': [],
            'M12': [],
            'M21': [],
            'M22': [],
            'u_12': [],
            'u_21': []
        }

        # 重置模型
        self.reset()

        # 仿真主循环
        num_steps = int(simulation_time / self.dt)

        for step_idx in range(num_steps):
            # ========== 使用 DDPG 生成控制动作 ==========
            # 获取当前状态（归一化）
            state = self.get_state()

            # 通过 Actor 网络生成动作（确定性策略）
            action = ddpg_agent.select_actions(state, deterministic=True)

            # 动作限幅到合理范围 [0.1, 0.9]
            u_12 = np.clip(action[0], 0.1, 0.9)
            u_21 = np.clip(action[1], 0.1, 0.9)

            # 执行一步
            next_state = self.step([u_12, u_21])

            # 记录数据
            time_series['t'].append(next_state['t'])
            time_series['n_11'].append(next_state['n_11'])
            time_series['n_12'].append(next_state['n_12'])
            time_series['n_21'].append(next_state['n_21'])
            time_series['n_22'].append(next_state['n_22'])
            time_series['n1'].append(next_state['n1'])
            time_series['n2'].append(next_state['n2'])
            time_series['q11'].append(next_state['q11'])
            time_series['q12'].append(next_state['q12'])
            time_series['q21'].append(next_state['q21'])
            time_series['q22'].append(next_state['q22'])
            time_series['M11'].append(next_state['M11'])
            time_series['M12'].append(next_state['M12'])
            time_series['M21'].append(next_state['M21'])
            time_series['M22'].append(next_state['M22'])
            time_series['u_12'].append(u_12)
            time_series['u_21'].append(u_21)

            if verbose and (step_idx + 1) % 100 == 0:
                current_time = (step_idx + 1) * self.dt
                print(f"t={current_time:.0f}s | n1={self.n1:.0f} | n2={self.n2:.0f} | "
                      f"u_12={u_12:.3f} | u_21={u_21:.3f} | "
                      f"M11={next_state['M11']:.2f} | M22={next_state['M22']:.2f}")

        # 计算统计指标
        total_completion_M11 = sum(self.M11_LIST)
        total_completion_M22 = sum(self.M22_LIST)
        total_completion = total_completion_M11 + total_completion_M22

        statistics = {
            'final_n1': self.n1,
            'final_n2': self.n2,
            'total_demand': self.total_demand,
            'total_completion': total_completion,
            'completion_rate': total_completion / (self.total_demand + 1e-6),
            'avg_n1': np.mean(time_series['n1']),
            'avg_n2': np.mean(time_series['n2']),
            'max_n1': np.max(time_series['n1']),
            'max_n2': np.max(time_series['n2']),
            'avg_completion_rate': total_completion / (self.total_demand + 1e-6),
            'control_type': 'DDPG',
            'model_path': model_path
        }

        # 计算平均控制系数
        avg_u_12 = np.mean(time_series['u_12'])
        avg_u_21 = np.mean(time_series['u_21'])
        std_u_12 = np.std(time_series['u_12'])
        std_u_21 = np.std(time_series['u_21'])

        control_info = {
            'model_path': model_path,
            'avg_u_12': avg_u_12,
            'avg_u_21': avg_u_21,
            'std_u_12': std_u_12,
            'std_u_21': std_u_21,
            'min_u_12': np.min(time_series['u_12']),
            'max_u_12': np.max(time_series['u_12']),
            'min_u_21': np.min(time_series['u_21']),
            'max_u_21': np.max(time_series['u_21'])
        }

        results = {
            'time_series': time_series,
            'statistics': statistics,
            'control_info': control_info
        }

        # ========== 绘制累计完成量图 ==========
        if plot:
            try:
                import matplotlib.pyplot as plt

                # 计算累计完成量
                cumulative_M11 = np.cumsum(self.M11_LIST)
                cumulative_M22 = np.cumsum(self.M22_LIST)
                cumulative_total = cumulative_M11 + cumulative_M22
                time_array = np.array(time_series['t'])

                # 计算累计车辆 - 时间（VHT）
                accumulation_array = np.array(time_series['n1']) + np.array(time_series['n2'])
                vht_per_step = accumulation_array * self.dt
                cumulative_vht = np.cumsum(vht_per_step)
                cumulative_vht_hours = cumulative_vht / 3600

                # 创建图形（2x3 布局）
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                # 子图 1：累计完成量
                ax1 = axes[0, 0]
                ax1.plot(time_array / 3600, cumulative_M11, 'b-', linewidth=2, label='Region 1 (M11)')
                ax1.plot(time_array / 3600, cumulative_M22, 'g-', linewidth=2, label='Region 2 (M22)')
                ax1.plot(time_array / 3600, cumulative_total, 'r-', linewidth=2.5, label='Total')
                ax1.set_xlabel('Time (hour)', fontsize=12)
                ax1.set_ylabel('Cumulative Completion (veh)', fontsize=12)
                ax1.set_title('Cumulative Vehicle Completions', fontsize=14)
                ax1.legend(loc='best', fontsize=11)
                ax1.grid(True, alpha=0.3)

                # 子图 2：累计 VHT
                ax2 = axes[0, 1]
                ax2.plot(time_array / 3600, cumulative_vht_hours, 'm-', linewidth=2.5)
                ax2.set_xlabel('Time (hour)', fontsize=12)
                ax2.set_ylabel('Cumulative VHT (veh-hour)', fontsize=12)
                ax2.set_title('Cumulative Vehicle-Hours Traveled', fontsize=14)
                ax2.grid(True, alpha=0.3)

                ax2_twin = ax2.twinx()
                vht_rate = vht_per_step / 3600
                ax2_twin.plot(time_array / 3600, vht_rate, 'm--', linewidth=1, alpha=0.5, label='VHT rate per step')
                ax2_twin.set_ylabel('VHT per Step (veh-hour)', fontsize=11)
                ax2_twin.legend(loc='upper right', fontsize=10)

                # 子图 3：区域累积量
                ax3 = axes[0, 2]
                ax3.plot(time_array / 3600, time_series['n1'], 'b-', linewidth=2, label='Region 1 (n1)')
                ax3.plot(time_array / 3600, time_series['n2'], 'r-', linewidth=2, label='Region 2 (n2)')
                ax3.plot(time_array / 3600, accumulation_array, 'k--', linewidth=1.5, label='Total (n1+n2)')
                ax3.set_xlabel('Time (hour)', fontsize=12)
                ax3.set_ylabel('Accumulation (veh)', fontsize=12)
                ax3.set_title('Vehicle Accumulation in Regions', fontsize=14)
                ax3.legend(loc='best', fontsize=11)
                ax3.grid(True, alpha=0.3)

                # 子图 4：累计完成量增长率
                ax4 = axes[1, 0]
                if len(self.M11_LIST) > 1:
                    dM11 = np.diff(self.M11_LIST) * 3600
                    dM22 = np.diff(self.M22_LIST) * 3600
                    time_diff = time_array[:-1] / 3600
                    ax4.plot(time_diff, dM11, 'b-', linewidth=1.5, label='Region 1 (M11)', alpha=0.7)
                    ax4.plot(time_diff, dM22, 'g-', linewidth=1.5, label='Region 2 (M22)', alpha=0.7)
                    ax4.set_xlabel('Time (hour)', fontsize=12)
                    ax4.set_ylabel('Completion Rate (veh/h)', fontsize=12)
                    ax4.set_title('Completion Rate Over Time', fontsize=14)
                    ax4.legend(loc='best', fontsize=11)
                    ax4.grid(True, alpha=0.3)

                # 子图 5：系统效率
                ax5 = axes[1, 1]
                efficiency = cumulative_total / (cumulative_vht_hours + 1e-6)
                ax5.plot(time_array / 3600, efficiency, 'c-', linewidth=2)
                ax5.set_xlabel('Time (hour)', fontsize=12)
                ax5.set_ylabel('Efficiency (1/hour)', fontsize=12)
                ax5.set_title('System Efficiency (Completion/VHT)', fontsize=14)
                ax5.grid(True, alpha=0.3)

                # 子图 6：控制信息和统计
                ax6 = axes[1, 2]
                ax6.axis('off')

                # 显示控制参数和统计信息
                stats_text = f"DDPG Control Information:\n"
                stats_text += f"  Model: {model_path.split('/')[-1]}\n"
                stats_text += f"  u_12: {avg_u_12:.3f} ± {std_u_12:.3f}\n"
                stats_text += f"  u_21: {avg_u_21:.3f} ± {std_u_21:.3f}\n\n"
                stats_text += f"Statistics:\n"
                stats_text += f"  Total Demand: {statistics['total_demand']:.0f} veh\n"
                stats_text += f"  Total Completion: {statistics['total_completion']:.0f} veh\n"
                stats_text += f"  Completion Rate: {statistics['completion_rate'] * 100:.2f}%\n\n"
                stats_text += f"  Region 1 Completion: {total_completion_M11:.0f} veh\n"
                stats_text += f"  Region 2 Completion: {total_completion_M22:.0f} veh\n\n"
                stats_text += f"  Total VHT: {cumulative_vht_hours[-1]:.1f} veh-hour\n"
                stats_text += f"  Avg Accumulation: {np.mean(accumulation_array):.0f} veh\n"
                stats_text += f"  Max Accumulation: {np.max(accumulation_array):.0f} veh\n\n"
                stats_text += f"  Avg n1: {statistics['avg_n1']:.0f} veh\n"
                stats_text += f"  Avg n2: {statistics['avg_n2']:.0f} veh\n"
                stats_text += f"  Max n1: {statistics['max_n1']:.0f} veh\n"
                stats_text += f"  Max n2: {statistics['max_n2']:.0f} veh"

                ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

                plt.suptitle(
                    f'Two-Region Traffic Simulation Results (DDPG Control)',
                    fontsize=16, y=1.02)
                plt.tight_layout()

                # 保存图片
                save_path = f"results_ddpg_control.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✅ 累计完成量图已保存至：{save_path}")

                plt.show()

            except ImportError:
                print("⚠️  警告：matplotlib 未安装，无法绘制图表")
            except Exception as e:
                print(f"⚠️  绘图时出错：{e}")

        return results

    def run_with_ppo_control(
            self,
            ppo_agent,
            model_path: str,
            simulation_time: float = 3600.0,
            verbose: bool = False,
            plot: bool = True
    ) -> Dict:
        """
        使用训练好的 PPO 模型进行控制仿真
        加载预训练的 PPO 智能体，通过 Actor 网络生成实时控制动作

        Args:
            ppo_agent: PPOAgent 实例（需先初始化）
            model_path: 训练好的模型参数文件路径 (.pth 或 .pt)
            simulation_time: 仿真总时长 (s)
            verbose: 是否打印进度
            plot: 是否绘制累计完成量图

        Returns:
            results: 仿真结果字典
                - time_series: 时间序列数据
                - statistics: 统计指标
                - control_info: 控制信息
        """
        # ========== 加载训练好的 PPO 模型 ==========
        print(f"\n正在加载 PPO 模型：{model_path}")
        ppo_agent.load_model(model_path)
        ppo_agent.actor.eval()  # 设置为评估模式
        print("✅ PPO 模型加载完成，切换到评估模式 (eval)\n")

        # 初始化记录器
        time_series = {
            't': [],
            'n_11': [],
            'n_12': [],
            'n_21': [],
            'n_22': [],
            'n1': [],
            'n2': [],
            'q11': [],
            'q12': [],
            'q21': [],
            'q22': [],
            'M11': [],
            'M12': [],
            'M21': [],
            'M22': [],
            'u_12': [],
            'u_21': []
        }

        # 重置模型
        self.reset()

        # 仿真主循环
        num_steps = int(simulation_time / self.dt)

        for step_idx in range(num_steps):
            # ========== 使用 PPO 生成控制动作 ==========
            # 获取当前状态（归一化）
            state = self.get_state()

            # 通过 Actor 网络生成动作（确定性策略）
            action, _ = ppo_agent.select_actions(torch.tensor(state, dtype=torch.float32, device=ppo_agent.config['device']).unsqueeze(0), deterministic=True)

            # PPO 输出已经是 [0.1, 0.9] 范围（在 select_actions 中转换）
            u_12 = np.clip(action[0,0], 0.1, 0.9)
            u_21 = np.clip(action[0,1], 0.1, 0.9)

            # 执行一步
            next_state = self.step([u_12, u_21])

            # 记录数据
            time_series['t'].append(next_state['t'])
            time_series['n_11'].append(next_state['n_11'])
            time_series['n_12'].append(next_state['n_12'])
            time_series['n_21'].append(next_state['n_21'])
            time_series['n_22'].append(next_state['n_22'])
            time_series['n1'].append(next_state['n1'])
            time_series['n2'].append(next_state['n2'])
            time_series['q11'].append(next_state['q11'])
            time_series['q12'].append(next_state['q12'])
            time_series['q21'].append(next_state['q21'])
            time_series['q22'].append(next_state['q22'])
            time_series['M11'].append(next_state['M11'])
            time_series['M12'].append(next_state['M12'])
            time_series['M21'].append(next_state['M21'])
            time_series['M22'].append(next_state['M22'])
            time_series['u_12'].append(u_12)
            time_series['u_21'].append(u_21)

            if verbose and (step_idx + 1) % 100 == 0:
                current_time = (step_idx + 1) * self.dt
                print(f"t={current_time:.0f}s | n1={self.n1:.0f} | n2={self.n2:.0f} | "
                      f"u_12={u_12:.3f} | u_21={u_21:.3f} | "
                      f"M11={next_state['M11']:.2f} | M22={next_state['M22']:.2f}")

        # 计算统计指标
        total_completion_M11 = sum(self.M11_LIST)
        total_completion_M22 = sum(self.M22_LIST)
        total_completion = total_completion_M11 + total_completion_M22

        statistics = {
            'final_n1': self.n1,
            'final_n2': self.n2,
            'total_demand': self.total_demand,
            'total_completion': total_completion,
            'completion_rate': total_completion / (self.total_demand + 1e-6),
            'avg_n1': np.mean(time_series['n1']),
            'avg_n2': np.mean(time_series['n2']),
            'max_n1': np.max(time_series['n1']),
            'max_n2': np.max(time_series['n2']),
            'avg_completion_rate': total_completion / (self.total_demand + 1e-6),
            'control_type': 'PPO',
            'model_path': model_path
        }

        # 计算平均控制系数
        avg_u_12 = np.mean(time_series['u_12'])
        avg_u_21 = np.mean(time_series['u_21'])
        std_u_12 = np.std(time_series['u_12'])
        std_u_21 = np.std(time_series['u_21'])

        control_info = {
            'model_path': model_path,
            'avg_u_12': avg_u_12,
            'avg_u_21': avg_u_21,
            'std_u_12': std_u_12,
            'std_u_21': std_u_21,
            'min_u_12': np.min(time_series['u_12']),
            'max_u_12': np.max(time_series['u_12']),
            'min_u_21': np.min(time_series['u_21']),
            'max_u_21': np.max(time_series['u_21'])
        }

        results = {
            'time_series': time_series,
            'statistics': statistics,
            'control_info': control_info
        }

        # ========== 绘制累计完成量图 ==========
        if plot:
            try:
                import matplotlib.pyplot as plt

                # 计算累计完成量
                cumulative_M11 = np.cumsum(self.M11_LIST) * self.dt
                cumulative_M22 = np.cumsum(self.M22_LIST) * self.dt
                cumulative_total = cumulative_M11 + cumulative_M22
                time_array = np.array(time_series['t'])

                # 计算累计车辆 - 时间（VHT）
                accumulation_array = np.array(time_series['n1']) + np.array(time_series['n2'])
                vht_per_step = accumulation_array * self.dt
                cumulative_vht = np.cumsum(vht_per_step)
                cumulative_vht_hours = cumulative_vht / 3600

                # 创建图形（2x3 布局）
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                # 子图 1：累计完成量
                ax1 = axes[0, 0]
                ax1.plot(time_array / 3600, cumulative_M11, 'b-', linewidth=2, label='Region 1 (M11)')
                ax1.plot(time_array / 3600, cumulative_M22, 'g-', linewidth=2, label='Region 2 (M22)')
                ax1.plot(time_array / 3600, cumulative_total, 'r-', linewidth=2.5, label='Total')
                ax1.set_xlabel('Time (hour)', fontsize=12)
                ax1.set_ylabel('Cumulative Completion (veh)', fontsize=12)
                ax1.set_title('Cumulative Vehicle Completions', fontsize=14)
                ax1.legend(loc='best', fontsize=11)
                ax1.grid(True, alpha=0.3)

                # 子图 2：累计 VHT
                ax2 = axes[0, 1]
                ax2.plot(time_array / 3600, cumulative_vht_hours, 'm-', linewidth=2.5)
                ax2.set_xlabel('Time (hour)', fontsize=12)
                ax2.set_ylabel('Cumulative VHT (veh-hour)', fontsize=12)
                ax2.set_title('Cumulative Vehicle-Hours Traveled', fontsize=14)
                ax2.grid(True, alpha=0.3)

                ax2_twin = ax2.twinx()
                vht_rate = vht_per_step / 3600
                ax2_twin.plot(time_array / 3600, vht_rate, 'm--', linewidth=1, alpha=0.5, label='VHT rate per step')
                ax2_twin.set_ylabel('VHT per Step (veh-hour)', fontsize=11)
                ax2_twin.legend(loc='upper right', fontsize=10)

                # 子图 3：区域累积量
                ax3 = axes[0, 2]
                ax3.plot(time_array / 3600, time_series['n1'], 'b-', linewidth=2, label='Region 1 (n1)')
                ax3.plot(time_array / 3600, time_series['n2'], 'r-', linewidth=2, label='Region 2 (n2)')
                ax3.plot(time_array / 3600, accumulation_array, 'k--', linewidth=1.5, label='Total (n1+n2)')
                ax3.set_xlabel('Time (hour)', fontsize=12)
                ax3.set_ylabel('Accumulation (veh)', fontsize=12)
                ax3.set_title('Vehicle Accumulation in Regions', fontsize=14)
                ax3.legend(loc='best', fontsize=11)
                ax3.grid(True, alpha=0.3)

                # 子图 4：累计完成量增长率
                ax4 = axes[1, 0]
                if len(self.M11_LIST) > 1:
                    dM11 = np.diff(self.M11_LIST) * 3600
                    dM22 = np.diff(self.M22_LIST) * 3600
                    time_diff = time_array[:-1] / 3600
                    ax4.plot(time_diff, dM11, 'b-', linewidth=1.5, label='Region 1 (M11)', alpha=0.7)
                    ax4.plot(time_diff, dM22, 'g-', linewidth=1.5, label='Region 2 (M22)', alpha=0.7)
                    ax4.set_xlabel('Time (hour)', fontsize=12)
                    ax4.set_ylabel('Completion Rate (veh/h)', fontsize=12)
                    ax4.set_title('Completion Rate Over Time', fontsize=14)
                    ax4.legend(loc='best', fontsize=11)
                    ax4.grid(True, alpha=0.3)

                # 子图 5：系统效率
                ax5 = axes[1, 1]
                efficiency = cumulative_total / (cumulative_vht_hours + 1e-6)
                ax5.plot(time_array / 3600, efficiency, 'c-', linewidth=2)
                ax5.set_xlabel('Time (hour)', fontsize=12)
                ax5.set_ylabel('Efficiency (1/hour)', fontsize=12)
                ax5.set_title('System Efficiency (Completion/VHT)', fontsize=14)
                ax5.grid(True, alpha=0.3)

                # 子图 6：控制信息和统计
                ax6 = axes[1, 2]
                ax6.axis('off')

                # 显示控制参数和统计信息
                stats_text = f"PPO Control Information:\n"
                stats_text += f"  Model: {model_path.split('/')[-1]}\n"
                stats_text += f"  u_12: {avg_u_12:.3f} ± {std_u_12:.3f}\n"
                stats_text += f"  u_21: {avg_u_21:.3f} ± {std_u_21:.3f}\n\n"
                stats_text += f"Statistics:\n"
                stats_text += f"  Total Demand: {statistics['total_demand']:.0f} veh\n"
                stats_text += f"  Total Completion: {statistics['total_completion']:.0f} veh\n"
                stats_text += f"  Completion Rate: {statistics['completion_rate'] * 100:.2f}%\n\n"
                stats_text += f"  Region 1 Completion: {total_completion_M11:.0f} veh\n"
                stats_text += f"  Region 2 Completion: {total_completion_M22:.0f} veh\n\n"
                stats_text += f"  Total VHT: {cumulative_vht_hours[-1]:.1f} veh-hour\n"
                stats_text += f"  Avg Accumulation: {np.mean(accumulation_array):.0f} veh\n"
                stats_text += f"  Max Accumulation: {np.max(accumulation_array):.0f} veh\n\n"
                stats_text += f"  Avg n1: {statistics['avg_n1']:.0f} veh\n"
                stats_text += f"  Avg n2: {statistics['avg_n2']:.0f} veh\n"
                stats_text += f"  Max n1: {statistics['max_n1']:.0f} veh\n"
                stats_text += f"  Max n2: {statistics['max_n2']:.0f} veh"

                ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

                plt.suptitle(
                    f'Two-Region Traffic Simulation Results (PPO Control)',
                    fontsize=16, y=1.02)
                plt.tight_layout()

                # 保存图片
                model_name = model_path.split('/')[-1].split('.')[0]
                save_path = f"results_ppo_control_{model_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✅ 累计完成量图已保存至：{save_path}")

                plt.show()

            except ImportError:
                print("⚠️  警告：matplotlib 未安装，无法绘制图表")
            except Exception as e:
                print(f"⚠️  绘图时出错：{e}")

        return results

    def run_multi_algorithm_comparison(
            self,
            ddpg_agent=None,
            ddpg_model_path: str = None,
            ppo_agent=None,
            ppo_model_path: str = None,
            simulation_time: float = 3600.0,
            fixed_u_12: float = 0.5,
            fixed_u_21: float = 0.5,
            verbose: bool = False,
            plot: bool = True
    ) -> Dict:
        """
        多算法对比：DDPG vs PPO vs Fixed Control

        依次运行所有可用的控制策略，并在同一张图上进行全面对比

        Args:
            ddpg_agent: DDPGAgent 实例（可选）
            ddpg_model_path: DDPG 模型路径（可选）
            ppo_agent: PPOAgent 实例（可选）
            ppo_model_path: PPO 模型路径（可选）
            simulation_time: 仿真总时长 (s)
            fixed_u_12: 固定控制系数 u_12
            fixed_u_21: 固定控制系数 u_21
            verbose: 是否打印进度
            plot: 是否绘制对比图

        Returns:
            results: 包含所有策略的结果字典
                - ddpg: DDPG 策略结果（如果提供）
                - ppo: PPO 策略结果（如果提供）
                - fixed: 固定控制策略结果
                - comparison: 性能对比指标
        """
        print("\n" + "=" * 80)
        print("开始运行多算法对比实验 (DDPG vs PPO vs Fixed)")
        print("=" * 80)

        results = {}
        step_count = 1

        # ========== 运行 DDPG 控制策略 ==========
        if ddpg_agent is not None and ddpg_model_path is not None:
            print(f"\n【步骤 {step_count}/?】运行 DDPG 控制策略...")
            ddpg_agent.load_model(ddpg_model_path)
            ddpg_agent.actor.eval()

            results['ddpg'] = self.run_with_ddpg_control(
                ddpg_agent=ddpg_agent,
                model_path=ddpg_model_path,
                simulation_time=simulation_time,
                verbose=verbose,
                plot=False
            )
            step_count += 1

        # ========== 运行 PPO 控制策略 ==========
        if ppo_agent is not None and ppo_model_path is not None:
            print(f"\n【步骤 {step_count}/?】运行 PPO 控制策略...")
            ppo_agent.load_model(ppo_model_path)
            ppo_agent.actor.eval()

            results['ppo'] = self.run_with_ppo_control(
                ppo_agent=ppo_agent,
                model_path=ppo_model_path,
                simulation_time=simulation_time,
                verbose=verbose,
                plot=False
            )
            step_count += 1

        # ========== 运行固定控制策略 ==========
        print(f"\n【步骤 {step_count}/?】运行固定控制策略...")
        results['fixed'] = self.run_without_control(
            simulation_time=simulation_time,
            fixed_u_12=fixed_u_12,
            fixed_u_21=fixed_u_21,
            verbose=verbose,
            plot=False
        )

        # ========== 计算对比指标 ==========
        comparison = {}

        # 计算各策略相对于固定控制的改进
        for algo in ['ddpg', 'ppo']:
            if algo in results:
                algo_stats = results[algo]['statistics']
                fixed_stats = results['fixed']['statistics']

                comparison[algo] = {
                    'completion_rate_improvement': (
                            (algo_stats['completion_rate'] - fixed_stats['completion_rate']) /
                            (fixed_stats['completion_rate'] + 1e-6) * 100
                    ),
                    'total_completion_improvement': (
                            (algo_stats['total_completion'] - fixed_stats['total_completion']) /
                            (fixed_stats['total_completion'] + 1e-6) * 100
                    ),
                    'avg_n1_improvement': (
                            (fixed_stats['avg_n1'] - algo_stats['avg_n1']) /
                            (fixed_stats['avg_n1'] + 1e-6) * 100
                    ),
                    'avg_n2_improvement': (
                            (fixed_stats['avg_n2'] - algo_stats['avg_n2']) /
                            (fixed_stats['avg_n2'] + 1e-6) * 100
                    ),
                    'vht_improvement': None
                }

                # 计算 VHT 改进
                algo_vht = sum(np.array(results[algo]['time_series']['n1']) +
                               np.array(results[algo]['time_series']['n2'])) * self.dt / 3600
                fixed_vht = sum(np.array(results['fixed']['time_series']['n1']) +
                                np.array(results['fixed']['time_series']['n2'])) * self.dt / 3600
                comparison[algo]['vht_improvement'] = (fixed_vht - algo_vht) / (fixed_vht + 1e-6) * 100

        results['comparison'] = comparison

        # ========== 绘制综合对比图表 ==========
        # ========== 绘制综合对比图表 ==========
        if plot:
            try:
                import matplotlib.pyplot as plt

                time_array = np.array(results['fixed']['time_series']['t']) / 3600

                # 确定有哪些算法参与对比
                algorithms = ['fixed']
                algorithm_names = {'fixed': 'Fixed Control'}
                colors = {'fixed': 'blue'}
                linestyles = {'fixed': '--'}

                if 'ddpg' in results:
                    algorithms.append('ddpg')
                    algorithm_names['ddpg'] = 'DDPG'
                    colors['ddpg'] = 'red'
                    linestyles['ddpg'] = '-'

                if 'ppo' in results:
                    algorithms.append('ppo')
                    algorithm_names['ppo'] = 'PPO'
                    colors['ppo'] = 'green'
                    linestyles['ppo'] = '-.'

                # 创建图形 (3x4 布局)
                fig = plt.figure(figsize=(24, 14))
                gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3,
                                      width_ratios=[1.0, 1.0, 0.9, 0.9])

                # 子图 1：累计完成量对比 (占据左上 2x2 区域)
                ax1 = fig.add_subplot(gs[:2, :2])
                for algo in algorithms:
                    cumulative = (np.cumsum(results[algo]['time_series']['M11']) +
                                  np.cumsum(results[algo]['time_series']['M22'])) * self.dt
                    ax1.plot(time_array, cumulative, color=colors[algo],
                             linestyle=linestyles[algo], linewidth=2.5,
                             label=algorithm_names[algo], alpha=0.8)

                ax1.set_xlabel('Time (hour)', fontsize=12)
                ax1.set_ylabel('Cumulative Completion (veh)', fontsize=12)
                ax1.set_title('Cumulative Vehicle Completions Comparison', fontsize=14)
                ax1.legend(loc='best', fontsize=11)
                ax1.grid(True, alpha=0.3)

                # 子图 2：区域累积量 n1 对比 (右侧上方)
                ax2 = fig.add_subplot(gs[0, 2])
                for algo in algorithms:
                    ax2.plot(time_array, results[algo]['time_series']['n1'],
                             color=colors[algo], linestyle=linestyles[algo],
                             linewidth=2, label=algorithm_names[algo], alpha=0.8)
                ax2.set_xlabel('Time (hour)', fontsize=12)
                ax2.set_ylabel('Accumulation in Region 1 (veh)', fontsize=12)
                ax2.set_title('Vehicle Accumulation - Region 1', fontsize=13)
                ax2.legend(loc='best', fontsize=9)
                ax2.grid(True, alpha=0.3)

                # 子图 3：区域累积量 n2 对比 (右侧第二列)
                ax3 = fig.add_subplot(gs[1, 2])
                for algo in algorithms:
                    ax3.plot(time_array, results[algo]['time_series']['n2'],
                             color=colors[algo], linestyle=linestyles[algo],
                             linewidth=2, label=algorithm_names[algo], alpha=0.8)
                ax3.set_xlabel('Time (hour)', fontsize=12)
                ax3.set_ylabel('Accumulation in Region 2 (veh)', fontsize=12)
                ax3.set_title('Vehicle Accumulation - Region 2', fontsize=13)
                ax3.legend(loc='best', fontsize=9)
                ax3.grid(True, alpha=0.3)

                # 子图 4：u_12 控制率对比 (右侧第三列上方)
                ax4 = fig.add_subplot(gs[0, 3])
                rl_algos = [a for a in algorithms if a != 'fixed']
                for algo in rl_algos:
                    ax4.plot(time_array, results[algo]['time_series']['u_12'],
                             color=colors[algo], linestyle=linestyles[algo],
                             linewidth=2, label=f'{algorithm_names[algo]} - u₁₂', alpha=0.8)

                # 固定控制作为基线
                ax4.axhline(y=fixed_u_12, color='gray', linestyle=':',
                            linewidth=2, label=f'Fixed - u₁₂ ({fixed_u_12:.2f})', alpha=0.6)

                ax4.set_xlabel('Time (hour)', fontsize=12)
                ax4.set_ylabel('Control Coefficient u₁₂', fontsize=12)
                ax4.set_title('Control Rate u₁₂ Comparison', fontsize=13)
                ax4.legend(loc='best', fontsize=8)
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1.0)

                # 子图 5：u_21 控制率对比 (右侧第三列下方) - 新增
                ax5 = fig.add_subplot(gs[1, 3])
                for algo in rl_algos:
                    ax5.plot(time_array, results[algo]['time_series']['u_21'],
                             color=colors[algo], linestyle=linestyles[algo],
                             linewidth=2, label=f'{algorithm_names[algo]} - u₂₁', alpha=0.8)

                # 固定控制作为基线
                ax5.axhline(y=fixed_u_21, color='gray', linestyle=':',
                            linewidth=2, label=f'Fixed - u₂₁ ({fixed_u_21:.2f})', alpha=0.6)

                ax5.set_xlabel('Time (hour)', fontsize=12)
                ax5.set_ylabel('Control Coefficient u₂₁', fontsize=12)
                ax5.set_title('Control Rate u₂₁ Comparison', fontsize=13)
                ax5.legend(loc='best', fontsize=8)
                ax5.grid(True, alpha=0.3)
                ax5.set_ylim(0, 1.0)

                # 子图 6：最终完成量柱状图 (底部左侧)
                ax6 = fig.add_subplot(gs[2, 0:2])
                bar_width = 0.13
                x_positions = [0.5, 1.5]

                for i, algo in enumerate(algorithms):
                    offset = (i - len(algorithms) / 2 + 0.5) * bar_width
                    M11_final = np.sum(results[algo]['time_series']['M11']) * self.dt
                    M22_final = np.sum(results[algo]['time_series']['M22']) * self.dt

                    ax6.bar(x_positions[0] + offset, M11_final, bar_width,
                            color=colors[algo], alpha=0.7,
                            label=f'{algorithm_names[algo]} - M11' if i == 0 else '')
                    ax6.bar(x_positions[1] + offset, M22_final, bar_width,
                            color=colors[algo], alpha=0.7, hatch='//')

                    # 在柱子上方标注数值
                    ax6.text(x_positions[0] + offset, M11_final, f'{M11_final:.0f}',
                             ha='center', va='bottom', fontsize=9, fontweight='bold')
                    ax6.text(x_positions[1] + offset, M22_final, f'{M22_final:.0f}',
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

                ax6.set_xticks(x_positions)
                ax6.set_xticklabels(['M11', 'M22'], fontsize=11)
                ax6.set_ylabel('Final Completion (veh)', fontsize=11)
                ax6.set_title('Final Completion by Region', fontsize=12)
                ax6.legend(loc='upper left', fontsize=9)
                ax6.grid(True, alpha=0.3, axis='y')

                # 子图 7：关键指标对比（多组柱状图）(底部中间) - 使用双 Y 轴
                ax7 = fig.add_subplot(gs[2, 2:4])
                ax7_twin = ax7.twinx()  # 创建共享 X 轴的双 Y 轴

                metrics = ['Completion\nRate (%)', 'Avg N1\n(veh)', 'Avg N2\n(veh)',
                           'Max N1\n(veh)', 'Max N2\n(veh)']

                x = np.arange(len(metrics))
                bar_width = 0.2

                # 分离数据：第一组（完成率）用左轴，其余四组用右轴
                for i, algo in enumerate(algorithms):
                    stats = results[algo]['statistics']
                    values = [
                        stats['completion_rate'] * 100,  # 左轴：完成率 (%)
                        stats['avg_n1'],  # 右轴：平均 n1
                        stats['avg_n2'],  # 右轴：平均 n2
                        stats['max_n1'],  # 右轴：最大 n1
                        stats['max_n2']  # 右轴：最大 n2
                    ]

                    # 计算偏移位置
                    offset = i * bar_width - len(algorithms) * bar_width / 2 + bar_width / 2

                    # 第一组数据（完成率）画在左轴
                    ax7.bar(x[0] + offset, values[0], bar_width,
                            label=algorithm_names[algo] if i == 0 else '',
                            color=colors[algo], alpha=0.7, edgecolor='black', linewidth=1.5)

                    # 后四组数据画在右轴（带斜线填充区分）
                    for j in range(1, 5):
                        ax7_twin.bar(x[j] + offset, values[j], bar_width,
                                     color=colors[algo], alpha=0.5, hatch='///')

                # 设置左 Y 轴标签和范围
                ax7.set_ylabel('Completion Rate (%)', fontsize=12, color='black', fontweight='bold')
                ax7.set_ylim(0, max(results[algo]['statistics']['completion_rate'] for algo in algorithms) * 100 * 1.2)

                # 设置右 Y 轴标签
                max_vehicle_value = max(
                    max(results[algo]['statistics']['avg_n1'],
                        results[algo]['statistics']['avg_n2'],
                        results[algo]['statistics']['max_n1'],
                        results[algo]['statistics']['max_n2'])
                    for algo in algorithms
                )
                ax7_twin.set_ylabel('Vehicle Accumulation (veh)', fontsize=12, color='black', fontweight='bold')
                ax7_twin.set_ylim(0, max_vehicle_value * 1.2)

                # 设置标题和 X 轴
                ax7.set_title('Key Performance Metrics (Dual Y-axis)', fontsize=13)
                ax7.set_xticks(x)
                ax7.set_xticklabels(metrics, fontsize=10)

                # 添加网格
                ax7.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
                ax7_twin.grid(False)  # 右轴不显示网格

                # 在柱子上标注数值
                # 左轴：完成率
                for i, algo in enumerate(algorithms):
                    stats = results[algo]['statistics']
                    completion_rate = stats['completion_rate'] * 100
                    offset = i * bar_width - len(algorithms) * bar_width / 2 + bar_width / 2
                    ax7.text(x[0] + offset, completion_rate, f'{completion_rate:.1f}%',
                             ha='center', va='bottom', fontsize=8, fontweight='bold')

                # 右轴：车辆相关指标
                for i, algo in enumerate(algorithms):
                    stats = results[algo]['statistics']
                    vehicle_values = [stats['avg_n1'], stats['avg_n2'], stats['max_n1'], stats['max_n2']]
                    offset = i * bar_width - len(algorithms) * bar_width / 2 + bar_width / 2

                    for j, val in enumerate(vehicle_values):
                        ax7_twin.text(x[j + 1] + offset, val, f'{val:.0f}',
                                      ha='center', va='bottom', fontsize=7)

                plt.suptitle(
                    f'Multi-Algorithm Comparison: DDPG vs PPO vs Fixed Control '
                    f'(Simulation Time: {simulation_time / 3600:.1f} hours)',
                    fontsize=16, y=1.02)

                # 保存图片
                save_path = f"results_multi_algo_comparison_ddpg_ppo_fixed_u12_{fixed_u_12}_u21_{fixed_u_21}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✅ 多算法对比图已保存至：{save_path}")

                # 打印对比摘要
                print("\n" + "=" * 80)
                print("多算法性能对比摘要")
                print("=" * 80)

                header = f"{'指标':<28}"
                for algo in algorithms:
                    header += f"{algorithm_names[algo]:>16}"
                print(header)
                print("-" * 80)

                metrics_list = [
                    ('完成率 (%)', 'completion_rate', lambda x: x * 100),
                    ('总完成量 (veh)', 'total_completion', lambda x: x),
                    ('平均 n1 (veh)', 'avg_n1', lambda x: x),
                    ('平均 n2 (veh)', 'avg_n2', lambda x: x),
                    ('最大 n1 (veh)', 'max_n1', lambda x: x),
                    ('最大 n2 (veh)', 'max_n2', lambda x: x)
                ]

                for metric_name, metric_key, func in metrics_list:
                    row = f"{metric_name:<28}"
                    for algo in algorithms:
                        value = func(results[algo]['statistics'][metric_key])
                        row += f"{value:>16.1f}"
                    print(row)

                print("=" * 80)

                # 打印改进百分比
                if 'ddpg' in results or 'ppo' in results:
                    print("\n相对于 Fixed Control 的性能提升 (%)")
                    print("-" * 80)
                    improvement_header = f"{'改进指标':<28}"
                    if 'ddpg' in results:
                        improvement_header += f"{'DDPG':>16}"
                    if 'ppo' in results:
                        improvement_header += f"{'PPO':>16}"
                    print(improvement_header)
                    print("-" * 80)

                    improvements = [
                        ('完成率', 'completion_rate_improvement'),
                        ('总完成量', 'total_completion_improvement'),
                        ('平均 n1 降低', 'avg_n1_improvement'),
                        ('平均 n2 降低', 'avg_n2_improvement'),
                        ('VHT 降低', 'vht_improvement')
                    ]

                    for name, key in improvements:
                        row = f"{name:<28}"
                        for algo in ['ddpg', 'ppo']:
                            if algo in comparison:
                                row += f"{comparison[algo][key]:>+15.2f}"
                        print(row)
                    print("=" * 80)

                plt.show()

            except ImportError:
                print("⚠️  警告：matplotlib 未安装，无法绘制图表")
            except Exception as e:
                print(f"⚠️  绘图时出错：{e}")

        return results

if  __name__ == '__main__':
    # pass
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    config_ddpg = {
        "sumocfg": "test_validation_rl.sumocfg",
        "net_file": "grid_network.net.xml",

        "simulation_steps": 60,
        "num_workers": 32,

        "control_period": 60,
        "yellow_time": 3,

        "noise_base": 0.3,
        "noise_decay": 0.997,

        "tau": 0.01,
        'sample_size': 3000,
        "max_trajectory_steps": 10000,

        "max_iterations": 3000,
        "DDPG_epochs": 128,
        'actor_update_freq': 64,
        "batch_size": 256,
        "seed": 42,

        "hidden_dim": 16,
        "lr_actor": 0.0025,
        "lr_critic": 0.001,
        "lr_decay_actor": 0.93,
        "lr_decay_critic": 0.98,

        'entropy_coef': 0.01,
        "vf_clip_coef": 0.5,
        "clip_coef": 0.2,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "grad_clip": 0.5,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "save_model": True,
        "pretrained_path_prefix": "./model_param_DDPG/multi_junction_ddpg",
        "base_port": 8020,
    }
    config_ppo = {
        "sumocfg": "test_validation_rl.sumocfg",
        "net_file": "grid_network.net.xml",

        "simulation_steps": 60,
        "num_workers": 32,

        "control_period": 60,
        "yellow_time": 3,

        "max_iterations": 3000,
        "ppo_epochs": 20,
        "batch_size": 32,
        "seed": 111,

        "hidden_dim": 16,
        "lr_actor": 4e-3,
        "lr_critic": 4e-3,
        "lr_decay": 0.98,

        'kl_threshold': 100,
        'entropy_coef': 0.01,
        "vf_clip_coef": 0.5,
        "clip_coef": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "grad_clip": 0.5,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "save_model": True,
        "pretrained_path_prefix": "./model_param_PPO/multi_junction_ddpg",
        "base_port": 8020,
    }
    model = TwoRegionTrafficModel()
    # model.run_without_control(simulation_time=3600.0, fixed_u_12=0.9, fixed_u_21=0.9, verbose=True)
    # model.run_with_ddpg_control(DDPGAgent(
    #                                 state_dim=8,
    #                                 action_dim=2,
    #                                 config=config,
    #                                 mode='test',
    #                                 num_junctions=2
    #                             ),
    #                         model_path=r'F:\sumo\network_file\小型城市路网\model_param_DDPG\multi_junction_ddpg_iter25.pth',
    #                         simulation_time= 3600.0)

    # 运行对比实验
    model.run_multi_algorithm_comparison(
        # DDPG 配置
        ddpg_agent=DDPGAgent(
            state_dim=8,
            action_dim=2,
            config=config_ddpg,
            mode='test',
            num_junctions=2
        ),
        ddpg_model_path=r'F:\sumo\network_file\小型城市路网\model_param_DDPG\multi_junction_ddpg_iter25.pth',

        # PPO 配置
        ppo_agent=PPOAgent(
            state_dim=8,
            config=config_ppo,
            mode='test',
            num_junctions=2
        ),
        ppo_model_path=r'F:\sumo\network_file\小型城市路网\model_param_PPO\multi_junction_ddpg_iter31.pth',

        # 固定控制配置
        fixed_u_12=0.9,
        fixed_u_21=0.9,

        # 通用配置
        simulation_time=3600.0,
        verbose=False,
        plot=True
    )
