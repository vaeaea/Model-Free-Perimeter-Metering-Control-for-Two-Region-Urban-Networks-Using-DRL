"""
两区域宏观交通流量转移模型
灵感来源于 C-D-RL-main 的 rlplant.py，纯交通流仿真模型
可嵌入到任意 RL 框架中使用
"""
import numpy as np
import random
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


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

if  __name__ == '__main__':
    model = TwoRegionTrafficModel()
    model.run_without_control(simulation_time=5400.0, fixed_u_12=0.9, fixed_u_21=0.9, verbose=True)
