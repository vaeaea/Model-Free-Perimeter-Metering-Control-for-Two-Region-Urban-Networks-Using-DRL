import torch
import numpy as np
# ==================== 轨迹缓冲区（模仿预分配数组写法重构） ====================
class RolloutBuffer(object):
    def __init__(self, num_env: int, steps: int, state_shape: tuple, action_shape: tuple, device):
        self.steps = steps
        self.device = device

        # 预分配numpy数组，避免动态扩容，提升效率
        self.state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.action = np.zeros((steps, num_env) + action_shape, dtype=np.float32)
        self.log_prob = np.zeros((steps, num_env), dtype=np.float32)
        self.next_state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.reward = np.zeros((steps, num_env) , dtype=np.float32)
        self.done = np.zeros((steps, num_env), dtype=np.float32)

        self.ptr = 0

    def push(self, state, action, log_prob, reward, next_state, done):
        """存入一步经验，写入预分配数组的对应位置"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.log_prob[self.ptr] = log_prob
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.steps

    def pull(self):
        """一次性拉取所有数据，转换为Tensor并放到目标设备"""
        return (
            torch.tensor(self.state[:self.ptr], dtype=torch.float32).to(self.device),
            torch.tensor(self.action[:self.ptr], dtype=torch.float32).to(self.device),
            torch.tensor(self.log_prob[:self.ptr], dtype=torch.float32).to(self.device),
            torch.tensor(self.reward[:self.ptr], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state[:self.ptr], dtype=torch.float32).to(self.device),
            torch.tensor(self.done[:self.ptr], dtype=torch.float32).to(self.device)
        )

    @property
    def full(self):
        """判断缓冲区是否已满"""
        return self.ptr == 0

    def clear(self):
        """清空缓冲区，重置指针"""
        self.ptr = 0
        # 重置数组内容（可选，也可以直接覆盖）
        self.state.fill(0)
        self.action.fill(0)
        self.log_prob.fill(0)
        self.reward.fill(0)
        self.next_state.fill(0)
        self.done.fill(0)

    def __len__(self) -> int:
        """当前已存储的步数"""
        return self.ptr if self.ptr != 0 else self.steps