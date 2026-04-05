import sys
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")
from util.RL.GYM_DDPG_S_mau.RLTrainer import RLTrainer
# 使用示例
if __name__ == "__main__":
    import torch

    # Windows 系统保护
    if sys.platform == 'win32':
        from multiprocessing import freeze_support

        freeze_support()

    # 示例配置
    config = {
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

        "max_iterations": 1000,
        "DDPG_epochs": 128,
        'actor_update_freq': 64,
        "batch_size": 256,
        "seed": 52,

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

    # 创建训练器（自动创建并行环境和智能体）
    trainer = RLTrainer(config)

    training_history = []
    history_save_path = "./results_rl_DDPG/training_history.pkl"
    json_save_path = "./results_rl_DDPG/training_history.json"
    test_save_path = "./results_rl_DDPG/test_results.json"
    test_results = []
    test_interval = 5  # 每 5 轮测试一次
    best_tts_mean = float('inf')  # 记录历史最小的 tts_mean
    # current_tts_mean = float('inf')  # 记录历史最小的 tts_mean
    try:
        # 训练循环
        for iter in range(config['max_iterations']):
            # 一体化训练（采样 + 训练）
            stats = trainer.train_policy(iter)
            training_history.append(stats)

            # 计算当前 tts_mean
            current_tts_mean = float(np.mean(stats['tts']) / 3600) if stats['tts'] else float('inf')
            # 每 10 次迭代保存一次，或最后一次迭代时保存
            save_interval = 1
            if (iter + 1) % save_interval == 0 or iter == config['max_iterations'] - 1:
                # 确保目录存在
                os.makedirs(os.path.dirname(history_save_path), exist_ok=True)

                # 保存为 pickle 格式（保留完整数据结构）
                with open(history_save_path, 'wb') as f:
                    pickle.dump(training_history, f)

                # 同时保存为 JSON 格式（方便查看）
                history_json = []
                for record in training_history:
                    json_record = {
                        'iteration': record['iteration'],
                        'samples_collected': record['samples_collected'],
                        'episode_rewards_mean': float(np.mean(record['episode_rewards'])) if record[
                            'episode_rewards'] else None,
                        'raw_rewards_mean': float(np.mean(record['raw_rewards'])) if record[
                            'raw_rewards'] else None,
                        'tts_mean': float(np.mean(record['tts']) / 3600) if record[
                            'tts'] else None,
                        'CTC_mean': float((np.sum(record['M11_LIST'])+np.sum(record['M22_LIST']))/config['num_workers']) if record[
                            'M11_LIST'] and record['M22_LIST'] else None,
                        'episode_rewards_std': float(np.std(record['episode_rewards'])) if record[
                            'episode_rewards'] else None,
                        'episode_lengths_mean': float(np.mean(record['episode_lengths'])) if record[
                            'episode_lengths'] else None,
                        'episode_times_mean': float(np.mean(record['episode_times'])) if record[
                            'episode_times'] else None,
                        'train_stats': {
                            'actor_loss': float(record['train_stats'].get('actor_loss', 0)),
                            'critic_loss': float(record['train_stats'].get('critic_loss', 0)),
                            'entropy_loss': float(record['train_stats'].get('entropy_loss', 0)),
                            'clip_ratio': float(record['train_stats'].get('clip_ratio', 0))
                        }
                    }
                    history_json.append(json_record)

                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(history_json, f, indent=2, ensure_ascii=False)

                print(f"   📁 训练历史已保存 ({len(training_history)} 条记录)")

            # ========== 新增：每 5 轮测试一次 ==========
            if (iter) % test_interval == 0 or iter == config['max_iterations'] - 1 or current_tts_mean < best_tts_mean:
                if current_tts_mean < best_tts_mean:
                    best_tts_mean = current_tts_mean
                    print(f"   🏆 当前最佳 TTS_mean：{best_tts_mean:.2f} 小时")
                print(f"\n🧪 开始第 {iter} 轮测试...")

                # 执行测试（不渲染 GUI）
                test_rewards, test_actions, test_tts, ctc  = trainer.test(render=False)

                # 记录测试结果
                test_result = {
                    'iteration': iter,
                    'total_reward': float(np.sum(test_rewards)),
                    'tts_mean': float(np.mean(test_tts) / 3600),
                    'ctc': float(ctc),
                    'avg_reward': float(np.mean(test_rewards)),
                    'std_reward': float(np.std(test_rewards)),
                    'max_reward': float(np.max(test_rewards)),
                    'min_reward': float(np.min(test_rewards)),
                    'num_steps': len(test_rewards)
                }
                test_results.append(test_result)

                # 保存测试结果
                with open(test_save_path, 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, indent=2, ensure_ascii=False)
                print(f"\n   💾 测试结果已保存至：{test_save_path}")
            # ========================================

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")

    finally:
        # 确保资源被正确释放
        trainer.close()