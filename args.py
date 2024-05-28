import argparse
import numpy as np

parser = argparse.ArgumentParser(description='test')

# 环境相关参数
parser.add_argument("--server_cache_size", type=float, default=30, help="服务器的缓存大小，单位为Mb")       # 超过限制则给惩罚
parser.add_argument("--server_frequency", type=float, default=3e10, help="服务器的CPU频率：30GHz")
parser.add_argument("--server_fc_alloc", type=float, default=3e9, help="服务器分配给用户的CPU频率：3GHz")   # 分给每个用户3GHz，总分配超过24GHz则给惩罚
parser.add_argument("--user_num", type=int, default=10, help="用户数量")
parser.add_argument("--user_frequency", type=float, default=1e9, help="用户的CPU频率：1GHz")
parser.add_argument("--user_e_coe", type=float, default=5e-27, help="用户能耗系数")
parser.add_argument("--server_e_coe", type=float, default=1e-29, help="服务器能耗系数")
parser.add_argument("--user_power", type=int, default=1000, help="用户发送功率，单位为mW")
parser.add_argument("--task_num", type=int, default=40, help="任务种类数量")
parser.add_argument("--task_min", type=int, default=20, help="任务大小最小值，单位为Mb")
parser.add_argument("--task_max", type=int, default=60, help="任务大小最大值，单位为Mb")
# parser.add_argument("--task_size", type=int, default=40, help="任务大小，单位为Mb")
parser.add_argument("--task_result_min", type=int, default=1, help="任务计算结果大小最小值，单位为Mb")
parser.add_argument("--task_result_max", type=int, default=5, help="任务计算结果大小最大值，单位为Mb")
parser.add_argument("--task_computation_load", type=int, default=1e8, help="任务每比特需要的计算量：100cycles/bit")
parser.add_argument("--s_slot_tau", type=int, default=1, help="小时隙长度，单位为s")
parser.add_argument("--s_slot_num", type=int, default=50, help="每个大时隙中的小时隙个数")
parser.add_argument("--bandwidth", type=int, default=10, help="通信带宽，单位为MHz")
parser.add_argument("--BS_pos", type=int, default=500, help="基站坐标：(500, 500)")
parser.add_argument("--sigma2", type=float, default=1e-10, help="噪声功率，单位为mW")
parser.add_argument("--zipf_alpha", type=float, default=0.7, help="Zipf分布参数alpha")
parser.add_argument("--zipf_R", type=float, default=0.1, help="表示不请求的概率")
parser.add_argument("--zipf_N", type=float, default=3, help="表示请求可能转移到index相邻的N个任务")
parser.add_argument("--weight_user_ec", type=float, default=1, help="用户的能耗成本权重")
parser.add_argument("--weight_BS_ec", type=float, default=0.01, help="基站的能耗成本权重")
parser.add_argument("--probability_bound", type=float, default=0.05, help="时延违反概率阈值")
parser.add_argument("--tolerant_latency", type=float, default=0.2, help="最大可容忍时延dmax")

# 实验相关参数（基本都是TD3在用）
parser.add_argument("--algorithm", type=str, default="TD3")
parser.add_argument("--max_action", type=float, default=0.999999, help="动作能取到的最大值")


parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.05)               # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0)                    # Discount factor
parser.add_argument("--tau", default=0.005)                     # Target network update rate
parser.add_argument("--policy_noise", default=0.1)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.2)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--save_freq", default=100, type=int)       # Save results every 100 episodes

parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
