import argparse
import numpy as np

parser = argparse.ArgumentParser(description='test')

# Environment related parameters
parser.add_argument("--server_cache_size", type=float, default=30, help="Cache size of the BS in Mb")
parser.add_argument("--server_frequency", type=float, default=3e10, help="CPU frequency of the BS: 30GHz")
parser.add_argument("--server_fc_alloc", type=float, default=3e9, help="CPU frequency allocated to users: 3GHz")
parser.add_argument("--user_num", type=int, default=10, help="Number of users")
parser.add_argument("--user_frequency", type=float, default=1e9, help="CPU frequency of a user: 1GHz")
parser.add_argument("--user_e_coe", type=float, default=5e-27, help="Energy consumption coefficient of users")
parser.add_argument("--server_e_coe", type=float, default=1e-29, help="Energy consumption coefficient of the BS")
parser.add_argument("--user_power", type=int, default=1000, help="Transmission power of an user in mW")
parser.add_argument("--task_num", type=int, default=40, help="Number of task types")
parser.add_argument("--task_min", type=int, default=20, help="The minimum task size in Mb")
parser.add_argument("--task_max", type=int, default=60, help="The maximum task size in Mb")
parser.add_argument("--task_result_min", type=int, default=1, help="The minimum result size in Mb")
parser.add_argument("--task_result_max", type=int, default=5, help="The maximum result size in Mb")
parser.add_argument("--task_computation_load", type=int, default=1e8, help="Computation required per bit of the task：100cycles/bit")
parser.add_argument("--s_slot_tau", type=int, default=1, help="The length of a small time slot in s")
parser.add_argument("--s_slot_num", type=int, default=50, help="The number of small time slots in each large time slot")
parser.add_argument("--bandwidth", type=int, default=10, help="Communication bandwidth in MHz")
parser.add_argument("--BS_pos", type=int, default=500, help="BS coordinates: (500, 500)")
parser.add_argument("--sigma2", type=float, default=1e-10, help="Noise power in mW")
parser.add_argument("--zipf_alpha", type=float, default=0.7, help="Zipf distribution parameter alpha")
parser.add_argument("--zipf_R", type=float, default=0.1, help="the probability of no request")
parser.add_argument("--zipf_N", type=float, default=3, help="The request may be transferred to N tasks adjacent to index")
parser.add_argument("--weight_user_ec", type=float, default=1, help="Energy consumption cost weight of users")
parser.add_argument("--weight_BS_ec", type=float, default=0.01, help="Energy consumption cost weight of the BS")
parser.add_argument("--probability_bound", type=float, default=0.05, help="Delay violation probability bound")
parser.add_argument("--tolerant_latency", type=float, default=0.2, help="the maximum tolerable delay")

# Experiment related parameters
parser.add_argument("--max_action", type=float, default=0.999999, help="The maximum value that an action can take")
parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--expl_noise", default=0.05)               # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0)                    # Discount factor
parser.add_argument("--tau", default=0.005)                     # Target network update rate
parser.add_argument("--policy_noise", default=0.1)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.2)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
parser.add_argument("--save_freq", default=100, type=int)       # Save results every 100 episodes
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
